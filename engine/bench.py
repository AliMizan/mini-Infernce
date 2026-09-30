"""Benchmark: paged engine vs HuggingFace generate.

Examples
--------
python -m engine.bench
python -m engine.bench --prompt-len 512 --new 128 --kv-dtype float16
python -m engine.bench --kv-dtype int8

Both engines are forced to produce exactly --new tokens (no early stop),
after a warm-up run. Prefill and decode are timed separately, with a
GPU synchronize before each clock read.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Paged engine vs HF benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--new", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--kv-dtype", default="float16", choices=["int8", "float16", "bfloat16", "float32"])
    parser.add_argument("--compute-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    import torch

    from engine.config import EngineConfig
    from engine.generate import load_hf
    from engine.kv_cache import PagedKVCache
    from engine.model import PagedQwenEngine

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.prompt_len + args.new > args.max_seq:
        print("prompt-len + new must be <= max-seq", file=sys.stderr)
        return 1

    model, tok = load_hf(args.model, device, args.compute_dtype, load_4bit=False)
    cfg = EngineConfig.from_hf(
        model.config,
        device=device,
        kv_dtype=args.kv_dtype,
        compute_dtype=args.compute_dtype,
        max_seq_len=args.max_seq,
        page_size=args.page_size,
    )
    cache = PagedKVCache(cfg)
    engine = PagedQwenEngine(model, cache)
    on_gpu = cache.device.type == "cuda"

    def sync():
        if on_gpu:
            torch.cuda.synchronize()

    # Fixed-length prompt built from repeated code text (no chat template).
    text = "def add(a, b):\n    return a + b\n" * 400
    ids = tok(text, return_tensors="pt")["input_ids"][:, : args.prompt_len].to(cache.device)
    assert ids.shape[1] == args.prompt_len, "prompt too short; lower --prompt-len"

    @torch.no_grad()
    def paged_run(n_new: int):
        engine.reset()
        sync()
        t0 = time.perf_counter()
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        logits = engine.forward_tokens(ids, pos, last_only=True)
        cache.commit(engine.slot)
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        sync()
        t1 = time.perf_counter()
        for _ in range(n_new - 1):
            p = torch.tensor([[cache._len_cpu[engine.slot]]], device=ids.device)
            logits = engine.forward_tokens(nxt, p, last_only=True)
            cache.commit(engine.slot)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        sync()
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1

    @torch.no_grad()
    def hf_gen(n: int) -> float:
        sync()
        t0 = time.perf_counter()
        model.generate(
            ids,
            max_new_tokens=n,
            min_new_tokens=n,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.pad_token_id,
        )
        sync()
        return time.perf_counter() - t0

    print(f"device={device} kv={args.kv_dtype} prompt={args.prompt_len} new={args.new} runs={args.runs}")
    print("warming up...")
    paged_run(8)
    hf_gen(8)

    paged_pf, paged_dc, hf_pf, hf_dc = [], [], [], []
    n_dec = args.new - 1
    for r in range(args.runs):
        pf, dc = paged_run(args.new)
        paged_pf.append(pf)
        paged_dc.append(dc)
        t1 = hf_gen(1)
        tn = hf_gen(args.new)
        hf_pf.append(t1)
        hf_dc.append(max(tn - t1, 1e-9))
        print(f"  run {r + 1}: paged decode {n_dec / dc:.1f} tok/s | hf decode {n_dec / hf_dc[-1]:.1f} tok/s")

    med = statistics.median
    print()
    print(f"{'engine':8s} {'prefill (ms)':>14s} {'decode (tok/s)':>16s}")
    print(f"{'paged':8s} {med(paged_pf) * 1000:14.1f} {n_dec / med(paged_dc):16.1f}")
    print(f"{'hf':8s} {med(hf_pf) * 1000:14.1f} {n_dec / med(hf_dc):16.1f}")
    print()
    print("hf prefill = generate(1 token); hf decode = (generate(N) - generate(1)) / (N - 1).")
    print("These are approximate; use them to compare before/after, not as absolute numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())