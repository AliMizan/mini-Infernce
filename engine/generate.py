"""Generate with the paged cache. Greedy by default, optional sampling and streaming.

Examples
--------
python -m engine.generate --prompt "def add(a, b):" --max-new 32
python -m engine.generate --prompt "def add(a, b):" --max-new 16 --check --kv-dtype float16
python -m engine.generate --prompt "Write a fizzbuzz in Python" --max-new 200 --temperature 0.7 --top-p 0.9
python -m engine.generate --prompt "Write a fizzbuzz in Python" --temperature 0.7 --seed 1 --no-stream
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def _dtype(name: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def load_hf(model_id: str, device: str, compute_dtype: str, load_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    kwargs = {"trust_remote_code": True}
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_dtype(compute_dtype),
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = _dtype(compute_dtype)

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    if not load_4bit:
        model.to(device)
    return model, tok


class Streamer:
    """Prints new text as tokens arrive.

    Decoding one token at a time breaks multi-byte characters and spacing, so we
    decode all generated ids each time and print only the new part. If the text
    ends in a broken character, we wait for the next token.
    """

    def __init__(self, tok):
        self.tok = tok
        self.ids = []
        self.printed = ""

    def _flush(self, hold_broken: bool) -> None:
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        if hold_broken and text.endswith("\ufffd"):
            return
        print(text[len(self.printed):], end="", flush=True)
        self.printed = text

    def __call__(self, token_id: int) -> None:
        self.ids.append(token_id)
        self._flush(hold_broken=True)

    def finish(self) -> None:
        self._flush(hold_broken=False)
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paged-cache generate")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="def add(a, b):")
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--kv-dtype", default="int8", choices=["int8", "float16", "bfloat16", "float32"])
    parser.add_argument("--compute-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument("--top-k", type=int, default=0, help="0 = off")
    parser.add_argument("--top-p", type=float, default=1.0, help="1.0 = off")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="1.0 = off")
    parser.add_argument("--seed", type=int, default=None, help="make sampling repeatable")
    parser.add_argument("--no-stream", action="store_true", help="print the whole answer at the end")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare token ids with HuggingFace generate (use --kv-dtype float16, greedy)",
    )
    args = parser.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("Install PyTorch first: pip install torch transformers", file=sys.stderr)
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} model={args.model} kv={args.kv_dtype} "
        f"temperature={args.temperature} top_k={args.top_k} top_p={args.top_p} "
        f"rep_penalty={args.repetition_penalty}"
    )

    from engine.config import EngineConfig
    from engine.kv_cache import PagedKVCache
    from engine.model import PagedQwenEngine

    model, tok = load_hf(args.model, device, args.compute_dtype, args.load_4bit)
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

    messages = [{"role": "user", "content": args.prompt}]
    if getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = args.prompt
    inputs = tok(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(cache.device)

    stream = not args.no_stream
    streamer = Streamer(tok) if stream else None
    if stream:
        print("--- output ---")

    t0 = time.perf_counter()
    out = engine.generate(
        input_ids,
        max_new_tokens=args.max_new,
        eos_token_id=tok.eos_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        on_token=streamer,
    )
    dt = time.perf_counter() - t0
    if stream:
        streamer.finish()
        print("--------------")
    n_new = out.shape[1] - input_ids.shape[1]
    print(f"paged {n_new} new tokens in {dt:.2f}s ({max(n_new, 1) / dt:.1f} tok/s)")
    if not stream:
        print(tok.decode(out[0], skip_special_tokens=True))
    print("cache", cache.memory_bytes())

    if args.check:
        if args.kv_dtype == "int8":
            print(
                "warning: INT8 KV will not match HuggingFace exactly; "
                "rerun with --kv-dtype float16 for a strict check",
                file=sys.stderr,
            )
        if args.temperature > 0:
            print(
                "warning: sampling is random, so token ids will not match "
                "HuggingFace's greedy output; use --temperature 0 for the check",
                file=sys.stderr,
            )
        t1 = time.perf_counter()
        ref = model.generate(
            input_ids,
            max_new_tokens=args.max_new,
            do_sample=False,
            use_cache=True,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        dt_ref = time.perf_counter() - t1
        print(f"hf    {ref.shape[1] - input_ids.shape[1]} new tokens in {dt_ref:.2f}s")
        print(tok.decode(ref[0], skip_special_tokens=True))
        match = torch.equal(out[:, : min(out.shape[1], ref.shape[1])], ref[:, : min(out.shape[1], ref.shape[1])])
        if match and out.shape == ref.shape:
            print("check: PASS (token ids identical)")
            return 0
        # compare prefix until first mismatch
        n = min(out.shape[1], ref.shape[1])
        same = int((out[0, :n] == ref[0, :n]).sum().item())
        print(f"check: FAIL identical_prefix={same}/{n} out_len={out.shape[1]} hf_len={ref.shape[1]}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())