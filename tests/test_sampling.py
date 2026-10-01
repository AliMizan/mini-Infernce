"""Unit tests for sample_next. CPU only, no model download.

Run:
    python tests/test_sampling.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.model import sample_next


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_greedy_is_argmax() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 100)
    out = sample_next(logits, temperature=0.0)
    assert out.shape == (1, 1)
    assert int(out) == int(logits.argmax())
    print("  [greedy] temperature 0 equals argmax")


def test_top_k_one_and_tiny_top_p_are_greedy() -> None:
    torch.manual_seed(1)
    logits = torch.randn(1, 100)
    best = int(logits.argmax())
    g = _gen(0)
    for _ in range(20):
        assert int(sample_next(logits, temperature=1.0, top_k=1, generator=g)) == best
        assert int(sample_next(logits, temperature=1.0, top_p=1e-6, generator=g)) == best
    print("  [filters] top_k=1 and tiny top_p always pick the best token")


def test_top_k_restricts_choices() -> None:
    torch.manual_seed(2)
    logits = torch.randn(1, 100)
    allowed = set(logits.topk(5).indices[0].tolist())
    g = _gen(0)
    for _ in range(300):
        assert int(sample_next(logits, temperature=1.5, top_k=5, generator=g)) in allowed
    print("  [top_k] 300 draws all inside the top 5")


def test_top_p_restricts_choices() -> None:
    # probabilities 0.5, 0.3, 0.15, 0.05 ; top_p=0.7 keeps tokens 0 and 1 only
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()
    g = _gen(0)
    seen = {int(sample_next(logits, temperature=1.0, top_p=0.7, generator=g)) for _ in range(300)}
    assert seen == {0, 1}, seen
    print("  [top_p] only the smallest set covering 0.7 is used")


def test_seed_is_repeatable() -> None:
    torch.manual_seed(3)
    logits = torch.randn(1, 200)
    a = [int(sample_next(logits, temperature=1.0, generator=_gen(42))) for _ in range(1)]
    g1, g2 = _gen(7), _gen(7)
    s1 = [int(sample_next(logits, temperature=1.0, generator=g1)) for _ in range(20)]
    s2 = [int(sample_next(logits, temperature=1.0, generator=g2)) for _ in range(20)]
    assert s1 == s2
    print("  [seed] same seed gives the same 20 draws")


def test_temperature_matches_softmax() -> None:
    # true probabilities 0.25 / 0.75
    logits = torch.tensor([[0.0, math.log(3.0)]])
    g = _gen(0)
    n = 4000
    hits = sum(int(sample_next(logits, temperature=1.0, generator=g)) for _ in range(n))
    freq = hits / n
    assert abs(freq - 0.75) < 0.04, freq
    print(f"  [temperature] token 1 chosen {freq:.3f} of the time (expected 0.75)")


def test_repetition_penalty() -> None:
    logits = torch.zeros(1, 10)
    logits[0, 3] = 5.0   # best without penalty
    logits[0, 4] = 3.0   # runner up
    hist = torch.tensor([3])
    assert int(sample_next(logits, temperature=0.0)) == 3
    assert int(sample_next(logits, temperature=0.0, repetition_penalty=2.0, history=hist)) == 4
    # negative scores get pushed further down, not up
    neg = torch.zeros(1, 10)
    neg[0, 2] = -1.0
    out = sample_next(neg, temperature=0.0, repetition_penalty=2.0, history=torch.tensor([2]))
    assert int(out) != 2
    print("  [repetition] seen tokens are pushed down")


def main() -> None:
    test_greedy_is_argmax()
    test_top_k_one_and_tiny_top_p_are_greedy()
    test_top_k_restricts_choices()
    test_top_p_restricts_choices()
    test_seed_is_repeatable()
    test_temperature_matches_softmax()
    test_repetition_penalty()
    print("ok")


if __name__ == "__main__":
    main()