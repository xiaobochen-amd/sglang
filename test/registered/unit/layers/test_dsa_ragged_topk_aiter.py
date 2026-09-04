"""RAGGED prefill top-k on aiter must match the sgl-kernel transform it replaces."""

import pytest
import torch

from sglang.srt.utils import is_hip

pytestmark = pytest.mark.skipif(not is_hip(), reason="aiter top-k is ROCm-only")

TOPK = 2048


def _refs():
    from sgl_kernel import fast_topk_transform_ragged_fused

    from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
        _aiter_ragged_topk_available,
        _topk_transform_aiter_ragged,
    )

    if not _aiter_ragged_topk_available():
        pytest.skip("aiter top_k_per_row_prefill has no stable= argument")
    return fast_topk_transform_ragged_fused, _topk_transform_aiter_ragged


@pytest.mark.parametrize(
    "rows,ctx,offset",
    [
        (1605, 335000, 700000),  # the two grids a GLM-5.2 agentic server issues
        (1991, 335000, 0),
        (203, 4000, 12345),
        (64, 1000, 0),  # rows shorter than topk take the naive emit
        (8, 512, 99),
    ],
)
def test_matches_sgl_kernel(rows, ctx, offset):
    shipped, ours = _refs()
    torch.manual_seed(rows)
    logits = torch.randn(rows, ctx, device="cuda", dtype=torch.float32)
    # causal on a cached prefix: row i sees [0, ctx - rows + i)
    lengths = (
        torch.arange(rows, device="cuda", dtype=torch.int32) + (ctx - rows)
    ).clamp_min(1)
    row_starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    offsets = torch.full((rows,), offset, device="cuda", dtype=torch.int32)

    ref = shipped(
        score=logits,
        lengths=lengths,
        topk_indices_offset=offsets,
        topk=TOPK,
        row_starts=row_starts,
    )
    got = ours(logits, lengths, offsets, TOPK, row_starts=row_starts)

    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert int((got < 0).sum()) == int((ref < 0).sum())
    # The two resolve an exact score tie differently, so compare the selected set
    # and allow the handful of rows where two positions hold the same fp32 score.
    differing = [r for r in range(rows) if set(ref[r].tolist()) != set(got[r].tolist())]
    for r in differing:
        a = sorted(set(ref[r].tolist()) - set(got[r].tolist()))
        b = sorted(set(got[r].tolist()) - set(ref[r].tolist()))
        assert torch.equal(
            logits[r, a].sort().values, logits[r, b].sort().values
        ), f"row {r} differs on positions that are not tied"


def test_deterministic():
    """Every TP rank runs this on the same logits and must select the same set."""
    _, ours = _refs()
    rows, ctx = 512, 40000
    # only 64 distinct scores, so ties are everywhere
    logits = torch.randint(0, 64, (rows, ctx), device="cuda").float()
    lengths = torch.full((rows,), ctx, device="cuda", dtype=torch.int32)
    offsets = torch.zeros(rows, device="cuda", dtype=torch.int32)
    base = ours(logits, lengths, offsets, TOPK)
    for _ in range(8):
        assert torch.equal(ours(logits, lengths, offsets, TOPK), base)
