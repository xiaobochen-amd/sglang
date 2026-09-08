"""The sampler's wide-row softmax must match torch and keep in-place semantics."""

import pytest
import torch

from sglang.srt.layers.sampler import _WIDE_SOFTMAX_MIN_VOCAB, _softmax_wide

VOCAB = 154880


@pytest.mark.parametrize("rows", [1, 6, 8, 24, 48, 192])
def test_matches_torch_on_wide_rows(rows):
    torch.manual_seed(rows)
    logits = torch.randn(rows, VOCAB, device="cuda", dtype=torch.float32) * 3
    ref = torch.softmax(logits, dim=-1)
    torch.testing.assert_close(_softmax_wide(logits), ref, atol=2e-6, rtol=0)


def test_in_place_writes_the_same_tensor():
    """The caller relies on this to avoid a second [rows, vocab] allocation."""
    logits = torch.randn(6, VOCAB, device="cuda", dtype=torch.float32)
    ref = torch.softmax(logits, dim=-1)
    got = _softmax_wide(logits, out=logits)
    assert got.data_ptr() == logits.data_ptr()
    torch.testing.assert_close(got, ref, atol=2e-6, rtol=0)


def test_narrow_rows_keep_torch():
    """Below the threshold torch already fits the row in shared memory."""
    logits = torch.randn(4, _WIDE_SOFTMAX_MIN_VOCAB - 1, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(
        _softmax_wide(logits), torch.softmax(logits, dim=-1), atol=1e-7, rtol=0
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda: torch.randn(4, VOCAB, device="cuda", dtype=torch.bfloat16),
        lambda: torch.randn(4, VOCAB, device="cuda").t().contiguous().t(),
        lambda: torch.randn(2, 4, VOCAB, device="cuda"),
    ],
    ids=["bf16", "non-contiguous", "3d"],
)
def test_shapes_the_kernel_does_not_take_fall_back(make):
    logits = make()
    torch.testing.assert_close(
        _softmax_wide(logits), torch.softmax(logits, dim=-1), atol=2e-3, rtol=0
    )


def test_extreme_logits_do_not_overflow():
    logits = torch.full((4, VOCAB), -1e4, device="cuda", dtype=torch.float32)
    logits[:, 11] = 1e4
    got = _softmax_wide(logits)
    assert torch.isfinite(got).all()
    assert torch.allclose(got[:, 11], torch.ones(4, device="cuda"))
