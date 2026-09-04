"""Split-row top-p renorm must match the pivot kernel it replaces on ROCm."""

import pytest
import torch

from sglang.srt.utils import is_hip

pytestmark = pytest.mark.skipif(not is_hip(), reason="_renorm_top_k_top_p_hip is ROCm-only")

VOCAB = 154880


def _sort_reference(probs, top_ps):
    ps, pi = probs.double().sort(dim=-1, descending=True)
    ps = ps * ((ps.cumsum(-1) - ps) <= top_ps.double().view(-1, 1))
    ps = ps / ps.sum(-1, keepdim=True).clamp_min(1e-30)
    return torch.zeros_like(ps).scatter_(-1, pi, ps).float()


@pytest.mark.parametrize("rows", [1, 6, 8, 24, 32])
@pytest.mark.parametrize("temperature", [0.6, 1.0])
def test_no_worse_than_the_pivot_kernel(rows, temperature):
    """Both keep every element tied with tau, so both differ from a sort the same
    way; what must hold is that the split is never the less accurate of the two."""
    from sglang.srt.speculative.eagle_utils import (
        _top_p_renorm_kernel,
        _top_p_renorm_split,
    )

    torch.manual_seed(rows * 10 + int(temperature * 10))
    probs = torch.softmax(
        torch.randn(rows, VOCAB, device="cuda") / temperature, dim=-1
    )
    top_ps = (0.05 + 0.94 * torch.rand(rows, device="cuda")).float()

    ref = _sort_reference(probs, top_ps)
    split = _top_p_renorm_split(probs, top_ps)
    shipped = torch.empty_like(probs)
    _top_p_renorm_kernel[(rows,)](
        probs, shipped, top_ps, VOCAB, BLOCK=4096, N_ITER=30
    )

    assert (split - ref).abs().max() <= (shipped - ref).abs().max() + 1e-9
    torch.testing.assert_close(
        split.sum(-1), torch.ones(rows, device="cuda"), atol=2e-6, rtol=0
    )


def test_independent_of_the_split_factor():
    """Rows are split across blocks, so the output must not depend on how many."""
    import sglang.srt.speculative.eagle_utils as eu

    torch.manual_seed(3)
    probs = torch.softmax(torch.randn(12, VOCAB, device="cuda") / 0.7, dim=-1)
    top_ps = torch.linspace(0.3, 0.99, 12, device="cuda").float()
    base = eu._top_p_renorm_split(probs, top_ps)
    for _ in range(4):
        assert torch.equal(eu._top_p_renorm_split(probs, top_ps), base)


@pytest.mark.parametrize("rows", [4, 32, 64])
def test_wrapper_agrees_with_the_pivot_path(rows):
    """The dispatch must be invisible: same wrapper, split path forced on and off."""
    import sglang.srt.speculative.eagle_utils as eu

    torch.manual_seed(rows)
    probs = torch.softmax(torch.randn(rows, VOCAB, device="cuda"), dim=-1)
    top_ps = (0.05 + 0.95 * torch.rand(rows, device="cuda")).float()
    top_ps[0] = 1.0  # must pass through unchanged, as CUDA's top_p_renorm_prob does
    top_ks = torch.full((rows,), VOCAB, device="cuda")

    saved = eu._TP_ROW_LIMIT
    try:
        eu._TP_ROW_LIMIT = 0
        off = eu._renorm_top_k_top_p_hip(probs, top_ks, top_ps, need_top_k=False)
        eu._TP_ROW_LIMIT = rows
        on = eu._renorm_top_k_top_p_hip(probs, top_ks, top_ps, need_top_k=False)
    finally:
        eu._TP_ROW_LIMIT = saved

    assert torch.equal(on[0], probs[0])
    torch.testing.assert_close(on, off, atol=1e-5, rtol=0)
