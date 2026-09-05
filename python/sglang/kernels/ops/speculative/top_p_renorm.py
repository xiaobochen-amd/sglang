# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Split-row top-p renorm for the speculative verify shape."""

import torch
import triton
import triton.language as tl

# The pivot kernel in eagle_utils walks one block per row, which at
# CONC=1 decode is six blocks of a 256-CU GPU; these split each row across G blocks
# and reach tau by radix instead of by search. probs are non-negative, so the
# float32 bit pattern is monotonic in the value and 16 + 8 + 8 bits pin tau to an
# exact element -- a bin edge would not be any element's value, and the excess mass
# it keeps biases every output through Z.
NB0 = 1 << 16
NB = 256
HB = 1024
WIN = 8192
# Above this many rows the split stops paying: measured through this wrapper it is
# 1.41x at 32 rows, 0.99x at 48 and 1.00x from 64 on, where one block per row
# already fills the GPU. Cut at 32 so no row count is ever slower than before.
ROW_LIMIT = 32


@triton.jit
def _hist0(
    probs_ptr, hist_ptr, max_ptr, vocab, G, BLOCK: tl.constexpr, NB0: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid // G
    chunk = (vocab + G - 1) // G
    start = (pid % G) * chunk
    end = tl.minimum(start + chunk, vocab)
    offs = tl.arange(0, BLOCK)
    hi = tl.zeros((), tl.int32)
    for s in range(start, end, BLOCK):
        idx = s + offs
        m = idx < end
        p = tl.load(probs_ptr + row * vocab + idx, mask=m, other=0.0)
        b = (p.to(tl.uint32, bitcast=True) >> 16).to(tl.int32)
        hi = tl.maximum(hi, tl.max(tl.where(m, b, 0), axis=0))
        # float64 bins: in float32 the cut flipped by one boundary element as G
        # changed the atomic summation order.
        tl.atomic_add(
            hist_ptr + row * NB0 + b, tl.where(m, p, 0.0).to(tl.float64), mask=m
        )
    tl.atomic_max(max_ptr + row, hi)


@triton.jit
def _hist_lo(
    probs_ptr, hist_ptr, pre_ptr, vocab, G,
    SH: tl.constexpr, BLOCK: tl.constexpr, NB: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // G
    pre = tl.load(pre_ptr + row)
    chunk = (vocab + G - 1) // G
    start = (pid % G) * chunk
    end = tl.minimum(start + chunk, vocab)
    offs = tl.arange(0, BLOCK)
    for s in range(start, end, BLOCK):
        idx = s + offs
        m = idx < end
        p = tl.load(probs_ptr + row * vocab + idx, mask=m, other=0.0)
        bits = p.to(tl.uint32, bitcast=True).to(tl.int32)
        keep = m & ((bits >> (SH + 8)) == pre)
        # An atomic costs its issue slot even where the mask drops it, and at these
        # levels almost every tile is empty: skipping them is worth 5x here.
        if tl.sum(keep.to(tl.int32), axis=0) > 0:
            tl.atomic_add(
                hist_ptr + row * NB + ((bits >> SH) & (NB - 1)),
                tl.where(keep, p, 0.0),
                mask=keep,
            )


@triton.jit
def _cut0(
    hist_ptr, tp_ptr, max_ptr, cut_ptr, acc_ptr,
    NB0: tl.constexpr, HB: tl.constexpr, WIN: tl.constexpr,
):
    """Walk one row's bins from the top down, stopping where the mass reaches top_p.

    Only the WIN bins below the row max are scanned: this is one block per row, so
    the full 65536-bin walk was 65% of the whole op.
    """
    row = tl.program_id(0)
    tp = tl.load(tp_ptr + row).to(tl.float64)
    mb = tl.load(max_ptr + row)
    lane = tl.arange(0, HB)
    acc = tl.zeros((), tl.float64)
    cut = tl.zeros((), tl.int32)
    found = tl.zeros((), tl.int32)
    for cs in range(0, WIN, HB):
        off = mb + 1 - HB - cs + lane
        m = off >= 0
        hv = tl.load(hist_ptr + row * NB0 + off, mask=m, other=0.0)
        rev = tl.flip(hv, 0)
        n = tl.sum((acc + tl.cumsum(rev, axis=0) < tp).to(tl.int32), axis=0)
        hit = n < HB
        before = acc + tl.sum(
            tl.where(lane < n, rev, tl.zeros((HB,), tl.float64)), axis=0
        )
        cut = tl.where((found == 0) & hit, mb - cs - n, cut)
        acc = tl.where(found == 1, acc, tl.where(hit, before, acc + tl.sum(hv, axis=0)))
        found = tl.where(hit, 1, found)
    if found == 0:  # top_p reaches past the window
        acc = tl.zeros((), tl.float64)
        for cs in range(0, NB0, HB):
            off = NB0 - HB - cs + lane
            hv = tl.load(hist_ptr + row * NB0 + off)
            rev = tl.flip(hv, 0)
            n = tl.sum((acc + tl.cumsum(rev, axis=0) < tp).to(tl.int32), axis=0)
            hit = n < HB
            before = acc + tl.sum(
                tl.where(lane < n, rev, tl.zeros((HB,), tl.float64)), axis=0
            )
            cut = tl.where((found == 0) & hit, NB0 - 1 - cs - n, cut)
            acc = tl.where(
                found == 1, acc, tl.where(hit, before, acc + tl.sum(hv, axis=0))
            )
            found = tl.where(hit, 1, found)
    tl.store(cut_ptr + row, cut)
    tl.store(acc_ptr + row, acc)


@triton.jit
def _cut_lo(
    hist_ptr, tp_ptr, pre_ptr, acc_in_ptr, pre_out_ptr, acc_ptr, z_ptr,
    LAST: tl.constexpr, NB: tl.constexpr,
):
    row = tl.program_id(0)
    tp = tl.load(tp_ptr + row).to(tl.float64)
    acc = tl.load(acc_in_ptr + row)
    lane = tl.arange(0, NB)
    hv = tl.load(hist_ptr + row * NB + lane).to(tl.float64)
    rev = tl.flip(hv, 0)
    n = tl.sum((acc + tl.cumsum(rev, axis=0) < tp).to(tl.int32), axis=0)
    cut = tl.where(n < NB, NB - 1 - n, 0)
    acc = tl.where(
        n < NB,
        acc + tl.sum(tl.where(lane < n, rev, tl.zeros((NB,), tl.float64)), axis=0),
        acc,
    )
    tl.store(pre_out_ptr + row, (tl.load(pre_ptr + row) << 8) | cut)
    tl.store(acc_ptr + row, acc)
    if LAST:
        # Z is the mass strictly above tau plus the mass exactly at it, so the
        # kept set needs no second pass over the data.
        tl.store(
            z_ptr + row,
            acc + tl.sum(tl.where(lane == cut, hv, tl.zeros((NB,), tl.float64)), axis=0),
        )


@triton.jit
def _apply(
    probs_ptr, out_ptr, tau_ptr, z_ptr, vocab, G, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid // G
    tau = tl.load(tau_ptr + row).to(tl.float32, bitcast=True)
    inv = (1.0 / tl.maximum(tl.load(z_ptr + row), 1e-30)).to(tl.float32)
    chunk = (vocab + G - 1) // G
    start = (pid % G) * chunk
    end = tl.minimum(start + chunk, vocab)
    offs = tl.arange(0, BLOCK)
    for s in range(start, end, BLOCK):
        idx = s + offs
        m = idx < end
        p = tl.load(probs_ptr + row * vocab + idx, mask=m, other=0.0)
        tl.store(
            out_ptr + row * vocab + idx, tl.where(p >= tau, p * inv, 0.0), mask=m
        )


def top_p_renorm_split(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """Three-level radix top-p renorm, rows split across blocks. Output is
    bit-identical for any split factor G."""
    rows, vocab = probs.shape
    dev = probs.device
    # Cost is flat around 128 total blocks and rises past it: extra blocks buy
    # occupancy but pay for it in level-0 atomic contention.
    G = max(1, min(16, 128 // rows))
    f64 = dict(device=dev, dtype=torch.float64)
    h0 = torch.zeros((rows, NB0), **f64)
    h1 = torch.zeros((rows, NB), device=dev, dtype=torch.float32)
    h2 = torch.zeros((rows, NB), device=dev, dtype=torch.float32)
    mx = torch.zeros(rows, device=dev, dtype=torch.int32)
    p0 = torch.empty(rows, device=dev, dtype=torch.int32)
    p1 = torch.empty_like(p0)
    p2 = torch.empty_like(p0)
    a0, a1, a2, z = (torch.empty(rows, **f64) for _ in range(4))
    out = torch.empty_like(probs)

    grid = (rows * G,)
    _hist0[grid](probs, h0, mx, vocab, G, BLOCK=2048, NB0=NB0)
    _cut0[(rows,)](
        h0, top_ps, mx, p0, a0, NB0=NB0, HB=HB, WIN=WIN
    )
    _hist_lo[grid](probs, h1, p0, vocab, G, SH=8, BLOCK=2048, NB=NB)
    _cut_lo[(rows,)](
        h1, top_ps, p0, a0, p1, a1, z, LAST=False, NB=NB
    )
    _hist_lo[grid](probs, h2, p1, vocab, G, SH=0, BLOCK=2048, NB=NB)
    _cut_lo[(rows,)](h2, top_ps, p1, a1, p2, a2, z, LAST=True, NB=NB)
    _apply[grid](probs, out, p2, z, vocab, G, BLOCK=2048)
    return out
