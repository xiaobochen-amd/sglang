#!/usr/bin/env python3
"""Where aiter's Triton a16w16 beats what tuned_gemm picks, at decode shapes.

One GPU, no server, no weights:

    python3 bench_bf16_linear_routing.py
    python3 bench_bf16_linear_routing.py --n 2048,19360 --k 1024,2048,6144

Why this exists
    aiter's tuned GEMM keys its table on (N, K) and does not fall back to a
    neighbour, so a shape the table never saw lands on torch. A GLM-5.2 TP8
    decode step hits one of those 79 times -- the MLA q up-projection,
    (2048, 2048) -- and torch serves it well off the weight-bandwidth roofline.

    The routing gate in UnquantizedLinearMethod.apply is stated in terms of K
    rather than a list of shapes, because a list keyed on (N, K) is what makes
    the aiter table miss these shapes to begin with, and a different tensor
    parallel split moves every N. This sweep is what that threshold is read
    from: run it on a new part, or after an aiter bump, and the K column is the
    answer.

Method
    Both arms in one process, interleaved ABBA, under graph replay, with the
    ratio's own spread reported -- the ratio is formed inside a round from
    samples taken seconds apart and is far steadier than either arm.
"""

from __future__ import annotations

import argparse
import statistics as st

import torch


def _event_us(fn, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def _make_arms(x, w, w_t, out, triton_a16w16):
    """Callables for the two arms, so the tensors stay captured here.

    They are freed between shapes -- allocator fragmentation is enough to read
    one shape five times too slow -- and building the closures in their own
    scope keeps that teardown from looking like a use-after-delete.
    """
    return {
        "torch": lambda: torch.mm(x, w_t, out=out),
        "triton": lambda: triton_a16w16(x, w, dtype=torch.bfloat16),
    }


def _graph(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=torch.cuda.graphs.graph_pool_handle()):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", default="1024,2048,3072,4096,6144,8192,12288,19360")
    ap.add_argument("--k", default="1024,1536,2048,4096,6144")
    ap.add_argument("--m", default="1,6", help="decode rows: draft 1, verify batch*6")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=120)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU")
    from aiter.ops.triton.gemm_a16w16 import gemm_a16w16 as triton_a16w16

    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(
        f"\n{'M':>3} {'N':>6} {'K':>6} {'wMB':>7} {'torch':>8} {'triton':>8}"
        f" {'ratio':>7} {'sc%':>5}"
    )
    by_k = {}
    for M in (int(v) for v in args.m.split(",")):
        for N in (int(v) for v in args.n.split(",")):
            for K in (int(v) for v in args.k.split(",")):
                x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
                w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
                out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
                w_t = w.t().contiguous()
                try:
                    ref = torch.mm(x, w_t)
                    got = triton_a16w16(x, w, dtype=torch.bfloat16)
                except Exception as exc:  # noqa: BLE001
                    print(f"{M:>3} {N:>6} {K:>6}  declined: {str(exc)[:44]}")
                    continue
                err = (ref.float() - got.float()).abs().max().item()
                scale = ref.float().abs().max().item() or 1.0
                if err / scale > 1e-2:
                    print(f"{M:>3} {N:>6} {K:>6}  MISMATCH rel {err / scale:.2e}")
                    continue
                fns = _make_arms(x, w, w_t, out, triton_a16w16)
                arms = {
                    nm: _graph(fn, args.iters, args.warmup) for nm, fn in fns.items()
                }
                med, ratios = {"torch": [], "triton": []}, []
                for rep in range(args.reps):
                    order = ("torch", "triton", "triton", "torch")
                    if rep % 2:
                        order = order[::-1]
                    per = {"torch": [], "triton": []}
                    for nm in order:
                        t = _event_us(arms[nm].replay, args.iters)
                        med[nm].append(t)
                        per[nm].append(t)
                    ratios.append(st.median(per["triton"]) / st.median(per["torch"]))
                tm, gm = st.median(med["torch"]), st.median(med["triton"])
                rm = st.median(ratios)
                spread = (max(ratios) - min(ratios)) / rm * 100
                verdict = "win" if rm < 0.95 else ("--" if rm < 1.05 else "LOSE")
                print(
                    f"{M:>3} {N:>6} {K:>6} {N * K * 2 / 1e6:>7.1f} {tm:>8.2f}"
                    f" {gm:>8.2f} {rm:>7.3f} {spread:>4.1f}%  {verdict}"
                )
                by_k.setdefault(K, []).append(rm)
                del x, w, out, w_t, arms, fns
                torch.cuda.empty_cache()

    print("\nby K -- this is the column the routing threshold is read from:")
    for K in sorted(by_k):
        rs = by_k[K]
        print(
            f"  K={K:>5}  n={len(rs):3d}  median {st.median(rs):.3f}"
            f"  worst {max(rs):.3f}  wins {sum(1 for r in rs if r < 0.95)}/{len(rs)}"
        )


if __name__ == "__main__":
    main()
