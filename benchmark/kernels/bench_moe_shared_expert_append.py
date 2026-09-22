#!/usr/bin/env python3
"""Standalone benchmark for the folded shared-expert append on ROCm.

One GPU, no server, no model weights, about a minute:

    python3 bench_moe_shared_expert_append.py
    python3 bench_moe_shared_expert_append.py --rows 1,8,64 --top-k 8

What it measures
    _post_process_topk_ids used to bracket the shared-expert append with two
    separate padded-region fills -- one to stamp the routed ids of the padded
    rows, one to zero their weights -- for three launches at a shape whose work
    is a few microseconds. The append kernel can do both itself, which is one
    launch.

      base   _fill_padded_rows(ids) ; fused_append_shared_experts ; _fill_padded_rows(weights)
      pr     fused_append_shared_experts(..., num_token_non_padded=..., pad_fill_id=0)

    Both arms run in one process and interleave ABBA, so the clock drift that
    moves kernels this small by ~10% lands on both. Judge a row by the ratio's
    spread, the last column, not by either arm's: the ratio is formed inside a
    round from samples taken seconds apart, and is far steadier than either.

    This is a fold, not an approximation, so the two arms must write identical
    bytes. That is checked before anything is timed.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch


def event_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def build_graph(fn, iters: int, warmup: int):
    """Warm up, then capture `iters` calls into one graph.

    The warmup is long on purpose: an MI355X runs roughly 20% slow until its
    clocks ramp, and a short warmup reads that ramp as a difference between
    the arms.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graphs.graph_pool_handle()
    with torch.cuda.graph(graph, pool=pool):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    return graph


def time_arms_abba(fns, iters: int, reps: int, warmup: int, use_graph: bool):
    names = list(fns)
    if use_graph:
        graphs = {n: build_graph(fns[n], iters, warmup) for n in names}

        def run(n):
            return event_ms(graphs[n].replay) * 1e3 / iters

    else:
        for n in names:
            for _ in range(warmup):
                fns[n]()
        torch.cuda.synchronize()

        def run(n):
            def body():
                for _ in range(iters):
                    fns[n]()

            return event_ms(body) * 1e3 / iters

    samples = {n: [] for n in names}
    ratios = []
    a, b = names
    for rep in range(reps):
        order = (a, b, b, a) if rep % 2 == 0 else (b, a, a, b)
        per_rep = {a: [], b: []}
        for n in order:
            t = run(n)
            samples[n].append(t)
            per_rep[n].append(t)
        ratios.append(statistics.median(per_rep[b]) / statistics.median(per_rep[a]))
    return samples, ratios


def summarize(xs):
    med = statistics.median(xs)
    spread = (max(xs) - min(xs)) / med * 100 if med else float("nan")
    return med, spread


def make_case(
    append_mod, fill_mod, rows: int, top_k: int, shared: int, n_routed: int, device
):
    torch.manual_seed(999 + rows)
    ids0 = torch.randint(0, n_routed, (rows, top_k), device=device, dtype=torch.int32)
    w0 = torch.rand(rows, top_k, device=device, dtype=torch.float32)
    # Decode pads the batch out to the captured graph size, so trailing rows
    # carry no real token.
    ntnp = torch.tensor([max(1, rows // 2)], device=device, dtype=torch.int32)
    ids_base, w_base = ids0.clone(), w0.clone()
    ids_pr, w_pr = ids0.clone(), w0.clone()
    scale = 1.0

    def base():
        # In-place and idempotent, which is what makes repeated timing honest.
        fill_mod._fill_padded_rows(ids_base, ntnp, 0)
        out_ids, out_w = append_mod.fused_append_shared_experts(
            ids_base, w_base, shared, scale, N=n_routed
        )
        fill_mod._fill_padded_rows(out_w, ntnp, 0.0)
        return out_ids, out_w

    def pr():
        return append_mod.fused_append_shared_experts(
            ids_pr,
            w_pr,
            shared,
            scale,
            N=n_routed,
            num_token_non_padded=ntnp,
            pad_fill_id=0,
        )

    def check():
        bi, bw = base()
        pi, pw = pr()
        return int((bi != pi).sum().item()), int((bw != pw).sum().item())

    return base, pr, check


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tree",
        default=os.environ.get("SGLANG_TREE", ""),
        help="sglang checkout to import from; empty uses the installed package",
    )
    ap.add_argument("--rows", default="1,2,6,8,12,24,48,84")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--shared", type=int, default=1, help="fused shared experts")
    ap.add_argument("--n-routed", type=int, default=256)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=250)
    ap.add_argument("--eager", action="store_true", help="also time without graphs")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU")
    device = torch.device("cuda")
    if args.tree:
        sys.path.insert(0, os.path.join(args.tree, "python"))

    import sglang
    from sglang.kernels.ops.moe import fill_padded_rows as fill_mod
    from sglang.kernels.ops.moe import fused_moe_triton_kernels as append_mod

    rows_list = [int(r) for r in args.rows.split(",") if r]
    print(f"device : {torch.cuda.get_device_name(0)}")
    # Say which tree answered: a stale editable install pointing elsewhere is
    # the classic way to benchmark code you did not change.
    print(f"sglang : {sglang.__file__}")
    print(
        f"shape  : top_k {args.top_k}, shared {args.shared}, n_routed {args.n_routed}"
    )

    print("  -- correctness")
    bad = False
    for rows in rows_list:
        _, _, check = make_case(
            append_mod, fill_mod, rows, args.top_k, args.shared, args.n_routed, device
        )
        n_id, n_w = check()
        ok = n_id == 0 and n_w == 0
        bad |= not ok
        print(
            f"     rows={rows:<4} {'ok ' if ok else 'BAD'} "
            f"id mismatches {n_id}, weight mismatches {n_w}"
        )
    if bad:
        raise SystemExit("correctness gate failed; not timing")

    for label, use_graph in [("graph", True)] + (
        [("eager", False)] if args.eager else []
    ):
        print(f"  -- {label} (us/call)")
        print(
            f"     {'rows':>5}  {'base':>9} {'sc%':>6}  {'pr':>9} {'sc%':>6}"
            f"  {'ratio':>6} {'sc%':>6}"
        )
        for rows in rows_list:
            base, pr, _ = make_case(
                append_mod,
                fill_mod,
                rows,
                args.top_k,
                args.shared,
                args.n_routed,
                device,
            )
            s, ratios = time_arms_abba(
                {"base": base, "pr": pr}, args.iters, args.reps, args.warmup, use_graph
            )
            bm, bs = summarize(s["base"])
            pm, ps = summarize(s["pr"])
            rm, rs = summarize(ratios)
            flag = "" if rs < 3.0 else "   <- noisy"
            print(
                f"     {rows:>5}  {bm:>9.2f} {bs:>5.2f}%  {pm:>9.2f} {ps:>5.2f}%"
                f"  {rm:>6.3f} {rs:>5.2f}%{flag}"
            )


if __name__ == "__main__":
    main()
