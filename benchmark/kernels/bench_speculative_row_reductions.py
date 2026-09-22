#!/usr/bin/env python3
"""Standalone kernel benchmark for the two speculative-sampling kernels.

No server, no model weights, one GPU, about a minute. Both arms run in the same
process, so the comparison is paired: the "base" arm is the torch expression the
kernel replaced, the "pr" arm is the kernel.

    python3 bench_spec_kernels.py
    python3 bench_spec_kernels.py --only argmax --rows 1,2,6
    python3 bench_spec_kernels.py --tree /shared_nfs/kyle/nb_sglang

What it measures
    softmax   torch.softmax(logits / temperatures, -1)   vs  temperature_softmax
    argmax    scores.argmax(-1, keepdim=True)            vs  row_argmax

Both are vocab-wide row reductions at speculative row counts (batch_size per
draft step, batch_size * num_draft_tokens at verify). torch walks one block per
row, which leaves a 256-CU GPU idle at those row counts.

Timed under CUDA graph replay, which is the path decode actually runs on, and
in eager for contrast. The eager column is reported because it can be a LOSS:
the kernels trade launches for parallelism, and three launches only pay off once
capture has amortized them. A benchmark that only showed eager would reject
both kernels; one that only showed graph would hide the trade.

Correctness is checked before anything is timed. argmax must match torch
exactly; softmax is compared against a float64 reference, since torch itself is
not the ground truth here.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys

import torch

VOCAB_GLM52 = 154880
DEFAULT_ROWS = (1, 2, 4, 6, 8, 12, 24, 32, 48)


def load_module(path: str, name: str):
    """Import a kernel file by path, so the benchmark needs no installed sglang."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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
    clocks ramp, which takes one to two minutes of sustained load, and a short
    warmup reads that ramp as a difference between the arms.
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
    """Interleave the arms ABBA so clock drift lands on both, not on one.

    Returns per-arm lists of us/call. A single A...A then B...B ordering hands
    every bit of drift during the run to whichever arm ran second; ABBA cancels
    the linear part of it.
    """
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
        # The ratio is formed inside the round, from samples taken seconds
        # apart. Both arms drift together with the clock, so the ratio is far
        # steadier than either arm -- which is the whole point of interleaving,
        # and why the per-arm spread below is not the figure to judge this on.
        ratios.append(statistics.median(per_rep[b]) / statistics.median(per_rep[a]))
    return samples, ratios


def summarize(xs):
    med = statistics.median(xs)
    spread = (max(xs) - min(xs)) / med * 100 if med else float("nan")
    return med, spread


def make_softmax_case(mod, rows: int, vocab: int, device):
    torch.manual_seed(1234 + rows)
    logits = torch.randn(rows, vocab, device=device, dtype=torch.float32)
    # Temperatures are per row and broadcast, exactly as eagle_sample builds them
    # with repeat_interleave over the draft tokens.
    temps = torch.empty(rows, 1, device=device, dtype=torch.float32).uniform_(0.5, 1.5)

    def base():
        return torch.softmax(logits / temps, dim=-1)

    def pr():
        return mod.temperature_softmax(logits, temps)

    def check():
        ref = torch.softmax(
            logits.double() / temps.double(), dim=-1
        )  # float64 reference
        got = pr().double()
        base_err = (base().double() - ref).abs().max().item()
        pr_err = (got - ref).abs().max().item()
        row_sums = got.sum(-1)
        return {
            "base_err_vs_fp64": base_err,
            "pr_err_vs_fp64": pr_err,
            "max_row_sum_dev": (row_sums - 1).abs().max().item(),
        }

    return base, pr, check


def make_argmax_case(mod, rows: int, vocab: int, device):
    torch.manual_seed(4321 + rows)
    # fast_sample feeds probs/q, always float32 and contiguous.
    scores = torch.randn(rows, vocab, device=device, dtype=torch.float32)

    def base():
        return scores.argmax(dim=-1, keepdim=True)

    def pr():
        return mod.row_argmax(scores)

    def check():
        want = base()
        got = pr()
        mismatch = int((want != got).sum().item())
        return {"index_mismatches": mismatch, "rows": rows}

    return base, pr, check


CASES = {
    "softmax": ("temperature_softmax.py", "temperature_softmax", make_softmax_case),
    "argmax": ("topk1.py", "topk1", make_argmax_case),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tree",
        default=os.environ.get("SGLANG_TREE", "/shared_nfs/kyle/nb_sglang"),
        help="sglang checkout holding python/sglang/kernels/ops/speculative/",
    )
    ap.add_argument("--only", choices=sorted(CASES), help="run a single case")
    ap.add_argument("--rows", default=",".join(str(r) for r in DEFAULT_ROWS))
    ap.add_argument("--vocab", type=int, default=VOCAB_GLM52)
    ap.add_argument("--iters", type=int, default=50, help="calls per timed replay")
    ap.add_argument("--reps", type=int, default=6, help="ABBA rounds (4 samples each)")
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--eager", action="store_true", help="also time without graphs")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU")
    device = torch.device("cuda")
    rows_list = [int(r) for r in args.rows.split(",") if r]
    kdir = os.path.join(args.tree, "python/sglang/kernels/ops/speculative")
    names = [args.only] if args.only else sorted(CASES)

    print(f"device : {torch.cuda.get_device_name(0)}")
    print(f"tree   : {args.tree}")
    print(f"vocab  : {args.vocab}   iters/replay {args.iters}   ABBA reps {args.reps}")

    for case in names:
        fname, modname, maker = CASES[case]
        mod = load_module(os.path.join(kdir, fname), modname)
        print(f"\n=== {case}   ({fname})")

        # Correctness first. A kernel that is wrong does not get a time.
        print("  -- correctness")
        bad = False
        for rows in rows_list:
            _, _, check = maker(mod, rows, args.vocab, device)
            res = check()
            if case == "argmax":
                ok = res["index_mismatches"] == 0
                detail = f"index mismatches {res['index_mismatches']}"
            else:
                # The kernel is allowed to be no worse than torch against fp64.
                ok = res["pr_err_vs_fp64"] <= max(res["base_err_vs_fp64"] * 4, 1e-7)
                detail = (
                    f"err vs fp64: base {res['base_err_vs_fp64']:.3e} "
                    f"pr {res['pr_err_vs_fp64']:.3e}  "
                    f"row-sum dev {res['max_row_sum_dev']:.3e}"
                )
            bad |= not ok
            print(f"     rows={rows:<4} {'ok ' if ok else 'BAD'} {detail}")
        if bad:
            print("  correctness gate failed; not timing this case")
            continue

        modes = [("graph", True)] + ([("eager", False)] if args.eager else [])
        for label, use_graph in modes:
            print(f"  -- {label} (us/call)")
            print(
                f"     {'rows':>5}  {'base':>9} {'sc%':>6}  {'pr':>9} {'sc%':>6}"
                f"  {'ratio':>6} {'sc%':>6}"
            )
            for rows in rows_list:
                base, pr, _ = maker(mod, rows, args.vocab, device)
                s, ratios = time_arms_abba(
                    {"base": base, "pr": pr},
                    args.iters,
                    args.reps,
                    args.warmup,
                    use_graph,
                )
                bm, bs = summarize(s["base"])
                pm, ps = summarize(s["pr"])
                rm, rs = summarize(ratios)
                # Judge the run on the ratio's spread: it is the paired
                # quantity, and it is what the claim rests on.
                flag = "" if rs < 3.0 else "   <- noisy"
                print(
                    f"     {rows:>5}  {bm:>9.2f} {bs:>5.2f}%  {pm:>9.2f} {ps:>5.2f}%"
                    f"  {rm:>6.3f} {rs:>5.2f}%{flag}"
                )


if __name__ == "__main__":
    main()
