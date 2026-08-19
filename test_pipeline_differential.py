"""
test_pipeline_differential.py -- the timing engine against an independent model.

The 122 invariants in verify_results.py constrain the engine's OUTPUTS across
the reported grid. The 23 checks in test_cxl_link.py constrain the link model.
Neither constrains pipelined_time_s itself: whether it overlaps transfers with
compute correctly under bus and link contention. That is the arithmetic every
number in the paper rests on, and it was the last part with no independent
check.

Two attacks, both randomized over thousands of configurations:

  1. DIFFERENTIAL. `reference_time_s` below is written from the engine's
     documented semantics, not from its code, and is structured differently:
     it materialises an explicit transfer list and walks per-resource
     timelines, where the engine folds both into one pass. Any disagreement is
     either a bug in one of them or a place where the documentation does not
     say what the code does. Both are findings.

  2. BOUNDS. Six properties that hold for ANY correct implementation,
     independent of scheduling policy: a serial compute engine, serial buses,
     a serial shared link, and monotonicity in lookahead, staging budget and
     bandwidth. These cannot be satisfied by transcribing the engine's own
     mistakes, because they are derived from the hardware model, not the code.

    python test_pipeline_differential.py            # 4000 random cases
    python test_pipeline_differential.py 20000      # more
"""
import math
import random
import sys

from pipeline import pipelined_time_s, SHARED_LINK
from tiers import CXL_HOST_LINK_CEILING

TIERS = ["GPU HBM", "Host DRAM", "CXL Device DRAM", "CXL Device NAND"]
CHUNK = 256 * 1024


def _dur(nbytes, bw, lat):
    """Transfer occupancy of a tier: streaming time plus per-chunk latency."""
    d = nbytes / bw
    if lat:
        d += math.ceil(nbytes / CHUNK) * lat
    return d


def reference_time_s(units, depth, bw, lat, budget=None):
    """Independent model, written from the documented semantics.

    Structured deliberately unlike the engine: transfers are materialised as
    explicit jobs, and each resource (one per tier, plus the shared host link)
    carries its own timeline that jobs are appended to.
    """
    n = len(units)
    if n == 0:
        return 0.0

    busy = {t: 0.0 for t in TIERS}      # per-tier bus timelines
    link = 0.0                          # shared CXL host link timeline
    finished_compute = []               # compute completion, per unit
    engine = 0.0                        # the single compute engine

    for i in range(n):
        by_tier, comp = units[i]

        # Earliest issue permitted by lookahead alone.
        back = i - depth - 1
        earliest = finished_compute[back] if back >= 0 else 0.0

        # Staging room: units back+1..i are in flight if we issue at back.
        # Walk forward from the furthest permitted point to the nearest one
        # until the bytes held fit the budget. (The engine walks the other
        # direction; the fixpoint is the same.)
        if budget is not None:
            held = sum(units[i][0].values())
            chosen = i - 1
            for m in range(i - 1, max(back, -1), -1):   # never below unit 0
                extra = sum(units[m][0].values())
                if held + extra > budget:
                    break
                held += extra
                chosen = m - 1
            earliest = max(earliest, finished_compute[chosen] if chosen >= 0 else 0.0)

        arrived = earliest
        for tier in by_tier:                      # engine iterates dict order
            nbytes = by_tier[tier]
            if nbytes <= 0 or not bw.get(tier):
                continue
            begin = max(earliest, busy[tier])
            if tier in SHARED_LINK:
                begin = max(begin, link)
                link = begin + nbytes / CXL_HOST_LINK_CEILING
            end = begin + _dur(nbytes, bw[tier], lat.get(tier, 0.0))
            busy[tier] = end
            arrived = max(arrived, end)

        engine = max(arrived, engine) + comp
        finished_compute.append(engine)

    return engine


def random_case(rng):
    n = rng.randint(1, 12)
    live = rng.sample(TIERS, rng.randint(1, len(TIERS)))
    # Bandwidths stay inside the physical envelope each tier actually has. A
    # device tier faster than the link it sits behind is not a configuration
    # the hardware can be in -- tiers.py asserts it -- and it breaks the link
    # bound below for a reason that says nothing about the engine: the link is
    # then held longer than the transfer it carries.
    PLAUSIBLE = {"GPU HBM": [896e9, 1792e9], "Host DRAM": [25.6e9, 38.4e9],
                 "CXL Device DRAM": [16e9, 27e9, 31.5e9],
                 "CXL Device NAND": [3e9, 5e9, 8e9]}
    bw = {t: rng.choice(PLAUSIBLE[t]) for t in live}
    lat = {t: rng.choice([0.0, 200e-9, 505e-9, 1547e-9]) for t in live}
    units = []
    for _ in range(n):
        by = {}
        for t in live:
            if rng.random() < 0.7:
                by[t] = rng.choice([0.0, 1e6, 64e6, 512e6, 2e9])
        units.append((by, rng.choice([0.0, 1e-4, 5e-3, 2e-2])))
    depth = rng.choice([0, 1, 2, 4, 5, 32, n, n + 5])
    budget = rng.choice([None, 0.0, 1e6, 512e6, 4e9, 64e9])
    return units, depth, bw, lat, budget


def bounds(units, depth, bw, lat, budget, got):
    """Properties any correct implementation satisfies, from the hardware model."""
    tol = 1e-9
    problems = []

    # 1. Compute is one serial engine: the pass cannot beat its total compute.
    total_comp = sum(c for _, c in units)
    if got < total_comp - tol:
        problems.append(f"beats serial compute: {got:.9g} < {total_comp:.9g}")

    # 2. Each tier is a serial bus: cannot beat its own total occupancy.
    for t in TIERS:
        if not bw.get(t):
            continue
        occ = sum(_dur(u[0][t], bw[t], lat.get(t, 0.0))
                  for u in units if u[0].get(t, 0) > 0)
        if got < occ - tol:
            problems.append(f"beats {t} bus: {got:.9g} < {occ:.9g}")

    # 3. The shared host link is serial across both device tiers.
    shared = sum(u[0].get(t, 0.0) for u in units for t in SHARED_LINK)
    if got < shared / CXL_HOST_LINK_CEILING - tol:
        problems.append("beats the shared CXL link ceiling")

    # 4. Lookahead never hurts: deeper is never slower.
    if depth > 0:
        shallower = pipelined_time_s(units, depth - 1, bw, lat, inflight_budget=budget)
        if got > shallower + 1e-6 * max(1.0, shallower):
            problems.append(f"deeper lookahead is slower: {got:.9g} > {shallower:.9g}")

    # 5. Staging room never hurts: a larger budget is never slower.
    if budget is not None:
        bigger = pipelined_time_s(units, depth, bw, lat, inflight_budget=budget * 4 + 1e9)
        if bigger > got + 1e-6 * max(1.0, got):
            problems.append(f"more staging room is slower: {bigger:.9g} > {got:.9g}")

    # 6. Bandwidth never hurts: a faster machine is never slower.
    faster = pipelined_time_s(units, depth, {t: v * 2 for t, v in bw.items()}, lat,
                              inflight_budget=budget)
    if faster > got + 1e-6 * max(1.0, got):
        problems.append(f"doubling bandwidth is slower: {faster:.9g} > {got:.9g}")

    return problems


def main(n_cases):
    rng = random.Random(20260818)
    diffs = bad = 0
    worst = 0.0
    for k in range(n_cases):
        units, depth, bw, lat, budget = random_case(rng)
        got = pipelined_time_s(units, depth, bw, lat, inflight_budget=budget)
        ref = reference_time_s(units, depth, bw, lat, budget)

        scale = max(1e-12, abs(got), abs(ref))
        rel = abs(got - ref) / scale
        worst = max(worst, rel)
        if rel > 1e-9:
            diffs += 1
            if diffs <= 3:
                print(f"  DIFFERENTIAL case {k}: engine {got:.9g} vs reference {ref:.9g} "
                      f"(rel {rel:.3g}) depth={depth} budget={budget}")

        for p in bounds(units, depth, bw, lat, budget, got):
            bad += 1
            if bad <= 5:
                print(f"  BOUND case {k}: {p}")

    print(f"\n{n_cases} random configurations")
    print(f"  differential vs independent reference: "
          f"{n_cases - diffs} agree, {diffs} differ (worst rel {worst:.3g})")
    print(f"  hardware-model bounds: {bad} violations")
    ok = diffs == 0 and bad == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000))
