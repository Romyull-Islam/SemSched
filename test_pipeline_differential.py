"""
Test the timing engine, pipelined_time_s, against independent models.

verify_results.py checks the engine's outputs across the reported grid and
test_cxl_link.py checks the link model; this file checks that the engine
overlaps transfers with compute correctly under bus and link contention. The
checks are randomized over thousands of configurations:

  1. Differential. Each bus order has its own reference, written from the
     documented semantics rather than the engine's code and structured
     differently from it; main() runs the one pipeline.BUS_ORDER selects. A
     disagreement is a bug in the engine or the reference, or a gap between
     the documentation and the code. Both share the issue rule (_issue_at):
     prefetch depth and the staging-budget walk.

       reference_time_fifo   BUS_ORDER=fifo. Transfers are materialised as
                             explicit jobs and every resource (a bus per tier,
                             the shared host link, the GPU link, the whole
                             device when it serves one request at a time) is a
                             single busy-until timeline that jobs are appended
                             to; a transfer starts once every resource it
                             crosses is free.
       reference_time_fill   BUS_ORDER=fill, the default. A request takes the
                             earliest stretch of its resource, at or after the
                             time it may start, that no earlier booking
                             overlaps. Every booking goes into one flat job
                             list; a start is found by trying candidate points
                             in time order (the request's own ready time, then
                             each end of a job booked on that resource) and
                             taking the first that overlaps nothing, rather
                             than by searching a per-resource list of gaps.

  2. Bounds. Properties derived from the hardware model rather than the code:
     a serial compute engine, serial buses (device DRAM carrying the NAND
     bytes that pass through it), a serial shared host link and GPU link, and
     monotonicity in lookahead, staging budget and bandwidth. The floors hold
     under both bus orders. Monotonicity holds under fifo only: under fill an
     earlier or shorter booking can take part of a gap a later, longer
     request needed (the list-scheduling anomaly), so there a reversal is
     reported, not failed, provided the reference reproduces it.

  3. Work conservation (fill only). When every unit reads the same single
     tier, a bus never sits idle while a request that is ready waits for it.
     work_conserving_time states that as a closed-form recurrence, with no
     booking search, and the engine must match it. FIFO need not: there a
     bus also waits for the links its transfer crosses.

    python test_pipeline_differential.py            # 4000 random cases
    python test_pipeline_differential.py 20000      # more
"""
import math
import os
import random
import sys

if float(os.environ.get("GPU_LINK_BPS", "0")):
    # pipeline.py drops the GPU link on a host without an accelerator, which is
    # sim_cfg's default. Every case here can place bytes in GPU HBM, so give
    # this process one; otherwise GPU_LINK_BPS=63e9 would silently test nothing.
    import sim_cfg
    if not sim_cfg.gpu_hbm_capacity_bytes:
        sim_cfg.gpu_hbm_capacity_bytes = 80 * 2**30

import pipeline
from pipeline import (pipelined_time_s, SHARED_LINK, NAND_PASS_THROUGH, STREAM_COMPUTE,
                      DEVICE_SERIAL, GPU_LINK_BPS, STAGE_NAND_ONLY, BUS_ORDER)
from tiers import CXL_HOST_LINK_CEILING

TIERS = ["GPU HBM", "Host DRAM", "CXL Device DRAM", "CXL Device NAND"]
CHUNK = 256 * 1024
# Bandwidths inside each tier's physical envelope. A device tier faster than its
# host link is impossible (tiers.py asserts this) and would break bound 3 under
# fifo for reasons unrelated to the engine: the link would be held longer than
# the transfer it carries, and fifo counts a transfer done when its bus is.
PLAUSIBLE = {"GPU HBM": [896e9, 1792e9], "Host DRAM": [25.6e9, 38.4e9],
             "CXL Device DRAM": [16e9, 27e9, 31.5e9],
             "CXL Device NAND": [3e9, 5e9, 8e9]}


def _dur(nbytes, bw, lat):
    """Transfer occupancy of a tier: streaming time plus per-chunk latency."""
    d = nbytes / bw
    if lat:
        d += math.ceil(nbytes / CHUNK) * lat
    return d


def _room(by_tier):
    """Staging room a read issued ahead holds until it is consumed."""
    return by_tier.get("CXL Device NAND", 0.0) if STAGE_NAND_ONLY else sum(by_tier.values())


def _issue_at(units, i, depth, budget, finished_compute):
    """Earliest time unit i's reads may be issued; the same under both bus orders.

    Lookahead lets unit i issue once unit i-depth-1 has computed. Staging room
    then holds it back further: units back+1..i are in flight if it issues at
    back, so walk forward from the furthest permitted point to the nearest one
    until the bytes held fit the budget. (The engine walks the other direction;
    the fixpoint is the same.)
    """
    back = i - depth - 1
    earliest = finished_compute[back] if back >= 0 else 0.0
    if budget is not None:
        held = _room(units[i][0])
        chosen = i - 1
        for m in range(i - 1, max(back, -1), -1):   # never below unit 0
            extra = _room(units[m][0])
            if held + extra > budget:
                break
            held += extra
            chosen = m - 1
        earliest = max(earliest, finished_compute[chosen] if chosen >= 0 else 0.0)
    return earliest


def _compute_end(engine, issue, ready, comp):
    """When a unit's compute ends on the single serial compute engine."""
    if STREAM_COMPUTE:
        # Compute runs alongside the unit's own reads and ends after both.
        return max(max(issue, engine) + comp, ready)
    return max(ready, engine) + comp


def reference_time_fifo(units, depth, bw, lat, budget=None):
    """Independent model of BUS_ORDER=fifo, written from the documented semantics.

    Structured deliberately unlike the engine: transfers are materialised as
    explicit jobs, and each resource (one per tier, plus the shared host link)
    carries its own timeline that jobs are appended to.
    """
    n = len(units)
    if n == 0:
        return 0.0

    busy = {t: 0.0 for t in TIERS}      # per-tier bus timelines
    link = 0.0                          # shared CXL host link timeline
    gpu_link = 0.0                      # accelerator PCIe link timeline
    device = 0.0                        # the whole device, when it serves one request at a time
    finished_compute = []               # compute completion, per unit
    engine = 0.0                        # the single compute engine

    for i in range(n):
        by_tier, comp = units[i]
        earliest = _issue_at(units, i, depth, budget, finished_compute)

        arrived = earliest
        for tier in by_tier:                      # engine iterates dict order
            nbytes = by_tier[tier]
            if nbytes <= 0 or not bw.get(tier):
                continue
            on_device = tier in SHARED_LINK
            to_gpu = GPU_LINK_BPS and tier != "GPU HBM"
            begin = max(earliest, busy[tier])
            if DEVICE_SERIAL and on_device:
                begin = max(begin, device)
            if on_device:
                begin = max(begin, link)
            if to_gpu:
                begin = max(begin, gpu_link)
            # Each link is held from the transfer's start for its bytes at the
            # link's own rate.
            if on_device:
                link = begin + nbytes / CXL_HOST_LINK_CEILING
            if to_gpu:
                gpu_link = begin + nbytes / GPU_LINK_BPS
            end = begin + _dur(nbytes, bw[tier], lat.get(tier, 0.0))
            busy[tier] = end
            if DEVICE_SERIAL and on_device:
                # The device is busy until this request is done; its NAND fill
                # happens inside that time.
                device = end
            elif (NAND_PASS_THROUGH and tier == "CXL Device NAND"
                    and bw.get("CXL Device DRAM")):
                # The bytes also cross the device DRAM at its own rate, from the
                # start of the NAND read, once per pass; they land when both are done.
                fill = (max(begin, busy["CXL Device DRAM"])
                        + NAND_PASS_THROUGH * nbytes / bw["CXL Device DRAM"])
                busy["CXL Device DRAM"] = fill
                end = max(end, fill)
            arrived = max(arrived, end)

        engine = _compute_end(engine, earliest, arrived, comp)
        finished_compute.append(engine)

    return engine


def _bus(tier):
    """The bus a tier's transfers occupy: its own, or the device's when that
    serves one request at a time."""
    return "the device" if DEVICE_SERIAL and tier in SHARED_LINK else tier


def _side_requests(tier, nbytes, bw):
    """What a transfer occupies besides its bus, in booking order: the shared
    host link, the GPU link, and the device DRAM a NAND read passes through.
    Each is (resource, how long the transfer holds it)."""
    side = []
    if tier in SHARED_LINK:
        side.append(("host link", nbytes / CXL_HOST_LINK_CEILING))
    if GPU_LINK_BPS and tier != "GPU HBM":
        side.append(("GPU link", nbytes / GPU_LINK_BPS))
    if (NAND_PASS_THROUGH and not DEVICE_SERIAL and tier == "CXL Device NAND"
            and bw.get("CXL Device DRAM")):
        side.append((_bus("CXL Device DRAM"),
                     NAND_PASS_THROUGH * nbytes / bw["CXL Device DRAM"]))
    return side


def reference_time_fill(units, depth, bw, lat, budget=None):
    """Independent model of BUS_ORDER=fill, written from the documented semantics.

    A request that may start at t takes the earliest s >= t at which [s, s+d)
    overlaps nothing already booked on its resource. Structured deliberately
    unlike an engine that keeps a sorted interval list per resource: every
    booking, of every resource, goes into one flat job list, and a start is
    found by trying candidate points rather than walking gaps. The candidates
    are t and each end of a job on that resource after t: if s > t is the
    earliest feasible start, the instant just before s is covered by some job,
    and that job ends exactly at s.
    """
    jobs = []           # (resource, begin, end), in booking order

    def book(resource, t, length):
        mine = [(b, e) for r, b, e in jobs if r == resource]
        for s in sorted({t} | {e for _, e in mine if e > t}):
            if all(e <= s or s + length <= b for b, e in mine):
                jobs.append((resource, s, s + length))
                return s
        raise AssertionError("unreachable: the latest job end is always free")

    finished_compute = []
    engine = 0.0
    for i, (by_tier, comp) in enumerate(units):
        issue = _issue_at(units, i, depth, budget, finished_compute)
        ready = issue
        for tier, nbytes in by_tier.items():      # booking follows dict order
            if nbytes <= 0 or not bw.get(tier):
                continue
            occupancy = _dur(nbytes, bw[tier], lat.get(tier, 0.0))
            start = book(_bus(tier), issue, occupancy)
            landed = start + occupancy
            # The bus is booked first; each further resource from the bus start
            # on, and the bytes land when the last of them is done.
            for resource, held in _side_requests(tier, nbytes, bw):
                landed = max(landed, book(resource, start, held) + held)
            ready = max(ready, landed)
        engine = _compute_end(engine, issue, ready, comp)
        finished_compute.append(engine)
    return engine


REFERENCES = {"fifo": reference_time_fifo, "fill": reference_time_fill}


def work_conserving_time(units, depth, bw, lat, budget=None):
    """Closed form for BUS_ORDER=fill when every unit reads one tier, the same.

    Issue times never decrease along the unit list, so each unit reaches the
    bus no earlier than the one before it, and its requests of the host link,
    the GPU link and the device DRAM (a NAND read's pass-through) reach those
    no earlier than the previous unit's did. No other tier uses any of them.
    A resource that never idles while a ready request waits then serves its
    requests back to back: each starts at max(ready, end of the previous one).
    Nothing here searches for a gap; under fill there is never one to use.
    """
    assert len({t for by, _ in units for t in by}) <= 1, "one tier only"
    idle_from = {}      # resource -> when its last booking ends
    finished_compute = []
    engine = 0.0
    for i, (by_tier, comp) in enumerate(units):
        issue = _issue_at(units, i, depth, budget, finished_compute)
        ready = issue
        for tier, nbytes in by_tier.items():
            if nbytes <= 0 or not bw.get(tier):
                continue
            occupancy = _dur(nbytes, bw[tier], lat.get(tier, 0.0))
            start = max(issue, idle_from.get("bus", 0.0))
            idle_from["bus"] = start + occupancy
            ready = max(ready, start + occupancy)
            for resource, held in _side_requests(tier, nbytes, bw):
                begin = max(start, idle_from.get(resource, 0.0))
                idle_from[resource] = begin + held
                ready = max(ready, begin + held)
        engine = _compute_end(engine, issue, ready, comp)
        finished_compute.append(engine)
    return engine


def random_case(rng):
    n = rng.randint(1, 12)
    live = rng.sample(TIERS, rng.randint(1, len(TIERS)))
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


def random_single_tier_case(rng):
    """A unit list that reads one tier only, for the work-conservation check.

    Bandwidths may also exceed the links here (a device tier above the host
    link, host DRAM above a PCIe 5.0 x16 GPU link), so a link can be the
    binding resource; the property must hold whichever one binds. A NAND list
    usually also has device DRAM to pass through.
    """
    tier = rng.choice(TIERS)
    faster = {"GPU HBM": [], "Host DRAM": [76.8e9], "CXL Device DRAM": [40e9, 64e9],
              "CXL Device NAND": [40e9]}
    bw = {tier: rng.choice(PLAUSIBLE[tier] + faster[tier])}
    if tier == "CXL Device NAND" and rng.random() < 0.8:
        bw["CXL Device DRAM"] = rng.choice(PLAUSIBLE["CXL Device DRAM"])
    lat = {t: rng.choice([0.0, 200e-9, 1547e-9]) for t in bw}
    n = rng.randint(1, 12)
    units = []
    for _ in range(n):
        by = {tier: rng.choice([0.0, 1e6, 64e6, 512e6, 2e9])} if rng.random() < 0.85 else {}
        units.append((by, rng.choice([0.0, 1e-4, 5e-3, 2e-2])))
    depth = rng.choice([0, 1, 2, 4, 32, n + 5])
    budget = rng.choice([None, 0.0, 1e6, 512e6, 4e9])
    return units, depth, bw, lat, budget


def bounds(units, depth, bw, lat, budget, got, reference):
    """Properties of the hardware model, checked against the engine's result.

    Returns (violations, anomalies). Bounds 1-3c hold under both bus orders:
    every booking of a resource is disjoint from the others on it and ends
    before its unit's compute does. Bounds 4-6 (monotonicity) are implied by
    fifo only, where every start is a max-plus expression in issue times and
    durations, so nothing earlier or shorter can delay anything. Under fill a
    request takes the first gap it fits, and an earlier or shorter booking can
    land inside a gap a later, longer request needed, which then waits for the
    next one: the list-scheduling anomaly. A reversal under fill is an anomaly
    if the reference reproduces it on the perturbed configuration, and a
    violation if it does not.
    """
    tol = 1e-9
    problems, anomalies = [], []

    # 1. Compute is one serial engine: the pass cannot beat its total compute.
    total_comp = sum(c for _, c in units)
    if got < total_comp - tol:
        problems.append(f"beats serial compute: {got:.9g} < {total_comp:.9g}")

    # 2. Each tier is a serial bus: cannot beat its own total occupancy. Device
    # DRAM also carries every pass of the NAND bytes that cross it.
    for t in TIERS:
        if not bw.get(t):
            continue
        occ = sum(_dur(u[0][t], bw[t], lat.get(t, 0.0))
                  for u in units if u[0].get(t, 0) > 0)
        if t == "CXL Device DRAM" and NAND_PASS_THROUGH and not DEVICE_SERIAL:
            occ += sum(NAND_PASS_THROUGH * u[0].get("CXL Device NAND", 0.0) / bw[t]
                       for u in units if bw.get("CXL Device NAND"))
        if got < occ - tol:
            problems.append(f"beats {t} bus: {got:.9g} < {occ:.9g}")

    # 3. The shared host link is serial across both device tiers.
    shared = sum(u[0].get(t, 0.0) for u in units for t in SHARED_LINK if bw.get(t))
    if got < shared / CXL_HOST_LINK_CEILING - tol:
        problems.append("beats the shared CXL link ceiling")

    # 3b. A device serving one request at a time cannot beat the sum of its
    # two tiers' occupancy.
    if DEVICE_SERIAL:
        dev = sum(_dur(u[0][t], bw[t], lat.get(t, 0.0))
                  for u in units for t in SHARED_LINK if bw.get(t) and u[0].get(t, 0) > 0)
        if got < dev - tol:
            problems.append(f"beats the serial device: {got:.9g} < {dev:.9g}")

    # 3c. The GPU link is serial across every tier outside GPU HBM.
    if GPU_LINK_BPS:
        crossing = sum(v for by, _ in units for t, v in by.items()
                       if t != "GPU HBM" and bw.get(t) and v > 0)
        if got < crossing / GPU_LINK_BPS - tol:
            problems.append("beats the GPU link ceiling")

    def monotone(what, alt_case, alt_is_better):
        alt_units, alt_depth, alt_bw, alt_lat, alt_budget = alt_case
        alt = pipelined_time_s(alt_units, alt_depth, alt_bw, alt_lat, inflight_budget=alt_budget)
        better, worse = (alt, got) if alt_is_better else (got, alt)
        if better <= worse + 1e-6 * max(1.0, worse):
            return
        msg = f"{what}: {better:.9g} > {worse:.9g}"
        if BUS_ORDER == "fill" and _rel(alt, reference(*alt_case)) <= 1e-9:
            anomalies.append(msg)
        else:
            problems.append(msg)

    # 4. Lookahead never hurts: deeper is never slower.
    if depth > 0:
        monotone("deeper lookahead is slower", (units, depth - 1, bw, lat, budget), False)

    # 5. Staging room never hurts: a larger budget is never slower.
    if budget is not None:
        monotone("more staging room is slower", (units, depth, bw, lat, budget * 4 + 1e9), True)

    # 6. Bandwidth never hurts: a faster machine is never slower.
    monotone("doubling bandwidth is slower",
             (units, depth, {t: v * 2 for t, v in bw.items()}, lat, budget), True)

    return problems, anomalies


def regression_short_unit_lists(reference):
    """Unit lists shorter than depth+1, where the budget walk-back must clamp.

    j = i - depth - 1 bounds the walk, so it can reach up to depth+1 units below
    zero; without a clamp at unit 0 it indexes units[] from the end and then
    raises IndexError. With a depth of 32, any step under 33 units hits this
    path (the shipped 72B step has 163 units, the 7B step 67). Kept as a named
    case so coverage does not depend on the fuzzer generating short lists.
    """
    bw = {"CXL Device NAND": 5e9, "Host DRAM": 38.4e9}
    lat = {"CXL Device NAND": 1547e-9, "Host DRAM": 200e-9}
    problems = []
    for n in range(1, 40):
        units = [({"CXL Device NAND": 4e6, "Host DRAM": 1e6}, 1e-4)] * n
        for depth in (32, 64):
            for budget in (1e12, 4e9, 0.0):
                try:
                    got = pipelined_time_s(units, depth, bw, lat, inflight_budget=budget)
                except Exception as e:
                    problems.append(f"n={n} depth={depth} budget={budget}: {type(e).__name__}: {e}")
                    continue
                ref = reference(units, depth, bw, lat, budget)
                if abs(got - ref) > 1e-12 * max(1.0, abs(ref)):
                    problems.append(f"n={n} depth={depth} budget={budget}: {got} vs {ref}")
    return problems


def _rel(a, b):
    return abs(a - b) / max(1e-12, abs(a), abs(b))


def main(n_cases):
    if BUS_ORDER not in REFERENCES:
        print(f"no reference for BUS_ORDER={BUS_ORDER!r}; known: {sorted(REFERENCES)}")
        return 1
    blocks = getattr(pipeline, "PIPELINE_BLOCKS", 1)
    if blocks != 1:
        print(f"the references model one transfer block per unit; "
              f"PIPELINE_BLOCKS={blocks} has none")
        return 1
    reference = REFERENCES[BUS_ORDER]
    print(f"BUS_ORDER={BUS_ORDER} NAND_PASS_THROUGH={NAND_PASS_THROUGH} "
          f"DEVICE_SERIAL={int(DEVICE_SERIAL)} STAGE_NAND_ONLY={int(STAGE_NAND_ONLY)} "
          f"STREAM_COMPUTE={int(STREAM_COMPUTE)} GPU_LINK_BPS={GPU_LINK_BPS:g}"
          f"  (reference: {reference.__name__})")
    if float(os.environ.get("GPU_LINK_BPS", "0")) and not pipeline.GPU_LINK_BPS:
        print("  GPU_LINK_BPS is set but pipeline.py dropped the GPU link")
        return 1

    reg = regression_short_unit_lists(reference)
    for r in reg[:5]:
        print(f"  REGRESSION (unit list shorter than depth): {r}")
    print(f"  regression, unit lists 1..39 against depths 32 and 64: "
          f"{'PASS' if not reg else str(len(reg)) + ' FAILURES'}")

    rng = random.Random(20260818)
    diffs = bad = reversals = 0
    worst = 0.0
    for k in range(n_cases):
        units, depth, bw, lat, budget = random_case(rng)
        got = pipelined_time_s(units, depth, bw, lat, inflight_budget=budget)
        ref = reference(units, depth, bw, lat, budget)

        rel = _rel(got, ref)
        worst = max(worst, rel)
        if rel > 1e-9:
            diffs += 1
            if diffs <= 3:
                print(f"  DIFFERENTIAL case {k}: engine {got:.9g} vs reference {ref:.9g} "
                      f"(rel {rel:.3g}) depth={depth} budget={budget}")

        problems, anomalies = bounds(units, depth, bw, lat, budget, got, reference)
        for p in problems:
            bad += 1
            if bad <= 5:
                print(f"  BOUND case {k}: {p}")
        for a in anomalies:
            reversals += 1
            if reversals <= 3:
                print(f"  ANOMALY case {k} (fill permits it; the reference agrees): {a}")

    # Work conservation on single-tier unit lists. A property of fill only:
    # under fifo a bus also waits for the links its transfer crosses, so a
    # binding link holds it idle, and the count is shown for information.
    rng = random.Random(20261105)
    idle_waits = 0
    for k in range(n_cases):
        units, depth, bw, lat, budget = random_single_tier_case(rng)
        got = pipelined_time_s(units, depth, bw, lat, inflight_budget=budget)
        want = work_conserving_time(units, depth, bw, lat, budget)
        if _rel(got, want) > 1e-9:
            idle_waits += 1
            if idle_waits <= 3 and BUS_ORDER == "fill":
                print(f"  WORK CONSERVATION case {k}: engine {got:.9g} vs back-to-back "
                      f"{want:.9g} depth={depth} budget={budget} bw={bw}")

    print(f"\n{n_cases} random configurations")
    print(f"  differential vs independent reference: "
          f"{n_cases - diffs} agree, {diffs} differ (worst rel {worst:.3g})")
    print(f"  hardware-model bounds: {bad} violations")
    if BUS_ORDER == "fill":
        print(f"  monotonicity reversals fill permits, each reproduced by the reference: "
              f"{reversals}")
        print(f"  work conservation, single-tier lists: {n_cases - idle_waits} hold, "
              f"{idle_waits} violate")
    else:
        print(f"  work conservation, single-tier lists: not required under {BUS_ORDER} "
              f"({idle_waits} of {n_cases} would violate it)")
    ok = diffs == 0 and bad == 0 and not reg and (BUS_ORDER != "fill" or idle_waits == 0)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000))
