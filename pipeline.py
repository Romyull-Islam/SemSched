"""
pipeline.py — one timing engine, shared by every policy.

The previous model charged a step as sum(bytes(tier) / bw(tier)): every tier
read serialized behind every other. That is not the machine. Accelerator memory,
the host memory controller and the CXL device are independent buses, and reads
issued to different ones proceed at the same time. Charging them serially
overstates a step by up to 2.35x (INT8 16H+32C: 2.32 s serial against 0.99 s
when the tiers run concurrently) and, worse, makes the step time a function of
total bytes alone -- which is why every policy returned the same number no
matter how it placed them.

What a policy can actually capture of that concurrency is decided by how far
ahead it can issue a read. A transfer for layer i can only overlap work that
runs before layer i executes, so a scheduler that looks one layer ahead hides
one layer's worth of time and a scheduler that looks thirty-two ahead hides
thirty-two. That is a real difference between these systems and it is stated in
their papers, so it is a parameter here rather than an assumption:

    FlexGen    K=1   Algorithm 1 overlaps "the weights load of the next layer"
    LLMFlash   K=5   sliding window of k=5 tokens (Sec 3.2)
    SemSched   K=32  PREFETCH_QUEUE_DEPTH, 16-thread asynchronous I/O pool
    LIA        K=0   no weight prefetch described; CXL memory is read on demand
    CXLAimPod  K=0   pooled device memory, no prefetch described

Two constraints keep this honest. Transfers on the SAME tier serialize -- one
bus, one queue -- so a policy cannot conjure bandwidth by issuing early. And the
CXL device DRAM and its NAND backend share the host link, so their combined rate
is capped by it no matter how deep the lookahead.
"""
from tiers import CXL_HOST_LINK_CEILING

# Tiers sharing one physical link, and the ceiling they share. CMM-H device DRAM
# and its NAND backend both answer across the same PCIe Gen5 x8 host interface,
# so 27 + 5 GB/s cannot both be realised: the link carries 31.5.
SHARED_LINK = {"CXL Device DRAM", "CXL Device NAND"}


def pipelined_time_s(units, prefetch_depth, tier_bw, tier_lat=None,
                     chunk_bytes=256 * 1024, inflight_budget=None):
    """Time for one pass over `units` with reads issued `prefetch_depth` ahead.

    units            [(bytes_by_tier: {tier: bytes}, compute_s), ...] in
                     execution order.
    prefetch_depth   how many units ahead a transfer may be issued. 0 serializes
                     each transfer against the unit that needs it; a value >=
                     len(units) issues everything at t=0 and the pass costs the
                     slowest tier plus the compute tail.
    tier_bw          {tier: bytes/s}. tier_lat {tier: seconds per chunk}.

    Returns the completion time of the last unit's compute.
    """
    n = len(units)
    if n == 0:
        return 0.0
    tier_lat = tier_lat or {}
    tier_free = {}           # when each tier's bus is next idle
    link_free = 0.0          # when the shared CXL host link is next idle
    comp_done = [0.0] * n    # when each unit's compute finishes
    prev_comp_end = 0.0

    for i, (by_tier, comp_s) in enumerate(units):
        # A transfer may be issued once we are within `prefetch_depth` units of
        # needing it. Depth 0 means it starts only when the preceding unit has
        # finished computing -- demand paging, nothing overlapped. Depth d looks
        # d units further back, so the issue point is unit i-d-1.
        j = i - prefetch_depth - 1
        issue_at = comp_done[j] if j >= 0 else 0.0

        # Bytes fetched ahead of consumption have to live somewhere. A lookahead
        # of K units needs room for the units still in flight, and a scheduler
        # cannot prefetch into memory it does not have -- SemSched's device cache
        # has 0.02-0.36 GB free against sub-layers of 0.4-0.9 GB, so an unbounded
        # K=32 would be claiming a 14 GB staging buffer that does not exist.
        # `inflight_budget` caps how far ahead the issue point may actually move:
        # walk back from unit i only while the accumulated bytes fit.
        if inflight_budget is not None:
            # Issuing unit i's read at comp_done[m] leaves units m+1..i fetched
            # but not yet consumed, so that is what has to be buffered. Start at
            # m = i-1, which needs room for unit i alone, and walk back only
            # while the accumulated bytes still fit.
            m = i - 1
            acc = sum(units[i][0].values())
            # m >= 0 is load-bearing, not defensive: without it a negative m
            # indexes units[] from the END of the list, so the walk accumulated
            # some other unit's bytes and stopped in the wrong place, and once
            # m passed -len(units) it raised IndexError outright. Reaching
            # m < 0 means everything back to unit 0 fits, which is exactly the
            # case the issue_at guard below already reads as "issue at t=0".
            while m > j and m >= 0:
                nxt = sum(units[m][0].values())
                if acc + nxt > inflight_budget:
                    break
                acc += nxt
                m -= 1
            issue_at = max(issue_at, comp_done[m] if m >= 0 else 0.0)

        ready = issue_at
        for tier, nbytes in by_tier.items():
            if nbytes <= 0:
                continue
            bw = tier_bw.get(tier)
            if not bw:
                continue
            dur = nbytes / bw
            lat = tier_lat.get(tier, 0.0)
            if lat:
                dur += -(-nbytes // chunk_bytes) * lat
            start = max(issue_at, tier_free.get(tier, 0.0))
            if tier in SHARED_LINK:
                # Also wait for the shared host link, and hold it for the time
                # this transfer occupies it at the link's own rate.
                start = max(start, link_free)
                link_free = start + nbytes / CXL_HOST_LINK_CEILING
            finish = start + dur
            tier_free[tier] = finish
            ready = max(ready, finish)

        # Compute is one engine: units execute in order, each starting when its
        # bytes have landed and the previous unit has finished.
        start_comp = max(ready, prev_comp_end)
        prev_comp_end = start_comp + comp_s
        comp_done[i] = prev_comp_end

    return prev_comp_end
