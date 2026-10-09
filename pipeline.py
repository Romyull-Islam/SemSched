"""
Step timing engine shared by every policy.

Accelerator memory, the host memory controller and the CXL device are
independent buses, so reads issued to different tiers proceed concurrently.
Summing bytes(tier) / bw(tier) over all tiers would make step time depend on
total bytes alone, whatever the placement.

How much of that concurrency a policy captures depends on how far ahead it can
issue a read: a transfer for layer i can only overlap work that runs before
layer i. The lookahead K comes from each system's paper or artifact:

    FlexGen    K=1   Algorithm 1 overlaps "the weights load of the next layer"
    LLMFlash   K=5   sliding window of k=5 tokens (Sec 3.2)
    SemSched   K=32  PREFETCH_QUEUE_DEPTH, 16-thread asynchronous I/O pool
    LIA        K=1   next-layer double buffer in the ISCA'25 artifact
    CXLAimPod  K=4   sequential device-side prefetch window

Two constraints apply. Transfers on the same tier serialize (one bus, one
queue), so issuing early cannot create bandwidth. And CXL device DRAM and its
NAND backend share the host link, which caps their combined rate at any
lookahead depth.
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
    tier_bw          {tier: bytes/s}; tiers missing here are not charged.
    tier_lat         {tier: seconds per chunk_bytes chunk}.
    inflight_budget  bytes that may be fetched but not yet consumed; None means
                     unbounded.

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
        # needing it. Depth 0 starts it when the preceding unit finishes
        # computing (demand paging); depth d issues it at the end of unit i-d-1.
        j = i - prefetch_depth - 1
        issue_at = comp_done[j] if j >= 0 else 0.0

        # Prefetched bytes need somewhere to live until they are consumed, so
        # `inflight_budget` limits how far ahead the issue point can move. For
        # example, SemSched's device cache has 0.02-0.36 GB free against
        # sub-layers of 0.4-0.9 GB, far from the ~14 GB an unbounded K=32 needs.
        if inflight_budget is not None:
            # Issuing unit i's read at comp_done[m] leaves units m+1..i fetched
            # but not yet consumed. Start at m = i-1 (room for unit i alone) and
            # walk back while the accumulated bytes still fit.
            m = i - 1
            acc = sum(units[i][0].values())
            # m >= 0 is required: a negative m would index units[] from the end.
            # Reaching m < 0 means everything back to unit 0 fits, i.e. issue
            # at t=0.
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
