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
    TierSched  K=32  PREFETCH_QUEUE_DEPTH, 16-thread asynchronous I/O pool
    LIA        K=1   next-layer double buffer in the ISCA'25 artifact
    CXLAimPod  K=4   sequential device-side prefetch window
    ITME       K=5   three-slot ring, refilled as each layer is read

Two constraints apply. Transfers on the same tier serialize (one bus, one
queue), so issuing early cannot create bandwidth. And CXL device DRAM and its
NAND backend share the host link, which caps their combined rate at any
lookahead depth.
"""
import bisect
import os

from tiers import CXL_HOST_LINK_CEILING

# Tiers sharing one physical link, and the ceiling they share. CMM-H device DRAM
# and its NAND backend both answer across the same PCIe Gen5 x8 host interface,
# so 27 + 5 GB/s cannot both be realised: the link carries 31.5.
SHARED_LINK = {"CXL Device DRAM", "CXL Device NAND"}

# On CMM-H a NAND read is first copied into the DRAM cache and then returned
# from it (Soltaniyeh et al., HotStorage'25, Fig. 1b), so NAND bytes also
# occupy the device DRAM. NAND_PASS_THROUGH=1 charges them there, which caps the
# device's combined rate at the DRAM rate. This is the default; NAND_PASS_THROUGH=2
# charges them twice, as if the fill and the read-out each took a pass, and
# NAND_PASS_THROUGH=0 runs the two tiers in parallel instead.
NAND_PASS_THROUGH = int(os.environ.get("NAND_PASS_THROUGH", "1"))
DEVICE_DRAM, DEVICE_NAND = "CXL Device DRAM", "CXL Device NAND"

# By default the device serves its DRAM cache while it reads NAND. DEVICE_SERIAL=1
# makes it serve one request at a time instead: no cache hit is served while a
# NAND read is in flight, and no NAND read starts while a hit is being served.
DEVICE_SERIAL = os.environ.get("DEVICE_SERIAL", "0") == "1"

# Every byte read ahead holds staging room until it is consumed, by default.
# STAGE_NAND_ONLY=1 counts only NAND bytes against the in-flight budget: on
# CMM-H they are the only bytes the device reserve stages.
STAGE_NAND_ONLY = os.environ.get("STAGE_NAND_ONLY", "0") == "1"

# By default a unit computes only once all its bytes have landed, as when a GPU
# copies weights into its own memory first. STREAM_COMPUTE=1 lets a unit's
# compute run alongside its own reads instead, as a CPU computing on weights it
# loads in place would; the unit then ends when both are done.
STREAM_COMPUTE = os.environ.get("STREAM_COMPUTE", "0") == "1"

# With an accelerator, every byte not already in its memory crosses the GPU's
# PCIe link, so host DRAM and the device share that link too. GPU_LINK_BPS caps
# their combined rate (63e9 for PCIe 5.0 x16); 0, the default, leaves it uncapped.
# It applies only to configurations with an accelerator.
GPU_LINK_BPS = float(os.environ.get("GPU_LINK_BPS", "0"))
if GPU_LINK_BPS:
    from sim_cfg import gpu_hbm_capacity_bytes
    if not gpu_hbm_capacity_bytes:
        GPU_LINK_BPS = 0.0
GPU_TIER = "GPU HBM"

# How a bus orders its requests. By default ("fill") a request takes the earliest
# idle stretch of its bus at or after the time it can start, so a request that
# is ready early is served in a gap that an earlier-issued request, still
# waiting for another resource, left open: device DRAM serves hits while a NAND
# read queued behind other NAND reads waits, as a non-blocking cache serves hits
# under outstanding misses. BUS_ORDER=fifo serves every bus strictly in issue
# order instead, so a hit waits behind a queued NAND fill (Table IV).
BUS_ORDER = os.environ.get("BUS_ORDER", "fill")

# Transfer blocks per unit for callers that do not choose (see pipelined_time_s).
PIPELINE_BLOCKS = int(os.environ.get("PIPELINE_BLOCKS", "1"))


class _Bus:
    """Booked busy intervals of one bus, sorted and disjoint."""

    def __init__(self):
        self.iv = []

    def book(self, t, dur):
        """Book `dur` seconds in the earliest idle stretch starting at or after
        t; return its start."""
        if dur <= 0:
            return t
        iv = self.iv
        k = bisect.bisect_right(iv, (t, float("inf"))) - 1
        if k >= 0 and iv[k][1] > t:
            t = iv[k][1]
        k += 1
        while k < len(iv) and iv[k][0] < t + dur:
            t = max(t, iv[k][1])
            k += 1
        iv.insert(k, (t, t + dur))
        # Merge with neighbours it touches, so a backlog stays one interval.
        if k + 1 < len(iv) and iv[k + 1][0] == iv[k][1]:
            iv[k] = (iv[k][0], iv[k + 1][1])
            del iv[k + 1]
        if k > 0 and iv[k - 1][1] == iv[k][0]:
            iv[k - 1] = (iv[k - 1][0], iv[k][1])
            del iv[k]
        return t


def staged_bytes(by_tier):
    """Bytes a read issued ahead holds in staging room until it is consumed."""
    return by_tier.get(DEVICE_NAND, 0.0) if STAGE_NAND_ONLY else sum(by_tier.values())


def pipelined_time_s(units, prefetch_depth, tier_bw, tier_lat=None,
                     chunk_bytes=256 * 1024, inflight_budget=None, demand=(),
                     blocks=None, trace=None):
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
    demand           indices of units whose transfers are not prefetched: they
                     are issued when the preceding unit finishes, as on a miss.
    blocks           split every unit into this many equal blocks, each moved and
                     computed on its own (a weight matrix splits by output
                     columns), so a block computes as soon as it lands and
                     read-ahead is staged a block at a time. prefetch_depth stays
                     in units. None takes PIPELINE_BLOCKS (default 1).
    trace            a list to append the schedule to (BUS_ORDER=fill only):
                     ("xfer", block, bus, start, end) per bus booking and
                     ("comp", block, None, start, end) per compute.

    Returns the completion time of the last unit's compute.
    """
    k = PIPELINE_BLOCKS if blocks is None else int(blocks)
    if k > 1:
        # Block j of unit i is issued when unit i would be: at the end of unit
        # i - prefetch_depth - 1, i.e. block j looks prefetch_depth*k + j blocks
        # ahead. A demand unit's blocks wait for the end of the preceding unit.
        units = [({t: v / k for t, v in by.items()}, c / k)
                 for by, c in units for _ in range(k)]
        demand = {i * k + j: i * k - 1 for i in demand for j in range(k)}
        prefetch_depth = [prefetch_depth * k + j for _ in range(len(units) // k)
                          for j in range(k)]
    else:
        demand = {i: i - 1 for i in demand}
    n = len(units)
    if n == 0:
        return 0.0
    if BUS_ORDER != "fifo":
        return _pipelined_fill(units, prefetch_depth, tier_bw, tier_lat or {},
                               chunk_bytes, inflight_budget, demand, trace)
    assert trace is None, "trace needs BUS_ORDER=fill"
    tier_lat = tier_lat or {}
    staged_prefix = _staged_prefix(units) if inflight_budget is not None else None
    tier_free = {}           # when each tier's bus is next idle
    link_free = 0.0          # when the shared CXL host link is next idle
    gpu_link_free = 0.0      # when the GPU's PCIe link is next idle
    comp_done = [0.0] * n    # when each unit's compute finishes
    prev_comp_end = 0.0

    for i, (by_tier, comp_s) in enumerate(units):
        # A transfer may be issued once we are within `prefetch_depth` units of
        # needing it. Depth 0 starts it when the preceding unit finishes
        # computing (demand paging); depth d issues it at the end of unit i-d-1.
        j = i - (prefetch_depth[i] if isinstance(prefetch_depth, list)
                 else prefetch_depth) - 1
        issue_at = comp_done[j] if j >= 0 else 0.0

        # Prefetched bytes need somewhere to live until they are consumed, so
        # `inflight_budget` limits how far ahead the issue point can move. For
        # example, a filled device has less room free than one INT8 sub-layer
        # (0.15-0.73 GB), far from the ~14 GB an unbounded K=32 needs.
        if inflight_budget is not None:
            # Issuing unit i's read at comp_done[m] leaves units m+1..i fetched
            # but not yet consumed, so m is the latest unit whose bytes, added
            # to those of m+1..i, no longer fit (or j, or -1: issue at t=0).
            m = _budget_floor(staged_prefix, i, j, inflight_budget)
            issue_at = max(issue_at, comp_done[m] if m >= 0 else 0.0)

        if i in demand:
            f = demand[i]
            issue_at = max(issue_at, comp_done[f] if f >= 0 else 0.0)

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
            to_gpu = GPU_LINK_BPS and tier != GPU_TIER
            if tier in SHARED_LINK:
                start = max(start, link_free)
            if to_gpu:
                start = max(start, gpu_link_free)
            # Hold each shared link for the time this transfer occupies it at
            # the link's own rate.
            if tier in SHARED_LINK:
                link_free = start + nbytes / CXL_HOST_LINK_CEILING
            if to_gpu:
                gpu_link_free = start + nbytes / GPU_LINK_BPS
            finish = start + dur
            tier_free[tier] = finish
            if DEVICE_SERIAL and tier in SHARED_LINK:
                # One request at a time: neither device tier starts another
                # transfer until this one is done.
                tier_free[DEVICE_DRAM] = tier_free[DEVICE_NAND] = finish
            if (NAND_PASS_THROUGH and not DEVICE_SERIAL and tier == DEVICE_NAND
                    and tier_bw.get(DEVICE_DRAM)):
                # The cache fill and read-out stream alongside the NAND read but
                # hold the DRAM for these bytes at the DRAM's own rate, once per pass.
                d_start = max(start, tier_free.get(DEVICE_DRAM, 0.0))
                d_finish = d_start + NAND_PASS_THROUGH * nbytes / tier_bw[DEVICE_DRAM]
                tier_free[DEVICE_DRAM] = d_finish
                finish = max(finish, d_finish)
            ready = max(ready, finish)

        # Compute is one engine: units execute in order, each starting when its
        # bytes have landed and the previous unit has finished.
        if STREAM_COMPUTE:
            start_comp = max(issue_at, prev_comp_end)
            prev_comp_end = max(start_comp + comp_s, ready)
        else:
            start_comp = max(ready, prev_comp_end)
            prev_comp_end = start_comp + comp_s
        comp_done[i] = prev_comp_end

    return prev_comp_end


def _staged_prefix(units):
    """P[k] = staged bytes of units 0..k-1."""
    p = [0.0]
    for by, _ in units:
        p.append(p[-1] + staged_bytes(by))
    return p


def _budget_floor(p, i, j, budget):
    """The unit whose compute must end before unit i's read may issue: the
    latest m in (j, i-1] such that units m..i stage more than `budget` bytes,
    else max(j, -1). Equivalent to walking back from i-1 while the running
    total fits, in O(log n)."""
    lo = j if j >= -1 else -1
    m = bisect.bisect_left(p, p[i + 1] - budget) - 1    # latest m: p[m] < p[i+1] - budget
    if m > i - 1:
        m = i - 1
    return m if m > lo else lo


def _issue_time(units, i, prefetch_depth, comp_done, inflight_budget, demand,
                staged_prefix=None):
    """When unit i's reads may start: `prefetch_depth` units ahead of its
    compute, no earlier than the staging budget allows, and on demand for the
    units in `demand`."""
    j = i - (prefetch_depth[i] if isinstance(prefetch_depth, list)
             else prefetch_depth) - 1
    issue_at = comp_done[j] if j >= 0 else 0.0
    if inflight_budget is not None:
        m = _budget_floor(staged_prefix, i, j, inflight_budget)
        issue_at = max(issue_at, comp_done[m] if m >= 0 else 0.0)
    if i in demand:
        f = demand[i]
        issue_at = max(issue_at, comp_done[f] if f >= 0 else 0.0)
    return issue_at


def _pipelined_fill(units, prefetch_depth, tier_bw, tier_lat, chunk_bytes,
                    inflight_budget, demand, trace=None):
    """pipelined_time_s with BUS_ORDER=fill: the same issue rule, charges and
    compute model, with each bus booked by idle stretch rather than in order."""
    n = len(units)
    buses = {}
    if DEVICE_SERIAL:
        buses[DEVICE_DRAM] = buses[DEVICE_NAND] = _Bus()   # one request at a time
    link, gpu_link = _Bus(), _Bus()
    comp_done = [0.0] * n
    prev_comp_end = 0.0
    staged_prefix = _staged_prefix(units) if inflight_budget is not None else None
    for i, (by_tier, comp_s) in enumerate(units):
        issue_at = _issue_time(units, i, prefetch_depth, comp_done,
                               inflight_budget, demand, staged_prefix)
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
            bus = buses.get(tier)
            if bus is None:
                bus = buses[tier] = _Bus()
            start = bus.book(issue_at, dur)
            finish = start + dur
            if trace is not None:
                trace.append(("xfer", i, tier, start, finish))
            # A shared link carries these bytes at its own rate, in its own
            # idle stretch at or after the transfer starts.
            if tier in SHARED_LINK:
                hold = nbytes / CXL_HOST_LINK_CEILING
                finish = max(finish, link.book(start, hold) + hold)
            if GPU_LINK_BPS and tier != GPU_TIER:
                hold = nbytes / GPU_LINK_BPS
                finish = max(finish, gpu_link.book(start, hold) + hold)
            if (NAND_PASS_THROUGH and not DEVICE_SERIAL and tier == DEVICE_NAND
                    and tier_bw.get(DEVICE_DRAM)):
                # The cache fill and read-out hold the device DRAM for these
                # bytes at its own rate, once per pass.
                hold = NAND_PASS_THROUGH * nbytes / tier_bw[DEVICE_DRAM]
                dram = buses.get(DEVICE_DRAM)
                if dram is None:
                    dram = buses[DEVICE_DRAM] = _Bus()
                d0 = dram.book(start, hold)
                finish = max(finish, d0 + hold)
                if trace is not None:
                    trace.append(("xfer", i, DEVICE_DRAM, d0, d0 + hold))
            ready = max(ready, finish)
        if STREAM_COMPUTE:
            start_comp = max(issue_at, prev_comp_end)
            prev_comp_end = max(start_comp + comp_s, ready)
        else:
            start_comp = max(ready, prev_comp_end)
            prev_comp_end = start_comp + comp_s
        comp_done[i] = prev_comp_end
        if trace is not None:
            trace.append(("comp", i, None, prev_comp_end - comp_s, prev_comp_end))
    return prev_comp_end
