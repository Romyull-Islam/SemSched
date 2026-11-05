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
    TierSched   K=32  PREFETCH_QUEUE_DEPTH, 16-thread asynchronous I/O pool
    LIA        K=1   next-layer double buffer in the ISCA'25 artifact
    CXLAimPod  K=4   sequential device-side prefetch window

Two constraints apply. Transfers on the same tier serialize (one bus, one
queue), so issuing early cannot create bandwidth. And CXL device DRAM and its
NAND backend share the host link, which caps their combined rate at any
lookahead depth.
"""
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


def staged_bytes(by_tier):
    """Bytes a read issued ahead holds in staging room until it is consumed."""
    return by_tier.get(DEVICE_NAND, 0.0) if STAGE_NAND_ONLY else sum(by_tier.values())


def pipelined_time_s(units, prefetch_depth, tier_bw, tier_lat=None,
                     chunk_bytes=256 * 1024, inflight_budget=None, demand=()):
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

    Returns the completion time of the last unit's compute.
    """
    n = len(units)
    if n == 0:
        return 0.0
    tier_lat = tier_lat or {}
    tier_free = {}           # when each tier's bus is next idle
    link_free = 0.0          # when the shared CXL host link is next idle
    gpu_link_free = 0.0      # when the GPU's PCIe link is next idle
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
        # example, a filled device has less room free than one INT8 sub-layer
        # (0.15-0.73 GB), far from the ~14 GB an unbounded K=32 needs.
        if inflight_budget is not None:
            # Issuing unit i's read at comp_done[m] leaves units m+1..i fetched
            # but not yet consumed. Start at m = i-1 (room for unit i alone) and
            # walk back while the accumulated bytes still fit.
            m = i - 1
            acc = staged_bytes(units[i][0])
            # m >= 0 is required: a negative m would index units[] from the end.
            # Reaching m < 0 means everything back to unit 0 fits, i.e. issue
            # at t=0.
            while m > j and m >= 0:
                nxt = staged_bytes(units[m][0])
                if acc + nxt > inflight_budget:
                    break
                acc += nxt
                m -= 1
            issue_at = max(issue_at, comp_done[m] if m >= 0 else 0.0)

        if i in demand:
            issue_at = max(issue_at, comp_done[i - 1] if i else 0.0)

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
