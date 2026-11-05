"""
Lower bound on decode step time for any policy (Sec. V-B).

Every decode step reads each weight once (B rows of the embedding table) and
the whole KV cache, whatever the policy. In steady state a DRAM tier holding C
bytes serves at most C distinct bytes per step without refilling them from a
slower tier, and every tier moves at most its rate times the step time. So for
any placement, static or dynamic, any prefetch depth and any reserve, a step
of S bytes takes at least the smallest T with

    S <= min(C_gpu, T r_gpu) + min(C_host, T r_host) + D(T)

where D(T) is the most the device serves in T: its DRAM hits xd <= C_dev and
NAND reads xn with xd / r_dram + p xn / R_dram <= T (each NAND byte crosses the
device DRAM p times, NAND_PASS_THROUGH), xn <= T r_nand and xd + xn <= T times
the host link. Rates r include the per-chunk latency the engine charges. With
an accelerator link cap, every byte outside the GPU also crosses that link. T
is also at least the step's compute. The bound is the mean over the
generation's steps, as the KV cache grows.

Run inside a configured tree, as the harness runs a simulator:
    python bound.py        # prints "Decode bound: X t/s"
"""
import itertools
import os

from model_cfg import build_layers, bytes_read, BYTES_PER_PARAM
from sim_cfg import (TOKENS, BATCH_SIZE, cpu_freq_hz, cpu_cores,
                     flops_per_cycle_per_core, parallel_efficiency,
                     gpu_hbm_capacity_bytes, host_dram_capacity_bytes,
                     cxl_dev_dram_capacity_bytes)
from tiers import (GPU_HBM, HOST_DRAM, CXL_DEVICE_DRAM, CXL_DEVICE_NAND,
                   CXL_HOST_LINK_CEILING, IO_CHUNK_BYTES)

PREFILL_TOKENS = 512
PASS = int(os.environ.get("NAND_PASS_THROUGH", "1"))
SERIAL = os.environ.get("DEVICE_SERIAL", "0") == "1"
GPU_LINK = float(os.environ.get("GPU_LINK_BPS", "0")) if gpu_hbm_capacity_bytes else 0.0


def rate(tier):
    """Bytes/s of a long read from `tier`, per-chunk latency included."""
    return 1.0 / (1.0 / tier.bw_Bps + tier.chunk_latency_s / IO_CHUNK_BYTES)


def device_max(T):
    """Most bytes the device serves in T seconds (a two-variable LP, solved at
    its vertices)."""
    rd, rn = rate(CXL_DEVICE_DRAM), rate(CXL_DEVICE_NAND)
    dram_per_nand = 1.0 / rn if SERIAL else PASS / CXL_DEVICE_DRAM.bw_Bps
    rows = [(1, 0, cxl_dev_dram_capacity_bytes), (0, 1, rn * T),
            (1, 1, CXL_HOST_LINK_CEILING * T), (1 / rd, dram_per_nand, T),
            (-1, 0, 0), (0, -1, 0)]
    best = 0.0
    for (a1, b1, c1), (a2, b2, c2) in itertools.combinations(rows, 2):
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-30:
            continue
        x, y = (c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det
        if all(a * x + b * y <= c * (1 + 1e-9) + 1e-3 for a, b, c in rows):
            best = max(best, x + y)
    return best


def step_bound_s(S, compute_s):
    """Smallest step time in which S bytes can be read and compute_s run."""
    def fits(T):
        g = min(gpu_hbm_capacity_bytes, rate(GPU_HBM) * T) if gpu_hbm_capacity_bytes else 0.0
        rest = min(host_dram_capacity_bytes, rate(HOST_DRAM) * T) + device_max(T)
        if GPU_LINK:
            rest = min(rest, GPU_LINK * T)
        return g + rest >= S
    if fits(compute_s):
        return compute_s
    lo, hi = compute_s, 1.0
    while not fits(hi):
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if fits(mid) else (mid, hi)
    return hi


def main():
    layers = build_layers(sequence_length=PREFILL_TOKENS)
    weights = sum(bytes_read(L, BATCH_SIZE) for L in layers)
    kv_per_token = sum(2 * L.get("kv_heads", 8) * L.get("head_dim", 128) * BYTES_PER_PARAM
                       for L in layers if L.get("kv_cache_bytes", 0) > 0) * BATCH_SIZE
    compute_s = sum(L["flops"] for L in layers) * BATCH_SIZE / (
        cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency)
    steps = [step_bound_s(weights + kv_per_token * (PREFILL_TOKENS + t), compute_s)
             for t in range(TOKENS)]
    mean = sum(steps) / len(steps)
    print(f"Step bound: first {steps[0]:.4f} s, last {steps[-1]:.4f} s, "
          f"compute {compute_s:.4f} s, weights {weights / 2**30:.2f} GiB")
    print(f"Decode bound: {BATCH_SIZE / mean:.6f} t/s")


if __name__ == "__main__":
    main()
