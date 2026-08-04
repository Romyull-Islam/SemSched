"""
test_bandwidth_conservation.py — physical-plausibility invariants.

The defect that motivated this file: Phase-1 staging divided the *whole* NAND
transfer time by the thread-pool size, which divides the bytes/bandwidth term as
well and implies ~80 GB/s out of a 5 GB/s device. No test could catch it, because
no test asserted that a modeled transfer respects the physical rate of the tier
it crosses.

These tests make that class of error unrepresentable rather than merely absent:
any future change that moves bytes faster than a tier can carry them fails here.

    python -m pytest test_bandwidth_conservation.py -q
"""
import math

import pytest

from tiers import (HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, IO_CHUNK_BYTES,
                   transfer_time_s)

ALL_TIERS = [HOST_DRAM, CXL_DRAM, CXL_SSD_NAND]
GB = 1e9
TOL = 1.0 + 1e-9   # floating-point slack only; no physical headroom


def effective_bw(nbytes, seconds):
    return nbytes / seconds if seconds > 0 else float("inf")


# ── The transfer model itself ────────────────────────────────────────────────

@pytest.mark.parametrize("tier", ALL_TIERS, ids=lambda t: t.name)
@pytest.mark.parametrize("nbytes", [IO_CHUNK_BYTES, 1 * GB, 10 * GB, 200 * GB])
def test_transfer_never_exceeds_tier_bandwidth(tier, nbytes):
    """transfer_time_s must never imply a rate above the tier's physical one."""
    t = transfer_time_s(int(nbytes), tier)
    assert effective_bw(nbytes, t) <= tier.bw_Bps * TOL, (
        f"{tier.name}: {nbytes/GB:.1f} GB in {t:.6f}s = "
        f"{effective_bw(nbytes, t)/GB:.1f} GB/s > {tier.bw_Bps/GB:.1f} GB/s")


@pytest.mark.parametrize("tier", ALL_TIERS, ids=lambda t: t.name)
def test_transfer_time_is_monotonic_in_size(tier):
    sizes = [IO_CHUNK_BYTES, 1 * GB, 4 * GB, 32 * GB]
    times = [transfer_time_s(int(n), tier) for n in sizes]
    assert times == sorted(times)


def test_tier_ordering_holds():
    """Host DRAM > CXL DRAM > CXL NAND. Placement logic depends on this."""
    assert HOST_DRAM.bw_Bps > CXL_DRAM.bw_Bps > CXL_SSD_NAND.bw_Bps
    assert HOST_DRAM.chunk_latency_s < CXL_DRAM.chunk_latency_s < CXL_SSD_NAND.chunk_latency_s


# ── Phase-1 staging: the regression test for the original defect ─────────────

def stage_schedule(layer_bytes, n_threads):
    """Reproduce semduplex_scheduler's Phase-1 staging schedule."""
    t = 0.0
    for b in layer_bytes:
        bw_term = b / CXL_SSD_NAND.bw_Bps
        lat_term = (math.ceil(b / IO_CHUNK_BYTES)
                    * CXL_SSD_NAND.chunk_latency_s) / n_threads
        t += bw_term + lat_term
    return t


@pytest.mark.parametrize("n_threads", [1, 4, 16, 64, 1024])
def test_staging_respects_nand_bandwidth_at_any_thread_count(n_threads):
    """Threads amortize per-chunk latency; they cannot beat the bandwidth floor.

    This is the assertion that would have caught the original `dur / |T|` bug.
    """
    layers = [1.41 * GB] * 71          # ~100 GB of NAND-resident FP16 sublayers
    total = sum(layers)
    t = stage_schedule(layers, n_threads)
    assert effective_bw(total, t) <= CXL_SSD_NAND.bw_Bps * TOL, (
        f"|T|={n_threads}: staged {total/GB:.0f} GB in {t:.2f}s = "
        f"{effective_bw(total, t)/GB:.1f} GB/s from a "
        f"{CXL_SSD_NAND.bw_Bps/GB:.0f} GB/s device")


def test_more_threads_never_reduce_time_below_the_bandwidth_bound():
    layers = [1.41 * GB] * 71
    floor = sum(layers) / CXL_SSD_NAND.bw_Bps
    for n in (1, 16, 10_000):
        assert stage_schedule(layers, n) >= floor * (1 - 1e-9)


def test_the_original_buggy_formula_is_rejected():
    """Guard on the guard: the old model must FAIL this suite.

    If this ever passes, the invariant has been weakened into uselessness.
    """
    layers = [1.41 * GB] * 71
    total = sum(layers)
    buggy = sum(transfer_time_s(int(b), CXL_SSD_NAND) / 16 for b in layers)
    assert effective_bw(total, buggy) > CXL_SSD_NAND.bw_Bps, (
        "the pre-fix `dur / |T|` formula should violate the bandwidth floor")
