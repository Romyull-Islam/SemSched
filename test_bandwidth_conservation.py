"""
Physical-plausibility invariants: no modeled transfer may move bytes faster
than the tier it crosses can carry them.

Includes a regression test for dividing the whole NAND staging time, bandwidth
term included, by the thread-pool size, which implies ~80 GB/s from a 5 GB/s
device.

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


# The transfer model itself

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


# Phase-1 staging

def stage_schedule(layer_bytes, n_threads):
    """Phase-1 staging time with per-chunk latency amortized over n_threads.

    semduplex_scheduler.py keeps the latency serial, which is never faster than
    this, so the bounds below also cover it.
    """
    t = 0.0
    for b in layer_bytes:
        bw_term = b / CXL_SSD_NAND.bw_Bps
        lat_term = (math.ceil(b / IO_CHUNK_BYTES)
                    * CXL_SSD_NAND.chunk_latency_s) / n_threads
        t += bw_term + lat_term
    return t


@pytest.mark.parametrize("n_threads", [1, 4, 16, 64, 1024])
def test_staging_respects_nand_bandwidth_at_any_thread_count(n_threads):
    """Threads amortize per-chunk latency; they cannot beat the bandwidth floor."""
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
    """The `dur / |T|` formula must violate the bandwidth floor.

    If it does not, the invariant is too weak to catch that bug.
    """
    layers = [1.41 * GB] * 71
    total = sum(layers)
    buggy = sum(transfer_time_s(int(b), CXL_SSD_NAND) / 16 for b in layers)
    assert effective_bw(total, buggy) > CXL_SSD_NAND.bw_Bps, (
        "the pre-fix `dur / |T|` formula should violate the bandwidth floor")
