"""
test_cxl_link.py — verification suite for cxl_link.CXLLink.

Run before plugging the queue model into semduplex_scheduler.py.

Each test states the property under test and the expected numerical result.
A test failure means the queue model is wrong — fix cxl_link.py first.

Usage:
    python test_cxl_link.py
"""
from cxl_link import CXLLink

PASS = 0
FAIL = 0


def check(name, got, want, tol=1e-6, comment=""):
    global PASS, FAIL
    ok = abs(got - want) <= tol
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    msg = f"  [{status}] {name:<48} got={got:.6g}  want={want:.6g}"
    if comment:
        msg += f"   ({comment})"
    print(msg)


def check_close(name, got, want, rtol=0.01, comment=""):
    global PASS, FAIL
    ok = abs(got - want) <= rtol * abs(want)
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    msg = f"  [{status}] {name:<48} got={got:.6g}  want≈{want:.6g}  rtol={rtol*100:.1f}%"
    if comment:
        msg += f"   ({comment})"
    print(msg)


# ---------------------------------------------------------------------------
# Test group 1 — Single-lane bandwidth correctness
# ---------------------------------------------------------------------------
def test_pure_read_bandwidth():
    """A single 27 GB read at 27 GB/s should take ~1.0 s."""
    print("\n[1.1] pure read bandwidth — Rx lane only")
    link = CXLLink(bw_gbps=27.0, txn_overhead_s=10e-9)
    stall = link.schedule_read(t_now=0.0, n_bytes=27e9)
    check("read 27 GB at 27 GB/s ≈ 1.0 s", stall, 1.0, tol=1e-6,
          comment="time = bytes/bw + txn_overhead")
    check("rx_busy_until ≈ 1.0 s", link.rx_busy_until, 1.0, tol=1e-6)
    check("tx_busy_until == 0 (no writes)", link.tx_busy_until, 0.0)


def test_pure_write_bandwidth():
    """A single 27 GB write at 27 GB/s should take ~1.0 s on Tx lane."""
    print("\n[1.2] pure write bandwidth — Tx lane only")
    link = CXLLink(bw_gbps=27.0, txn_overhead_s=10e-9)
    # Generous deadline so write doesn't expose
    exposed = link.schedule_write_background(t_now=0.0, n_bytes=27e9, deadline=10.0)
    check("write fits under generous deadline → 0 exposed", exposed, 0.0)
    check("tx_busy_until ≈ 1.0 s", link.tx_busy_until, 1.0, tol=1e-6)
    check("rx_busy_until == 0 (no reads)", link.rx_busy_until, 0.0)


# ---------------------------------------------------------------------------
# Test group 2 — Full-duplex independence (the core property)
# ---------------------------------------------------------------------------
def test_reads_and_writes_run_concurrently():
    """
    Read 1 s + write 0.5 s issued simultaneously → t_end = 1.0 s.
    On a half-duplex bus this would be 1.5 s; on full-duplex it must be 1.0 s.
    """
    print("\n[2.1] full-duplex: concurrent read+write on separate lanes")
    link = CXLLink(bw_gbps=27.0, txn_overhead_s=10e-9)
    # 27 GB read = 1.0s on Rx
    read_stall = link.schedule_read(t_now=0.0, n_bytes=27e9)
    # 13.5 GB write = 0.5s on Tx, issued at same t=0
    write_exposed = link.schedule_write_background(t_now=0.0, n_bytes=13.5e9, deadline=1.0)
    check("read stall ≈ 1.0 s (independent of write)", read_stall, 1.0, tol=1e-6)
    check("write fits under 1.0 s deadline → 0 exposed", write_exposed, 0.0)
    check("t_end ≈ 1.0 s (NOT 1.5 s)", link.t_end, 1.0, tol=1e-6,
          comment="proves Rx and Tx are independent")
    check("rx busy ≈ 1.0 s", link.rx_busy_total, 1.0, tol=1e-6)
    check("tx busy ≈ 0.5 s", link.tx_busy_total, 0.5, tol=1e-6)


def test_write_exceeds_deadline_exposes_remainder():
    """
    Read window [0, 1.0]; issue 2.0s write at t=0.
    Write end = 2.0s; deadline = 1.0s; exposed = 1.0s.
    Caller must charge the 1.0s as a stall on the dependent next step.
    """
    print("\n[2.2] write exceeds deadline → exposed remainder accounted")
    link = CXLLink(bw_gbps=27.0, txn_overhead_s=10e-9)
    link.schedule_read(t_now=0.0, n_bytes=27e9)   # read ends at 1.0s
    # 54 GB write = 2.0s on Tx, deadline = 1.0s
    exposed = link.schedule_write_background(t_now=0.0, n_bytes=54e9, deadline=1.0)
    check_close("exposed write slip ≈ 1.0 s", exposed, 1.0, rtol=1e-4,
                comment="exposed = max(0, tx_end - deadline)")


def test_tx_does_not_block_subsequent_read():
    """
    On full-duplex, a busy Tx must not delay an Rx-only operation.
    Schedule a 1s write at t=0, then immediately read 27e6 bytes.
    Read time must be ~0.001s (its natural duration), not 1.001s.
    """
    print("\n[2.3] Tx busy does NOT delay subsequent Rx operation")
    link = CXLLink(bw_gbps=27.0, txn_overhead_s=10e-9)
    link.schedule_write_background(t_now=0.0, n_bytes=27e9, deadline=10.0)
    read_stall = link.schedule_read(t_now=0.0, n_bytes=27e6)   # 1 MB / 27 GB/s = ~37 us
    check_close("read stall ≈ 0.001 s (NOT 1.001 s)", read_stall, 0.001, rtol=0.05,
                comment="separate Rx lane is immediately free")


# ---------------------------------------------------------------------------
# Test group 3 — Hand-calculable end-to-end scenarios
# ---------------------------------------------------------------------------
def test_hand_calculable_10_layer_loop():
    """
    Loop scenario approximating LLM decode:
      10 layers, each:
        - read 2.7 GB (= 0.1 s) on Rx
        - write 135 MB (= 0.005 s) on Tx (hides under read)
    Expected: t_end = 1.0 s, tx_busy_total = 0.05 s, Uwrite = 5.0%.
    """
    print("\n[3.1] 10-layer loop — hand-calculable 5% Uwrite")
    link = CXLLink(bw_gbps=27.0, txn_overhead_s=10e-9)
    t = 0.0
    for _ in range(10):
        read_stall = link.schedule_read(t, 2.7e9)
        # Write issued at current t, deadline = t + read_stall (i.e. next read start)
        link.schedule_write_background(t, 135e6, deadline=t + read_stall)
        t += read_stall
    check_close("total wall time ≈ 1.0 s", t, 1.0, rtol=1e-4)
    check_close("tx_busy_total ≈ 0.05 s", link.tx_busy_total, 0.05, rtol=1e-3)
    check_close("Uwrite ≈ 5.0 %", link.write_utilization_pct(), 5.0, rtol=1e-3,
                comment="measured from queue, not derived")
    check_close("Uread ≈ 100 %", link.read_utilization_pct(), 100.0, rtol=1e-3)


def test_paper_scale_72b_b128_estimate():
    """
    Rough 72B INT8 B=128 sanity check:
      80 layers, each:
        - read ~875 MB weights (= 0.032 s @ 27 GB/s)
        - write ~10 MB KV (= 370 us @ 27 GB/s)
    Expected: t_end ≈ 2.6 s, Uwrite ≈ 1.1%.

    Paper Table III claims Uwrite = 2.71% — this test computes an analytically
    bounded value; the actual Uwrite depends on which layers are CXL-resident
    and is set by the scheduler, not by this micro-test.
    """
    print("\n[3.2] 72B INT8 B=128 micro-estimate (analytic bound)")
    link = CXLLink(bw_gbps=27.0, txn_overhead_s=10e-9)
    bytes_per_layer_read = 875e6
    bytes_per_layer_write = 10e6
    t = 0.0
    for _ in range(80):
        read_stall = link.schedule_read(t, bytes_per_layer_read)
        link.schedule_write_background(t, bytes_per_layer_write, deadline=t + read_stall)
        t += read_stall
    expected_t = 80 * (bytes_per_layer_read / 27e9)
    expected_uwrite = (bytes_per_layer_write / bytes_per_layer_read) * 100.0
    check_close("80-layer wall time", t, expected_t, rtol=1e-3)
    check_close("Uwrite analytic ratio", link.write_utilization_pct(),
                expected_uwrite, rtol=1e-3,
                comment="= write_bytes / read_bytes when fully hidden")


# ---------------------------------------------------------------------------
# Test group 4 — Boundary / corner cases
# ---------------------------------------------------------------------------
def test_zero_byte_operations():
    """0-byte ops should still cost txn_overhead but no bandwidth time."""
    print("\n[4.1] zero-byte transactions cost overhead only")
    link = CXLLink(bw_gbps=27.0, txn_overhead_s=10e-9)
    stall = link.schedule_read(0.0, 0)
    check_close("0-byte read stall ≈ txn_overhead", stall, 10e-9, rtol=1e-6)


def test_back_to_back_reads_serialize_on_rx():
    """
    Two consecutive reads on the same Rx lane must serialize.
    Issue 0.5s read at t=0, then 0.5s read at t=0; second must end at 1.0s.
    """
    print("\n[4.2] back-to-back reads serialize on Rx lane")
    link = CXLLink(bw_gbps=27.0, txn_overhead_s=10e-9)
    link.schedule_read(0.0, 13.5e9)   # 0.5 s
    link.schedule_read(0.0, 13.5e9)   # second read queued behind
    check_close("second read ends at 1.0 s", link.rx_busy_until, 1.0, rtol=1e-3)


def test_utilization_zero_when_idle():
    """Before any operations, utilization is 0% (no division by zero)."""
    print("\n[4.3] idle link reports 0% utilization without crash")
    link = CXLLink(bw_gbps=27.0, txn_overhead_s=10e-9)
    check("idle Uwrite == 0", link.write_utilization_pct(), 0.0)
    check("idle Uread == 0", link.read_utilization_pct(), 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("CXLLink verification suite — cxl_link.py")
    print("=" * 70)

    test_pure_read_bandwidth()
    test_pure_write_bandwidth()
    test_reads_and_writes_run_concurrently()
    test_write_exceeds_deadline_exposes_remainder()
    test_tx_does_not_block_subsequent_read()
    test_hand_calculable_10_layer_loop()
    test_paper_scale_72b_b128_estimate()
    test_zero_byte_operations()
    test_back_to_back_reads_serialize_on_rx()
    test_utilization_zero_when_idle()

    print("\n" + "=" * 70)
    print(f"Result: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    if FAIL > 0:
        raise SystemExit(1)
