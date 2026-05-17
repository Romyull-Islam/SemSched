"""
cxl_link.py — full-duplex CXL 2.0 link queueing model.

Models CXL.mem as a full-duplex link with two independent lanes:
    Rx (read) lane  — weight reads from CXL device DRAM/NAND
    Tx (write) lane — KV-cache writes back to CXL device DRAM

Each lane has independent bandwidth (typical: ~27 GB/s effective per direction
on a PCIe 5.0 x8 link, sources: Wang et al. IPDPS'25; Liu et al. arXiv 2409.14317).

NOT YET WIRED INTO semduplex_scheduler.py — this is a candidate replacement
for the current DUPLEX_PENALTY = 1.15 multiplier-based duplex model. Verify
with test_cxl_link.py before integration.
"""


class CXLLink:
    """
    Full-duplex CXL link with independent Rx and Tx queues.

    Parameters
    ----------
    bw_gbps : float
        Per-lane effective bandwidth in GB/s. Default 27.0 = PCIe 5.0 x8
        effective sustained throughput per direction.
    txn_overhead_s : float
        Per-transaction protocol overhead (flit framing, flow-control credit
        accounting). Default 10ns; documented range 7–15ns for CXL 2.0.

    State
    -----
    rx_busy_until : float
        Wall-clock time at which Rx lane next becomes free.
    tx_busy_until : float
        Wall-clock time at which Tx lane next becomes free.
    rx_busy_total : float
        Cumulative Rx-busy time (for read utilization computation).
    tx_busy_total : float
        Cumulative Tx-busy time (for write utilization computation).
    t_end : float
        Latest activity time across both lanes (for utilization denominator).
    """

    def __init__(self, bw_gbps=27.0, txn_overhead_s=10e-9):
        self.bw = bw_gbps * 1e9
        self.txn_overhead = txn_overhead_s
        self.rx_busy_until = 0.0
        self.tx_busy_until = 0.0
        self.rx_busy_total = 0.0
        self.tx_busy_total = 0.0
        self.t_end = 0.0

    def schedule_read(self, t_now, n_bytes):
        """
        Issue a synchronous read on the Rx lane.

        Returns the time the caller must wait (exposed stall = end - t_now).
        Caller blocks until the read completes — this is on the critical path.
        """
        start = max(t_now, self.rx_busy_until) + self.txn_overhead
        dur = n_bytes / self.bw
        end = start + dur
        self.rx_busy_until = end
        self.rx_busy_total += dur
        self.t_end = max(self.t_end, end)
        return end - t_now

    def schedule_write_background(self, t_now, n_bytes, deadline):
        """
        Issue a background write on the Tx lane.

        Tx lane runs independently of Rx (full duplex), so writes do NOT
        block reads. The caller continues immediately; the write completes
        asynchronously. We track whether the write meets its deadline.

        Parameters
        ----------
        t_now : float
            Time at which the write is issued.
        n_bytes : int
            Write payload size.
        deadline : float
            Time by which the write must complete (typically the next decode
            step start time, since KV must be visible by then).

        Returns
        -------
        exposed_s : float
            0.0 if the write finishes by `deadline`. Otherwise, the wall-clock
            slip — i.e. how far past `deadline` the Tx lane is still busy.
            The caller (scheduler) is responsible for accounting this slip as
            an actual stall on the next dependent step.
        """
        start = max(t_now, self.tx_busy_until) + self.txn_overhead
        dur = n_bytes / self.bw
        end = start + dur
        self.tx_busy_until = end
        self.tx_busy_total += dur
        self.t_end = max(self.t_end, end)
        return max(0.0, end - deadline)

    def read_utilization_pct(self):
        return (self.rx_busy_total / self.t_end * 100.0) if self.t_end > 0 else 0.0

    def write_utilization_pct(self):
        return (self.tx_busy_total / self.t_end * 100.0) if self.t_end > 0 else 0.0

    def reset(self):
        self.rx_busy_until = 0.0
        self.tx_busy_until = 0.0
        self.rx_busy_total = 0.0
        self.tx_busy_total = 0.0
        self.t_end = 0.0
