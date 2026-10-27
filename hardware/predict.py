"""
Predict reserve_bench.py's measurements with the paper's timing engine.

The engine (pipeline.pipelined_time_s) is given the benchmark's own layout and
its measured rates: NVMe read bandwidth, host-to-GPU copy bandwidth and the
per-layer GPU time. Each layer is one unit. NVMe-resident layers carry their
bytes on the NVMe tier; every layer's compute is its host-to-GPU copy plus its
matmul, because the benchmark copies a layer just before using it. Reads may be
issued K units ahead while the bytes in flight fit in the staging slots.

    python predict.py reserve_bench.csv [meta.json]   # default: *_meta.json next to the csv
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline import pipelined_time_s     # noqa: E402

STEPS = 3


def spread(n, k):
    if k <= 0:
        return set()
    step = n / k
    return {min(n - 1, int(i * step + step / 2)) for i in range(k)}


def predict(meta, row):
    n, lb = meta["layers"], meta["layer_bytes"]
    nvme = spread(n, int(row["nvme_layers"])) if int(row["nvme_layers"]) < n else set(range(n))
    comp = lb / meta["h2d_Bps"] + meta["matmul_s"]
    units = [({"NVMe": float(lb)} if i in nvme else {}, comp) for i in range(n)]
    # The NVMe rate at the concurrency this run reaches: one read in flight per
    # staging slot, measured independently by reserve_bench's calibration.
    rates = {int(k): v for k, v in meta.get("nvme_Bps_by_inflight", {}).items()}
    slots = int(row["slots"])
    nv = rates[max(k for k in rates if k <= slots)] if rates else meta["nvme_Bps"]
    bw = {"NVMe": nv}
    budget = int(row["slots"]) * lb
    k = int(row["K"])
    one = pipelined_time_s(units, k, bw, inflight_budget=budget)
    many = pipelined_time_s(units * STEPS, k, bw, inflight_budget=budget)
    return (many - one) / (STEPS - 1)       # steady state, without the cold start


def main(path):
    meta = json.load(open(sys.argv[2] if len(sys.argv) > 2 else path.replace(".csv", "_meta.json")))
    rows = list(csv.DictReader(open(path)))
    errs = []
    print(f"{meta['gpu']}: NVMe {meta['nvme_Bps'] / 1e9:.2f} GB/s, "
          f"host-to-GPU {meta['h2d_Bps'] / 1e9:.2f} GB/s, matmul {meta['matmul_s'] * 1e3:.2f} ms")
    print(f"{'reserve':>8}{'K':>4}{'measured s':>12}{'predicted s':>13}{'error':>8}")
    for r in rows:
        m, p = float(r["step_s"]), predict(meta, r)
        errs.append((p - m) / m)
        print(f"{float(r['reserve']):>8.2f}{int(r['K']):>4}{m:>12.3f}{p:>13.3f}{(p - m) / m * 100:>7.1f}%")
    best_m = min(rows, key=lambda r: float(r["step_s"]))
    best_p = min(rows, key=lambda r: predict(meta, r))
    print(f"\nmean |error| {sum(abs(e) for e in errs) / len(errs) * 100:.1f}%, "
          f"max {max(abs(e) for e in errs) * 100:.1f}%")
    print(f"best measured: reserve {float(best_m['reserve']):.2f} K={best_m['K']}   "
          f"best predicted: reserve {float(best_p['reserve']):.2f} K={best_p['K']}")


if __name__ == "__main__":
    main(sys.argv[1])
