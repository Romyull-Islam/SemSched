"""
Real-hardware check of the capacity-for-prefetch trade on a GPU server.

Host DRAM plays the fast tier, an NVMe SSD the slow tier, and the GPU computes.
A fixed budget of pinned host memory is split between resident weights and a
prefetch staging area; the benchmark sweeps the split (the reserve) and the
lookahead depth K, and times decode steps. Filling the budget with weights
(reserve 0) leaves one staging slot, the minimum any offloading runtime needs.

Weights are synthetic fp16 tensors with the layer sizes of a large model, so no
model download is needed. NVMe reads use O_DIRECT, so the page cache cannot
act as a hidden extra tier. With --pool-gb, the layers cycle through a smaller
set of files: every read still goes to the SSD, so a shared machine needs only
the pool's disk space, not the whole model's. Keep the pool well above the
SSD's internal cache (a few GB).

    python reserve_bench.py --dir /path/on/nvme --budget-gb 32
    python reserve_bench.py --dir /path/on/nvme --budget-gb 32 --reserves 0,0.05,0.1,0.2 --depths 1,8,32
    python reserve_bench.py --dir /tmp/rb --budget-gb 32 --pool-gb 16   # 16 GB on disk

Output: one line per (reserve, K) and reserve_bench.csv next to this script.
"""
import argparse
import concurrent.futures as cf
import csv
import json
import os
import time

import torch

CHUNK = 8 << 20                     # bytes per NVMe read request
COLS = 8192                         # weight matrix width (hidden size)


def make_files(d, n, nbytes):
    """Write n layer files of nbytes random data, then drop them from the page cache."""
    os.makedirs(d, exist_ok=True)
    buf = os.urandom(CHUNK)
    for i in range(n):
        p = os.path.join(d, f"layer_{i:03d}.bin")
        if os.path.exists(p) and os.path.getsize(p) == nbytes:
            continue
        with open(p, "wb") as f:
            for _ in range(nbytes // CHUNK):
                f.write(buf)
            f.flush()
            os.fsync(f.fileno())
    for i in range(n):
        fd = os.open(os.path.join(d, f"layer_{i:03d}.bin"), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


class Reader:
    """Reads layer files into pinned host buffers with O_DIRECT, many requests in
    flight. Layer i reads file i % files, so a pool smaller than the model works."""

    def __init__(self, d, files, threads):
        flags = os.O_RDONLY | getattr(os, "O_DIRECT", 0)
        try:
            fds = [os.open(os.path.join(d, f"layer_{i:03d}.bin"), flags) for i in range(files)]
            self.direct = bool(getattr(os, "O_DIRECT", 0))
        except OSError:
            fds = [os.open(os.path.join(d, f"layer_{i:03d}.bin"), os.O_RDONLY) for i in range(files)]
            self.direct = False
        self.fds = _Cycle(fds)
        self.pool = cf.ThreadPoolExecutor(threads)

    def _chunk(self, fd, view, off, size):
        got = os.preadv(fd, [view[off:off + size]], off)
        assert got == size, (got, size)

    def submit(self, layer, buf):
        view = memoryview(buf.numpy()).cast("B")
        size = view.nbytes
        if not self.direct:
            os.posix_fadvise(self.fds[layer], 0, 0, os.POSIX_FADV_DONTNEED)
        return [self.pool.submit(self._chunk, self.fds[layer], view, o, min(CHUNK, size - o))
                for o in range(0, size, CHUNK)]


class _Cycle:
    def __init__(self, fds):
        self.fds = fds

    def __getitem__(self, i):
        return self.fds[i % len(self.fds)]


def bandwidth(reader, buf, layers):
    t = time.perf_counter()
    for i in layers:
        for f in reader.submit(i, buf):
            f.result()
    return len(layers) * buf.numel() * buf.element_size() / (time.perf_counter() - t)


def bandwidth_inflight(reader, bufs, files, inflight, rounds=4):
    """NVMe read rate with `inflight` layers read at once, the concurrency a run
    with that many staging slots reaches. RAID-1 and multi-queue SSDs can be
    faster with more reads outstanding, so this is measured, not assumed."""
    n = inflight * rounds
    t = time.perf_counter()
    for r in range(rounds):
        futs = [f for j in range(inflight)
                for f in reader.submit((r * inflight + j) % files, bufs[j])]
        for f in futs:
            f.result()
    return n * bufs[0].numel() * bufs[0].element_size() / (time.perf_counter() - t)


def spread(n, k):
    """k of n positions, evenly spaced (the residue spreading of the design)."""
    if k <= 0:
        return set()
    step = n / k
    return {min(n - 1, int(i * step + step / 2)) for i in range(k)}


def run(args, reserve, depth, reader, pinned_pool, layer_bytes, gpu_w, x):
    n = args.layers
    budget = int(args.budget_gb * (1 << 30))
    slots = max(1, int(budget * reserve) // layer_bytes)
    resident_n = max(0, min(n, (budget - slots * layer_bytes) // layer_bytes))
    on_nvme = sorted(spread(n, n - resident_n)) if resident_n < n else []
    nvme = set(on_nvme)
    resident = [i for i in range(n) if i not in nvme]

    # Load the resident layers once; they stay for the whole run.
    res_buf = {}
    for k, i in enumerate(resident):
        b = pinned_pool[k]
        for f in reader.submit(i, b):
            f.result()
        res_buf[i] = b
    stage = pinned_pool[len(resident):len(resident) + slots]
    free = list(range(len(stage)))
    inflight = {}                    # global position -> (slot, futures)

    def issue(upto, p_now):
        for q in range(p_now, upto + 1):
            layer = q % n
            if layer not in nvme or q in inflight:
                continue
            if not free:
                return
            s = free.pop(0)
            inflight[q] = (s, reader.submit(layer, stage[s]))

    times = []
    for step in range(args.steps + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(n):
            p = step * n + i
            issue(p + depth, p)
            if i in nvme:
                s, futs = inflight.pop(p)
                for f in futs:
                    f.result()
                src = stage[s]
            else:
                src = res_buf[i]
            w = gpu_w[i % 2]
            w.copy_(src.view(-1), non_blocking=True)
            torch.matmul(x, w.view(-1, COLS).t())
            if i in nvme:
                torch.cuda.current_stream().synchronize()   # slot reusable once copied
                free.append(s)
        torch.cuda.synchronize()
        if step:                     # step 0 warms up
            times.append(time.perf_counter() - t0)
    for q, (s, futs) in list(inflight.items()):
        for f in futs:
            f.result()
        free.append(s)
        del inflight[q]
    step_s = sorted(times)[len(times) // 2]
    return {"reserve": reserve, "K": depth, "slots": slots, "resident_layers": len(resident),
            "nvme_layers": len(nvme), "nvme_GB_per_step": len(nvme) * layer_bytes / 1e9,
            "step_s": step_s, "tok_per_s": args.batch / step_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory on the NVMe drive")
    ap.add_argument("--layers", type=int, default=80)
    ap.add_argument("--layer-mb", type=int, default=800)
    ap.add_argument("--budget-gb", type=float, default=32)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--settle", type=float, default=60,
                    help="seconds to wait after writing the files before measuring")
    ap.add_argument("--calibrate-only", action="store_true",
                    help="measure the rates and exit without the sweep")
    ap.add_argument("--pool-gb", type=float, default=0,
                    help="disk space for weight files; 0 = one file per layer")
    ap.add_argument("--reserves", default="0,0.02,0.05,0.1,0.15,0.2,0.3,0.4")
    ap.add_argument("--depths", default="1,4,8,16,32")
    args = ap.parse_args()

    layer_bytes = args.layer_mb << 20
    assert layer_bytes % (COLS * 2) == 0
    budget = int(args.budget_gb * (1 << 30))
    total = args.layers * layer_bytes
    assert budget < total, "the budget must be smaller than the model, or nothing comes from NVMe"
    print(f"model {total / 1e9:.1f} GB in {args.layers} layers of {layer_bytes / 1e9:.2f} GB, "
          f"host budget {budget / 1e9:.1f} GB, batch {args.batch}")

    files = args.layers
    if args.pool_gb:
        files = max(1, min(args.layers, int(args.pool_gb * (1 << 30)) // layer_bytes))
        print(f"file pool: {files} files, {files * layer_bytes / 1e9:.1f} GB on disk")
    make_files(args.dir, files, layer_bytes)
    # Freshly written data may still be flushing inside the drives; measuring
    # during that window understates the read rate.
    time.sleep(args.settle)
    elems = layer_bytes // 2
    pool_n = budget // layer_bytes + 1
    pinned_pool = [torch.empty(elems, dtype=torch.float16, pin_memory=True) for _ in range(pool_n)]
    reader = Reader(args.dir, files, args.threads)
    print(f"O_DIRECT: {reader.direct}   NVMe read: "
          f"{bandwidth(reader, pinned_pool[0], range(min(8, args.layers))) / 1e9:.2f} GB/s")
    gpu_w = [torch.empty(elems, dtype=torch.float16, device="cuda") for _ in range(2)]
    x = torch.randn(args.batch, COLS, dtype=torch.float16, device="cuda")

    # Calibration for the simulator: host-to-GPU copy rate and per-layer compute.
    src = pinned_pool[0]
    for _ in range(2):
        gpu_w[0].copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(5):
        gpu_w[0].copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    h2d = 5 * layer_bytes / (time.perf_counter() - t)
    w = gpu_w[0].view(-1, COLS)
    for _ in range(3):
        torch.matmul(x, w.t())
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        torch.matmul(x, w.t())
    torch.cuda.synchronize()
    mm = (time.perf_counter() - t) / 10
    def calibrate():
        med = lambda xs: sorted(xs)[len(xs) // 2]
        nv = med([bandwidth(reader, pinned_pool[0], range(min(8, files))) for _ in range(3)])
        inflight = {k: med([bandwidth_inflight(reader, pinned_pool, files, k) for _ in range(3)])
                    for k in (1, 2, 4) if k <= len(pinned_pool)}
        return nv, inflight

    nvme, by_inflight = calibrate()
    print("NVMe by reads in flight: " + ", ".join(
        f"{k}: {v / 1e9:.2f} GB/s" for k, v in by_inflight.items()))
    meta = {"nvme_Bps_by_inflight": by_inflight,"gpu": torch.cuda.get_device_name(), "layer_bytes": layer_bytes,
            "layers": args.layers, "budget_bytes": budget, "batch": args.batch,
            "nvme_Bps": nvme, "h2d_Bps": h2d, "matmul_s": mm, "o_direct": reader.direct,
            "pool_files": files}
    print(f"calibration: NVMe {nvme / 1e9:.2f} GB/s, host-to-GPU {h2d / 1e9:.2f} GB/s, "
          f"matmul {mm * 1e3:.2f} ms per layer")

    if args.calibrate_only:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reserve_bench_meta.json")
        json.dump(meta, open(out, "w"), indent=1)
        print(f"wrote {out}")
        return
    rows = []
    print(f"{'reserve':>8}{'K':>4}{'slots':>6}{'resident':>9}{'nvme GB':>9}{'step s':>9}{'tok/s':>9}")
    for r in [float(v) for v in args.reserves.split(",")]:
        for k in [int(v) for v in args.depths.split(",")]:
            m = run(args, r, k, reader, pinned_pool, layer_bytes, gpu_w, x)
            rows.append(m)
            print(f"{r:>8.2f}{k:>4}{m['slots']:>6}{m['resident_layers']:>9}"
                  f"{m['nvme_GB_per_step']:>9.1f}{m['step_s']:>9.3f}{m['tok_per_s']:>9.2f}", flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reserve_bench.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    post_nvme, post_inflight = calibrate()
    meta["post_sweep"] = {"nvme_Bps": post_nvme, "nvme_Bps_by_inflight": post_inflight}
    print(f"after the sweep: NVMe {post_nvme / 1e9:.2f} GB/s, " + ", ".join(
        f"{k} in flight {v / 1e9:.2f}" for k, v in post_inflight.items()))
    json.dump(meta, open(out.replace(".csv", "_meta.json"), "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
