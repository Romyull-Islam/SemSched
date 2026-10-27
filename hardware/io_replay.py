"""
Replay one FlexGen decode step's disk I/O, without FlexGen and without a GPU,
to rank (CPU share, prefetch depth) plans before running any of them.

FlexGen's placement rule is replayed on the tensor sizes recorded by
flexgen_reserve.py (FG_CALIB), so the disk-resident tensors of every CPU share
are known exactly. The replay then reads them the way FlexGen does: layer by
layer, loading layer i+1 (memory-mapped, copied with torch into pinned buffers by 4 threads)
while layer i "computes". A layer's compute time is its share, by bytes, of the
decode step measured with every weight in host memory. With depth K it also
asks the kernel to read K layers ahead and drops layer i-2 afterwards, as
flexgen_reserve.py does. The replay runs in a cgroup whose page cache is limited
to the plan's room: the budget minus FlexGen's measured start-up footprint at
that share. Its steady-state step time is the plan's predicted step time.

    python io_replay.py setup  --calib C.json --dir D                # write tensor files once
    python io_replay.py replay --calib C.json --dir D --cpu 60 --k 8 \\
        --room-bytes 5.4e9 --step-s 1.02                             # inside a scope (io_replay.sh)
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import time

import numpy as np

CHUNK = 64 << 20
LOWEST_SHARE = 20          # setup writes the files of this share, a superset of higher shares


def disk_tensors(calib, cpu):
    """[(layer, tensor index, bytes)] that FlexGen puts on disk at CPU share `cpu`.

    flex_opt.init_weight_list: a tensor goes to the first device in (disk, cpu,
    gpu) whose cumulative percent exceeds the tensor's midpoint percent within
    its layer; the GPU share is 0 here."""
    out = []
    for layer, sizes in enumerate(calib["layer_tensor_bytes"]):
        cum = np.cumsum(sizes)
        for t, s in enumerate(sizes):
            mid = (cum[t] - s / 2) / cum[-1] * 100
            if mid < 100 - cpu:
                out.append((layer, t, s))
    return out


def path(d, layer, t):
    return os.path.join(d, f"l{layer:03d}_t{t:02d}.bin")


def setup(a):
    calib = json.load(open(a.calib))
    os.makedirs(a.dir, exist_ok=True)
    buf = np.ones(CHUNK, np.uint8)
    for layer, t, s in disk_tensors(calib, LOWEST_SHARE):
        p = path(a.dir, layer, t)
        if os.path.exists(p) and os.path.getsize(p) == s:
            continue
        with open(p, "wb") as f:
            for off in range(0, s, CHUNK):
                f.write(buf[:min(CHUNK, s - off)])
            f.flush()
            os.fsync(f.fileno())
        advise([p], os.POSIX_FADV_DONTNEED)


def advise(paths, advice):
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, advice)
        os.close(fd)


def own_scope():
    cg = open("/proc/self/cgroup").read().strip().split("::")[-1]
    return "/sys/fs/cgroup" + cg, os.path.basename(cg)


def replay(a):
    calib = json.load(open(a.calib))
    sizes = calib["layer_tensor_bytes"]
    total = sum(map(sum, sizes))
    gap = [a.step_s * sum(s) / total for s in sizes]
    files = [[] for _ in sizes]
    for layer, t, _ in disk_tensors(calib, a.cpu):
        files[layer].append(path(a.dir, layer, t))
    flat = [p for l in files for p in l]
    advise(flat, os.POSIX_FADV_DONTNEED)
    import torch
    # FlexGen's copy threads each relay through a pinned buffer and copy a whole
    # memory-mapped tensor into it with torch (pytorch_backend.copy_worker_func).
    biggest = max(os.path.getsize(p) for p in flat) if flat else 1
    bufs = [torch.empty(biggest, dtype=torch.uint8, pin_memory=True) for _ in range(4)]
    pool = cf.ThreadPoolExecutor(4)     # FlexGen's 4 copy threads
    ahead = cf.ThreadPoolExecutor(1)    # the load of layer i+1, overlapped with layer i
    side = cf.ThreadPoolExecutor(3)

    # Page cache limited to the plan's room, above what this process holds now.
    cg, unit = own_scope()
    base = int(open(cg + "/memory.current").read())
    subprocess.run(["systemctl", "--user", "set-property", "--runtime", unit,
                    f"MemoryMax={base + int(a.room_bytes)}"], check=True)

    def read_one(slot, p):
        src = torch.from_numpy(np.memmap(p, np.uint8, "r+"))
        bufs[slot][:src.numel()].copy_(src)

    def load(i):
        if files[i]:
            list(pool.map(read_one, [k % 4 for k in range(len(files[i]))], files[i]))

    stat = {}

    def sample():   # page-cache composition of this scope while it replays
        while not done:
            for l in open(cg + "/memory.stat"):
                k, v = l.split()
                if k in ("active_file", "inactive_file"):
                    stat.setdefault(k, []).append(int(v))
            c = int(open(cg + "/memory.current").read())
            stat.setdefault("current", []).append(c - base)
            time.sleep(0.5)

    done = False
    side.submit(sample)
    n = len(sizes)
    times = []
    for _ in range(a.steps):
        t0 = time.perf_counter()
        load(0)
        for i in range(n):
            if a.k:
                side.submit(advise, [p for d in range(1, a.k + 1) for p in files[(i + d) % n]],
                            os.POSIX_FADV_WILLNEED)
                side.submit(advise, files[(i - 2) % n], os.POSIX_FADV_DONTNEED)
            nxt = ahead.submit(load, (i + 1) % n) if i + 1 < n else None
            time.sleep(gap[i])
            if nxt:
                nxt.result()
        times.append(time.perf_counter() - t0)
    done = True
    steady = sorted(times[1:])[len(times[1:]) // 2]
    print(json.dumps({"cpu": a.cpu, "k": a.k, "room_bytes": a.room_bytes,
                      "disk_bytes": sum(os.path.getsize(p) for p in flat),
                      "step_s": steady, "steps": times,
                      "mean_bytes": {k: sum(v) / len(v) for k, v in stat.items()}}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["setup", "replay"])
    ap.add_argument("--calib", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cpu", type=float)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--room-bytes", type=float)
    ap.add_argument("--step-s", type=float, help="decode step with all weights in host memory")
    ap.add_argument("--steps", type=int, default=4)
    a = ap.parse_args()
    setup(a) if a.mode == "setup" else replay(a)
