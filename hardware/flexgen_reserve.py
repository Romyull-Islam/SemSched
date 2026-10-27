"""
FlexGen with a prefetch reserve, for the real-engine experiment.

FlexGen (FMInference/FlexLLMGen) is run unmodified, inside a memory-limited
cgroup that plays the fast tier. Its disk-resident weights are read through the
OS page cache, which counts against the cgroup limit. When FlexGen's CPU share
of the weights fills the limit, nothing can be read ahead. Lowering the CPU
share leaves part of the limit free; this runner then adds a prefetcher that,
while layer j runs, asks the kernel to read the disk files of layers j+1..j+K
into that free memory (POSIX_FADV_WILLNEED) and drops each layer's pages once
it has been used (POSIX_FADV_DONTNEED). FlexGen's own reads then find the data
in memory. Nothing else changes.

    systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 \\
        python flexgen_reserve.py --prefetch-depth 4 <usual flex_opt arguments>

With --prefetch-depth 0 this is plain FlexGen.

FG_CALIB=out.json adds a calibration at start-up, before any token is
generated: it records every weight tensor's size in FlexGen's own layer order,
so a planner can replay FlexGen's placement rule for any CPU share, and it
times reading the disk-resident weights two ways: the way FlexGen reads them
(memory-mapped, copied by 4 threads into pinned buffers) and through kernel
read-ahead issued K layers ahead, as this runner does.
"""
import argparse
import concurrent.futures as cf
import os
import json
import sys
import time

import numpy as np
import torch
from flexllmgen import flex_opt
from flexllmgen.pytorch_backend import DeviceType, TorchTensor
from flexllmgen.utils import ValueHolder

DEPTH = 0
POOL = cf.ThreadPoolExecutor(4)


def disk_paths(holder):
    """Files of the disk-resident tensors under a weight holder."""
    out, todo = [], [holder]
    while todo:
        x = todo.pop()
        if isinstance(x, ValueHolder):
            v = x.val
            todo.extend(v if isinstance(v, (list, tuple)) else [v])
        elif isinstance(x, (list, tuple)):
            todo.extend(x)
        elif isinstance(x, TorchTensor) and x.device is not None \
                and x.device.device_type == DeviceType.DISK:
            out.append(x.data)
    return out


def advise(paths, advice):
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, advice)
            os.close(fd)
        except OSError:
            pass


_orig_load_weight = flex_opt.OptLM.load_weight


def load_weight(self, i, j, k, overlap=True):
    if DEPTH and k == 0:
        n = self.num_layers
        paths = getattr(self, "_disk_paths", None)
        if paths is None:
            paths = self._disk_paths = [disk_paths(self.weight_home[x]) for x in range(n)]
        cur = j % n
        ahead = [p for d in range(1, DEPTH + 1) for p in paths[(cur + d) % n]]
        POOL.submit(advise, ahead, os.POSIX_FADV_WILLNEED)
        POOL.submit(advise, paths[(cur - 2) % n], os.POSIX_FADV_DONTNEED)
    return _orig_load_weight(self, i, j, k, overlap)


flex_opt.OptLM.load_weight = load_weight

SPECS = []          # per init_weight_list call: tensor sizes in bytes, in order
_orig_init_weight_list = flex_opt.init_weight_list


def init_weight_list(weight_specs, policy, env):
    SPECS.append([int(np.prod(s[0])) * np.dtype(s[1]).itemsize for s in weight_specs])
    return _orig_init_weight_list(weight_specs, policy, env)


def read_rate(paths, window):
    """Bytes/s to copy `paths` (grouped by layer) into pinned memory.

    window 0 reads the way FlexGen does: page faults on a memory map. window K
    first asks the kernel to read K layers ahead, as load_weight above does."""
    flat = [p for layer in paths for p in layer]
    advise(flat, os.POSIX_FADV_DONTNEED)
    biggest = max(os.path.getsize(p) for p in flat)
    bufs = [torch.empty(biggest, dtype=torch.uint8, pin_memory=True) for _ in range(4)]

    def copy(i, p):
        src = torch.from_numpy(np.lib.format.open_memmap(p, mode="r").reshape(-1).view(np.uint8))
        bufs[i % 4][:src.numel()].copy_(src)

    pool = cf.ThreadPoolExecutor(4)
    t0 = time.perf_counter()
    for j, layer in enumerate(paths):
        if window:
            ahead = [p for layer2 in paths[j + 1:j + 1 + window] for p in layer2]
            advise(ahead if j else [p for l2 in paths[:window + 1] for p in l2],
                   os.POSIX_FADV_WILLNEED)
        list(pool.map(copy, range(len(layer)), layer))
        advise(paths[j - 2] if j >= 2 else [], os.POSIX_FADV_DONTNEED)
    dt = time.perf_counter() - t0
    advise(flat, os.POSIX_FADV_DONTNEED)
    return sum(os.path.getsize(p) for p in flat) / dt


_orig_init = flex_opt.OptLM.__init__


def init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    out = os.environ.get("FG_CALIB")
    if not out:
        return
    paths = [[p for p in disk_paths(self.weight_home[x])] for x in range(self.num_layers)]
    paths = [l for l in paths if l]
    rates = {w: read_rate(paths, w) for w in (0, 1, 2, 4, 8, 16)}
    json.dump({"layer_tensor_bytes": SPECS, "num_layers": self.num_layers,
               "disk_bytes": sum(os.path.getsize(p) for l in paths for p in l),
               "read_Bps_by_window": rates}, open(out, "w"), indent=1)
    print(f"calibration written to {out}", flush=True)


flex_opt.init_weight_list = init_weight_list
flex_opt.OptLM.__init__ = init

if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--prefetch-depth", type=int, default=0)
    known, rest = pre.parse_known_args()
    DEPTH = known.prefetch_depth
    sys.argv = [sys.argv[0]] + rest
    parser = argparse.ArgumentParser()
    flex_opt.add_parser_arguments(parser)
    args = parser.parse_args()
    assert len(args.percent) == 6
    print(f"prefetch depth: {DEPTH}")
    flex_opt.run_flexllmgen(args)
