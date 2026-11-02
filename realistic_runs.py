"""
Realistic-host experiments, under the pass-through NAND model.

  host    the measured host bandwidth of the CMM-H test server: 161 GB/s for
          one socket of 8 DDR5-5600 channels (Soltaniyeh et al., HotStorage'25,
          Tables 2 and 4), with 256 or 512 GB of host DRAM, and the prototype's
          48 GB device cache.
  decode  decode length from 16 to 2048 tokens: decode throughput and the
          end-to-end ratio, prefill included.
  split   TierSched allowed to leave part of host DRAM unused, so weights also
          stream from the device in parallel with the host.

    NAND_PASS_THROUGH=1 python realistic_runs.py host
    NAND_PASS_THROUGH=1 python realistic_runs.py decode
    NAND_PASS_THROUGH=1 python realistic_runs.py split
"""
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import sys
import tempfile

import run_paper_tables as R

HOST_BW_MEASURED = 161e9
MODELS = {"72B": "Qwen2_5_72BCfg", "405B": "Llama3_1_405BCfg"}
BASELINES = [s for n, s in R.SIMS if n != "TierSched"]
NAMES = {s: n for n, s in R.SIMS}

# The search's device-size loop, extended to host sizes as well. Each anchor
# is asserted, so a drifted file cannot silently run the unmodified search.
SPLIT = [
    ("        for _devcap in _dev_caps:\n            _caps0[\"cxl\"] = _devcap",
     "        _host_caps = [host_dram_capacity_bytes * f for f in (1.0, 0.9, 0.75, 0.5)]\n"
     "        for _devcap in [(d, h) for d in _dev_caps for h in _host_caps]:\n"
     "            _caps0[\"cxl\"], _caps0[\"host\"] = _devcap"),
    ("            _caps0[\"cxl\"] = _devcap          # _kv_place inside _exact reads it",
     "            _caps0[\"cxl\"], _caps0[\"host\"] = _devcap"),
    ("        _caps0[\"cxl\"] = _best_devcap\n",
     "        _caps0[\"cxl\"], _caps0[\"host\"] = _best_devcap\n"
     "        print(f\"Host used: {_best_devcap[1] / GiB:.0f} GB\")\n"
     "        _best_devcap = _best_devcap[0]\n"),
]


def cell(sim, quant, h, c, gpu, model, decode=R.DECODE, host_bw=None, rewrites=()):
    """One simulator, one cell. Returns (decode_tps, prefill_tps, stdout)."""
    with tempfile.TemporaryDirectory() as td:
        for f in R.DEPS + [s for _, s in R.SIMS]:
            shutil.copy(os.path.join(R.REPO, f), td)
        R.patch(td, quant, h, c, gpu, model=MODELS[model])
        R._patch_lengths(td, 512, decode)
        if host_bw:
            p = os.path.join(td, "tiers.py")
            s = open(p).read()
            for a, b in (('HOST_DRAM = Tier("Host DRAM", 38.4e9',
                          f'HOST_DRAM = Tier("Host DRAM", {host_bw!r}'),
                         ("HOST_DRAM_CEILING      = 38.4e9",
                          f"HOST_DRAM_CEILING      = {host_bw!r}")):
                assert a in s, a
                s = s.replace(a, b, 1)
            open(p, "w").write(s)
        if rewrites:
            p = os.path.join(td, sim)
            s = open(p).read()
            for a, b in rewrites:
                assert a in s, a[:60]
                s = s.replace(a, b, 1)
            open(p, "w").write(s)
        r = subprocess.run([sys.executable, sim], cwd=td, capture_output=True,
                           text=True, timeout=1800)
        out = r.stdout + r.stderr
        d = re.search(r"Decode throughput:\s*([\d.]+)", out)
        p = re.search(r"Prefill throughput:\s*([\d.]+)", out)
        return (float(d.group(1)) if d else None,
                float(p.group(1)) if p else None, out)


def _job(j):
    return j, cell(*j[1:])[:2]


def run_all(jobs):
    with cf.ProcessPoolExecutor() as ex:
        return dict(ex.map(_job, jobs))


def end_to_end(dec, pre, decode):
    """Simulated seconds for one 512-token prompt and `decode` tokens of B sequences."""
    return 512.0 / pre + decode * R.BATCH / dec


def host():
    grid = [(g, q, h, m) for g in (0, R.GPU_GB) for m in MODELS for q in R.QUANTS
            for h in (256, 512)]
    jobs = [(s, s, q, h, 48, g, m, R.DECODE, HOST_BW_MEASURED) for g, q, h, m in grid
            for s in BASELINES + ["tiersched.py"]]
    res = run_all(jobs)
    print(f"Realistic host: {HOST_BW_MEASURED / 1e9:.0f} GB/s, 48 GB device cache, B={R.BATCH}, decode {R.DECODE}")
    print(f"{'plat':<5}{'model':<6}{'quant':<6}{'host':>6}{'best baseline':>24}{'TierSched':>10}{'ratio':>8}")
    for g, q, h, m in grid:
        base = max((res[(s, s, q, h, 48, g, m, R.DECODE, HOST_BW_MEASURED)][0] or 0, NAMES[s])
                   for s in BASELINES)
        sem = res[("tiersched.py",) * 2 + (q, h, 48, g, m, R.DECODE, HOST_BW_MEASURED)][0]
        print(f"{'GPU' if g else 'CPU':<5}{m:<6}{q:<6}{h:>5}G{base[0]:>13.3f} ({base[1]:<8}){sem:>10.3f}"
              f"{sem / base[0]:>7.2f}x")


def decode():
    lengths = [16, 128, 512, 2048]
    setups = [("paper", 16, 48, None, "72B"), ("realistic", 256, 48, HOST_BW_MEASURED, "405B")]
    grid = [(g, q, name, h, c, bw, m, n) for name, h, c, bw, m in setups
            for g in (0, R.GPU_GB) for q in R.QUANTS for n in lengths]
    jobs = [(s, s, q, h, c, g, m, n, bw) for g, q, name, h, c, bw, m, n in grid
            for s in BASELINES + ["tiersched.py"]]
    res = run_all(jobs)
    print("Decode length: decode ratio and end-to-end ratio against the best baseline on each metric")
    print(f"{'setup':<11}{'plat':<5}{'quant':<6}{'decode':>7}{'decode ratio':>14}{'end-to-end':>12}")
    for g, q, name, h, c, bw, m, n in grid:
        r = {s: res[(s, s, q, h, c, g, m, n, bw)] for s in BASELINES + ["tiersched.py"]}
        sd, sp = r["tiersched.py"]
        bd = max(r[s][0] or 0 for s in BASELINES)
        bt = min(end_to_end(*r[s], n) for s in BASELINES if r[s][0] and r[s][1])
        print(f"{name:<11}{'GPU' if g else 'CPU':<5}{q:<6}{n:>7}{sd / bd:>13.2f}x"
              f"{bt / end_to_end(sd, sp, n):>11.2f}x")


def split():
    setups = [("paper host", 16, None), ("realistic host", 256, HOST_BW_MEASURED)]
    grid = [(name, h, bw, g, q, m) for name, h, bw in setups for g in (0, R.GPU_GB)
            for q in R.QUANTS for m in MODELS]
    jobs = []
    for name, h, bw, g, q, m in grid:
        jobs.append(("plain", "tiersched.py", q, h, 48, g, m, R.DECODE, bw))
        jobs.append(("split", "tiersched.py", q, h, 48, g, m, R.DECODE, bw, tuple(SPLIT)))
    with cf.ProcessPoolExecutor() as ex:
        res = dict(zip(jobs, ex.map(_split_job, jobs)))
    print("Host/device split: TierSched allowed to leave host DRAM unused (100/90/75/50%)")
    print(f"{'setup':<16}{'plat':<5}{'model':<6}{'quant':<6}{'search':>10}{'with split':>12}{'gain':>8}  host used")
    for name, h, bw, g, q, m in grid:
        a = res[("plain", "tiersched.py", q, h, 48, g, m, R.DECODE, bw)]
        b = res[("split", "tiersched.py", q, h, 48, g, m, R.DECODE, bw, tuple(SPLIT))]
        print(f"{name:<16}{'GPU' if g else 'CPU':<5}{m:<6}{q:<6}{a[0]:>10.3f}{b[0]:>12.3f}"
              f"{(b[0] / a[0] - 1) * 100:>7.1f}%  {b[1]}")


def _split_job(j):
    rewrites = j[9] if len(j) > 9 else ()
    d, _, out = cell(*j[1:9], rewrites=rewrites)
    m = re.search(r"Host used: (\d+ GB)", out)
    return d, m.group(1) if m else "-"


if __name__ == "__main__":
    {"host": host, "decode": decode, "split": split}[sys.argv[1]]()
