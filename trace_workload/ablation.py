"""
ablation.py — disable each SemSched mechanism in turn and measure decode TPS.

Runs the headline config (Qwen 72B, FP16, 16H+64C, B=128, prefill=512, decode=16)
for the full SemSched and four ablation variants. Each ablation is a string-based
patch applied to the simulator source in a tempdir (originals untouched).

Output: trace_workload/ablation_results.csv
"""
import os
import sys
import re
import csv
import shutil
import tempfile
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_runner import (
    SIM_FILES, MEM_MAP, DEPS,
    patch_sim_cfg, patch_model_cfg, patch_simulator,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------- Ablation patches ----------
def ablate_no_warmup(src):
    """A1: skip prefill warmup loop — layers stay on NAND."""
    old = (
        '        if place[i] == PL_CXL_DEV_NAND:\n'
        '            dur   = transfer_time_s(L["bytes"], CXL_SSD_NAND)\n'
        '            start = max(lat, nand_link_free_at)\n'
        '            nand_link_free_at = start + (dur / IO_THREAD_POOL_SIZE)\n'
        '            cache.add(i, L["bytes"])\n'
        '            cache.pin_for_session(i)\n'
        '            stats["bytes_prefetched"] += L["bytes"]\n'
    )
    new = (
        '        if place[i] == PL_CXL_DEV_NAND:\n'
        '            pass  # ABLATION A1: prefill warmup disabled\n'
    )
    assert old in src, "A1 patch failed to match"
    return src.replace(old, new)


def ablate_no_semantic_placement(src):
    """A2: disable both attention-to-Host-DRAM AND high-sparsity MLP priorities.
    Layers fall through to size-based first-fit (CXL DRAM, then NAND)."""
    # Disable Priority 1 (attention pinning)
    old1 = (
        '    # 1. Pinned Attention → Host DRAM\n'
        '    for i, L in enumerate(layers):\n'
        '        if ltypes[i] == LayerType.ATTENTION and place[i] is None:\n'
    )
    new1 = (
        '    # 1. ABLATION A2a: attention pinning disabled\n'
        '    for i, L in enumerate(layers):\n'
        '        if False and ltypes[i] == LayerType.ATTENTION and place[i] is None:\n'
    )
    assert old1 in src, "A2a patch failed to match"
    src = src.replace(old1, new1)
    # Disable Priority 3 (high-sparsity MLP pinning)
    old2 = '        if sp > 0.60 and L["bytes"] <= h_free:'
    new2 = '        if False and sp > 0.60 and L["bytes"] <= h_free:  # ABLATION A2b'
    assert old2 in src, "A2b patch failed to match"
    src = src.replace(old2, new2)
    return src


def ablate_no_two_queue(src):
    """A3: serialize KV writes — no two-queue full-duplex link."""
    old = (
        '                        exposed = link.schedule_write_background(\n'
        '                            lat, kv_write_scaled, deadline=lat + ltime_base)\n'
    )
    new = (
        '                        exposed = transfer_time_s(kv_write_scaled, CXL_DRAM)  # ABLATION A3\n'
    )
    assert old in src, "A3 patch failed to match"
    return src.replace(old, new)


def ablate_no_sparsity_eviction(src):
    """A4: uniform eviction scores (LRU-like)."""
    old = '                        p_score = 2.0 if ltypes[fi] == LayerType.ATTENTION else sparsity[fi]'
    new = '                        p_score = 0.5  # ABLATION A4: uniform scoring'
    assert old in src, "A4 patch failed to match"
    return src.replace(old, new)


ABLATIONS = [
    ("Full SemSched", None),
    ("- Parallel Warmup", ablate_no_warmup),
    ("- Semantic Placement", ablate_no_semantic_placement),
    ("- Two-Queue Link", ablate_no_two_queue),
    ("- Sparsity Eviction", ablate_no_sparsity_eviction),
]


def run_variant(name, patch_fn,
                prefill=512, decode=16, batch=128,
                quant="fp16", mem="16H+64C", timeout_s=300):
    """Run SemSched with one ablation patch applied and capture Decode_TPS."""
    sim_file = SIM_FILES["SemSched"]
    host_gb, cxl_gb = MEM_MAP[mem]

    with tempfile.TemporaryDirectory(prefix="cxl_ablation_") as tmp:
        for dep in DEPS:
            shutil.copy(os.path.join(REPO, dep), tmp)

        # Patch sim_cfg.py
        cfg_path = os.path.join(tmp, "sim_cfg.py")
        with open(cfg_path) as f:
            cfg_src = f.read()
        with open(cfg_path, "w") as f:
            f.write(patch_sim_cfg(cfg_src, decode=decode, batch=batch,
                                  host_gb=host_gb, cxl_gb=cxl_gb))

        # Patch model_cfg.py
        mcfg_path = os.path.join(tmp, "model_cfg.py")
        with open(mcfg_path) as f:
            mcfg_src = f.read()
        with open(mcfg_path, "w") as f:
            f.write(patch_model_cfg(mcfg_src, quant=quant,
                                    model_cls="Qwen2_5_72BCfg"))

        # Patch simulator: PREFILL_TOKENS, then apply ablation
        with open(os.path.join(REPO, sim_file)) as f:
            sim_src = f.read()
        sim_src = patch_simulator(sim_src, prefill=prefill)
        if patch_fn is not None:
            sim_src = patch_fn(sim_src)
        sim_tgt = os.path.join(tmp, sim_file)
        with open(sim_tgt, "w") as f:
            f.write(sim_src)

        try:
            result = subprocess.run(
                [sys.executable, sim_file],
                cwd=tmp, env=os.environ.copy(),
                capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {"Decode_TPS": None, "err": "timeout"}
        if result.returncode != 0:
            return {"Decode_TPS": None, "err": result.stderr[-500:]}

        tps = None
        wstall = None
        for line in result.stdout.splitlines():
            m = re.match(r"Decode throughput:\s*([\d.]+)", line)
            if m:
                tps = float(m.group(1))
            m = re.match(r"Write_Stall_Time_s:\s*([\d.]+)", line)
            if m:
                wstall = float(m.group(1))
        return {"Decode_TPS": tps, "Write_Stall_s": wstall}


CONFIGS = [
    # (quant, mem, llm_ref_tps, batch)
    ("fp16", "16H+64C", 11.41, 128),   # headline config
    ("int8", "16H+32C", 26.25, 128),   # tight-memory config
]


def run_config(quant, mem, llm_ref, batch):
    print(f"\nAblation @ Qwen 72B {quant.upper()} {mem} B={batch} prefill=512 decode=16")
    print("-" * 72)
    rows = []
    for name, patch_fn in ABLATIONS:
        print(f"  {name:<24s} ...", end="", flush=True)
        res = run_variant(name, patch_fn, quant=quant, mem=mem, batch=batch)
        tps = res.get("Decode_TPS")
        wst = res.get("Write_Stall_s")
        if tps is None:
            print(f" FAIL ({res.get('err', '')[:80]})")
        else:
            print(f" TPS={tps:8.3f}  write_stall(ms)={(wst or 0)*1000:6.3f}")
        rows.append({"config": f"{quant}_{mem}", "Variant": name,
                     "Decode_TPS": tps, "Write_Stall_s": wst})

    full_tps = next((r["Decode_TPS"] for r in rows if r["Variant"] == "Full SemSched"), None)
    print(f"\nSummary @ {quant.upper()} {mem}:")
    print(f"{'Variant':<24s} | {'TPS':>8s} | {'vs LLMFlash':>11s} | {'vs Full':>8s} | {'Loss':>6s}")
    print("-" * 72)
    print(f"{'LLMFlash (Ref)':<24s} | {llm_ref:>8.2f} | {1.00:>10.2f}x | {'-':>8s} | {'-':>6s}")
    for r in rows:
        tps = r["Decode_TPS"]
        if tps is None or full_tps is None:
            continue
        ratio_llm = tps / llm_ref
        ratio_full = tps / full_tps
        loss = (1 - ratio_full) * 100
        print(f"{r['Variant']:<24s} | {tps:>8.2f} | {ratio_llm:>10.2f}x | {ratio_full:>7.2f}x | {loss:>5.1f}%")
    return rows


def main():
    all_rows = []
    for quant, mem, llm_ref, batch in CONFIGS:
        all_rows.extend(run_config(quant, mem, llm_ref, batch))

    out = os.path.join(REPO, "trace_workload/ablation_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
