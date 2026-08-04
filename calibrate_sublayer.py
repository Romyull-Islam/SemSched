"""
calibrate_sublayer.py — derive SemSched's compute term from LLMCompass.

The previous calibration (calibrate_compute.py) only *bounded* our compute term.
This one replaces the assumed utilization with a measured one.

Our simulator's compute model is FLOPs / (peak x util), with util assumed. A flat
utilization is wrong in a specific and knowable way: at decode the GEMMs have
M = batch, so at small batch a GEMV leaves most of a systolic array idle and
achieved utilization collapses. LLMCompass models the array explicitly, so we run
our exact per-sub-layer GEMM shapes through it and read the achieved utilization
off the result, per batch size.

We take LLMCompass's roofline (the max of its compute and memory arms) as the
sub-layer compute time. That is conservative for us: it can only make the modeled
engine look slower, which makes the workload look *less* memory-bound and shrinks
our reported advantage.

    python calibrate_sublayer.py --llmcompass /path/to/LLMCompass
"""
import argparse
import contextlib
import io
import os
import sys

# Qwen2.5 72B geometry (matches model_cfg.py)
D, N_HEADS, KV_HEADS, MLP_H = 8192, 64, 8, 28672
HEAD_DIM = D // N_HEADS
KV_TOTAL = KV_HEADS * HEAD_DIM


def gemms(kind, b):
    """(M, K, N) for each GEMM in one sub-layer at batch b, one decode step."""
    if kind == "attn":
        return [(b, D, D),          # Q
                (b, D, KV_TOTAL),   # K
                (b, D, KV_TOTAL),   # V
                (b, D, D)]          # O
    return [(b, D, MLP_H),          # gate
            (b, D, MLP_H),          # up
            (b, MLP_H, D)]          # down


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llmcompass", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 128])
    args = ap.parse_args()

    sys.path.insert(0, args.llmcompass)
    os.chdir(args.llmcompass)
    from software_model.matmul import Matmul
    from software_model.utils import data_type_dict, Tensor
    from hardware_model.device import device_dict

    device = device_dict["A100_80GB_fp16"]
    PEAK = 312e12                  # A100 80GB SXM dense FP16, single device

    def lc_time(M, K, N):
        op = Matmul(data_type_dict["fp16"])
        _ = op(Tensor([M, K], data_type_dict["fp16"]),
               Tensor([K, N], data_type_dict["fp16"]))
        with contextlib.redirect_stdout(io.StringIO()):
            return op.roofline_model(device)

    print(f"Per-sub-layer GEMM calibration on A100_80GB_fp16 "
          f"({PEAK/1e12:.0f} TFLOPS peak)\n")
    print(f"{'batch':>6}{'sublayer':>10}{'FLOPs':>12}{'LLMCompass':>13}"
          f"{'achieved':>12}{'util':>8}")
    print("-" * 61)

    util_by_batch = {}
    for b in args.batches:
        totals = {}
        for kind in ("attn", "mlp"):
            t = sum(lc_time(M, K, N) for M, K, N in gemms(kind, b))
            fl = sum(2 * M * K * N for M, K, N in gemms(kind, b))
            totals[kind] = (fl, t)
            print(f"{b:>6}{kind:>10}{fl/1e9:>10.2f}G{t*1e3:>11.4f}ms"
                  f"{fl/t/1e12:>10.1f}TF{fl/t/PEAK:>8.3f}")
        fl_tot = sum(v[0] for v in totals.values())
        t_tot = sum(v[1] for v in totals.values())
        util_by_batch[b] = fl_tot / t_tot / PEAK
        print(f"{'':>6}{'block':>10}{fl_tot/1e9:>10.2f}G{t_tot*1e3:>11.4f}ms"
              f"{fl_tot/t_tot/1e12:>10.1f}TF{util_by_batch[b]:>8.3f}")
        print()

    print("Achieved utilization vs batch (this is what a flat assumption misses):")
    for b, u in util_by_batch.items():
        print(f"  B={b:<5} util={u:.4f}   ({u/util_by_batch[max(util_by_batch)]*100:5.1f}% "
              f"of the B={max(util_by_batch)} value)")
    print()
    print("Recommended calibrated setting for sim_cfg.parallel_efficiency at the")
    print(f"canonical operating point B=128: {util_by_batch.get(128, float('nan')):.3f}")


if __name__ == "__main__":
    main()
