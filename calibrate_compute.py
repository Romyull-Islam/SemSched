"""
calibrate_compute.py — validate SemSched's compute term against LLMCompass.

Reviewer 2.2 and 3.5 asked how the simulator is calibrated. Our compute model is
a closed-form rate expression, FLOPs / (freq x units x FLOP-per-cycle x util),
which is fast but unvalidated on its own. LLMCompass (ISCA 2024) is a published
LLM-inference hardware model validated against real GPUs to within 4.1% average
error for end-to-end inference. This script runs LLMCompass on the same decoder
block our simulator models and compares the two per-layer compute times.

We compare on LLMCompass's own validated A100 configuration rather than our H100,
so that any disagreement is attributable to the model rather than to a device
config LLMCompass never validated.

Usage (LLMCompass must be on the path):
    python calibrate_compute.py --llmcompass /path/to/LLMCompass
"""
import argparse
import os
import sys

# SemSched's analytical compute model, verbatim in form.
def semsched_compute_time_s(flops, peak_flops, util):
    return flops / (peak_flops * util) if flops > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llmcompass", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 128])
    args = ap.parse_args()

    sys.path.insert(0, args.llmcompass)
    os.chdir(args.llmcompass)
    from software_model.transformer import TransformerBlockAutoRegressionTP
    from software_model.utils import data_type_dict, Tensor
    from hardware_model.system import system_dict

    # Qwen2.5 72B decoder block geometry (matches model_cfg.py).
    D_MODEL, N_HEADS, SEQ = 8192, 64, 512
    system = system_dict["A100_4_fp16"]
    n_dev = 4

    # A100 80GB SXM dense FP16 tensor throughput, x4 devices.
    PEAK = 312e12 * n_dev
    UTIL = 0.70

    # Per decoder block: attention projections (4 x d^2) + MLP (3 x d x 4d).
    attn_params = 4 * D_MODEL * D_MODEL
    mlp_params = 3 * D_MODEL * (4 * D_MODEL)
    block_params = attn_params + mlp_params

    print(f"Qwen-72B-shaped decoder block: d_model={D_MODEL}, heads={N_HEADS}")
    print(f"Reference device: LLMCompass A100_4_fp16 "
          f"({PEAK/1e12:.0f} TFLOPS peak, util={UTIL})\n")
    print(f"{'batch':>6}{'LLMCompass (ms)':>18}{'SemSched (ms)':>16}"
          f"{'ratio':>9}{'error':>9}")
    print("-" * 58)

    errs = []
    for bs in args.batches:
        model = TransformerBlockAutoRegressionTP(
            d_model=D_MODEL, n_heads=N_HEADS, device_count=n_dev,
            data_type=data_type_dict["fp16"],
        )
        _ = model(Tensor([bs, 1, D_MODEL], data_type_dict["fp16"]), SEQ)
        t_ref = model.roofline_model(system)

        flops = 2 * block_params * bs
        t_ours = semsched_compute_time_s(flops, PEAK, UTIL)

        ratio = t_ours / t_ref if t_ref > 0 else float("nan")
        err = abs(t_ours - t_ref) / t_ref * 100 if t_ref > 0 else float("nan")
        errs.append(err)
        print(f"{bs:>6}{t_ref*1e3:>18.4f}{t_ours*1e3:>16.4f}"
              f"{ratio:>9.2f}{err:>8.1f}%")

    print(f"\nMean absolute deviation: {sum(errs)/len(errs):.1f}%")
    print("\nNote: LLMCompass's roofline includes attention-score compute and")
    print("KV reads, which our compute term excludes by construction (we account")
    print("KV traffic separately in the memory model). The comparison therefore")
    print("bounds the GEMM term, which is what our expression claims to cover.")


if __name__ == "__main__":
    main()
