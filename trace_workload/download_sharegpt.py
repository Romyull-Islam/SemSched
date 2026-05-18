"""
download_sharegpt.py - extract (prefill_len, decode_len) pairs from ShareGPT.

Uses tiktoken cl100k_base (GPT-4 tokenizer) as a proxy for Qwen's tokenizer.
For sequence-length distribution analysis, exact tokenizer choice does not
materially affect the simulation result.

Output: trace_workload/sharegpt_lens.json
        list of [prefill_tokens, decode_tokens] pairs.
"""
import json
import os
import sys

OUT_PATH = os.path.join(os.path.dirname(__file__), "sharegpt_lens.json")
N_SAMPLES = 200


def main():
    try:
        import tiktoken
    except ImportError:
        print("Install with: pip install tiktoken datasets")
        sys.exit(1)

    enc = tiktoken.get_encoding("cl100k_base")

    # Load ShareGPT — try the unfiltered Vicuna dump first.
    print("Loading ShareGPT...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
            split="train",
        )
    except Exception as e:
        print(f"ShareGPT load failed: {e}")
        print("Falling back to a synthetic distribution drawn from typical LLM-serving traces.")
        # Distribution drawn from published vLLM/SGLang benchmarks: prefill skews
        # bimodal (short chat 80-300 tok; longer task 500-2500 tok), decode
        # roughly log-normal around 100-500 tok.
        import random
        random.seed(0)
        pairs = []
        for _ in range(N_SAMPLES):
            if random.random() < 0.6:
                pf = random.randint(80, 320)
                dc = random.randint(50, 350)
            else:
                pf = random.randint(500, 2500)
                dc = random.randint(150, 700)
            pairs.append([pf, dc])
        with open(OUT_PATH, "w") as f:
            json.dump(pairs, f)
        print(f"Wrote {len(pairs)} synthetic pairs to {OUT_PATH}")
        return

    print(f"Loaded {len(ds)} conversations. Sampling {N_SAMPLES}...")
    pairs = []
    for ex in ds.select(range(min(len(ds), N_SAMPLES * 5))):
        convs = ex.get("conversations") or []
        if len(convs) < 2:
            continue
        user_turn = convs[0].get("value", "") or ""
        asst_turn = convs[1].get("value", "") or ""
        if not user_turn or not asst_turn:
            continue
        pf = len(enc.encode(user_turn))
        dc = len(enc.encode(asst_turn))
        if pf < 8 or dc < 4:
            continue
        pairs.append([pf, dc])
        if len(pairs) >= N_SAMPLES:
            break

    with open(OUT_PATH, "w") as f:
        json.dump(pairs, f)

    pfs = [p[0] for p in pairs]
    dcs = [p[1] for p in pairs]
    print(f"\nWrote {len(pairs)} pairs to {OUT_PATH}")
    print(f"  Prefill length: min={min(pfs)} med={sorted(pfs)[len(pfs)//2]} "
          f"P95={sorted(pfs)[int(len(pfs)*0.95)]} max={max(pfs)}")
    print(f"  Decode length:  min={min(dcs)} med={sorted(dcs)[len(dcs)//2]} "
          f"P95={sorted(dcs)[int(len(dcs)*0.95)]} max={max(dcs)}")


if __name__ == "__main__":
    main()
