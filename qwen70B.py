# examples/inspect_qwen2p5_72b.py
from model_cfg import Qwen2_5_72BCfg, build_layers, decomposed_build_layers, BYTES_PER_PARAM, QUANT

def bytes_gib(n): return n / (1024**3)

if __name__ == "__main__":
    cfg = Qwen2_5_72BCfg()  # 80 blocks, d=8192, MLP=28672, 64 Q-heads, 8 KV-heads, ~152K vocab
    seq_len = 4096          # pick a realistic context for sizing

    fused = build_layers(cfg, sequence_length=seq_len)
    decomp = decomposed_build_layers(cfg, sequence_length=seq_len)

    total_params = sum(L["params"] for L in fused)
    total_bytes  = sum(L["bytes"]  for L in fused)

    # Per-token KV update (bytes) for attention layers:
    head_dim = cfg.emb_dim // cfg.q_heads
    kv_per_token_bytes = int(2 * cfg.kv_heads * head_dim * BYTES_PER_PARAM)

    # Per-sequence KV (bytes) per DecoderBlock (should match layer["kv_cache_bytes"] in fused)
    kv_per_block_seq_bytes = int(2 * cfg.kv_heads * head_dim * seq_len * BYTES_PER_PARAM)

    print("=== Qwen2.5-72B (decoder-only) ===")
    print(f"Quant: {QUANT}, BYTES_PER_PARAM={BYTES_PER_PARAM}")
    print(f"Blocks: {cfg.num_blocks}, d={cfg.emb_dim}, MLP={cfg.mlp_hidden}, Q-heads={cfg.q_heads}, KV-heads={cfg.kv_heads}, Vocab={cfg.vocab_size}")
    print(f"Total params (approx): {total_params:,}")
    print(f"Total model bytes: {total_bytes:,}  ({bytes_gib(total_bytes):.3f} GiB)")
    print(f"Per-token KV update (bytes): {kv_per_token_bytes:,}")
    print(f"Per-block KV at seq_len={seq_len}: {kv_per_block_seq_bytes:,}  ({bytes_gib(kv_per_block_seq_bytes):.6f} GiB)")
    print(f"Layers (fused): {len(fused)}  |  Layers (decomposed): {len(decomp)}")

    # Show a few rows as a sanity check (avoid dumping thousands of lines)
    print("\nSample fused layers:")
    for L in (fused[:2] + fused[-2:]):
        print({k: L[k] for k in ("name","kind","bytes","flops","kv_cache_bytes")})

    print("\nSample decomposed layers:")
    for L in (decomp[1:5]):  # first decoder block split into attn/mlp
        print({k: L[k] for k in ("name","kind","bytes","flops","kv_cache_bytes")})
