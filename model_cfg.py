from dataclasses import dataclass

# ---- QUANT PRESET (change this one line to switch quantization for all sims) ----
QUANT = "int8"  # options: "fp32", "fp16", "int8", "int4"
# ----------------------------------------------------------------

# --- NEW: Scaling factor to create a more realistic, compute-bound scenario ---
FLOP_PROXY_SCALE = 1.0
# ----------------------------------------------------------------

BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.5,  # packed
}[QUANT]

# Original 7B model configuration
@dataclass(frozen=True)
class Llama7BCfg:
    num_blocks: int = 32
    vocab_size: int = 32000
    emb_dim: int = 4096
    mlp_hidden: int = 14336
    q_heads: int = 32
    kv_heads: int = 8

# 13B model configuration, based on Llama 2 13B
@dataclass(frozen=True)
class Llama13BCfg:
    num_blocks: int = 40
    vocab_size: int = 32000
    emb_dim: int = 5120
    mlp_hidden: int = 13824
    q_heads: int = 40
    kv_heads: int = 40  # Multi-Head Attention

# Default model configuration (change to switch models for all sims)
DEFAULT_MODEL_CFG = Llama13BCfg  # Set to Llama7BCfg for 7B model
# ----------------------------------------------------------------

def build_layers(cfg=DEFAULT_MODEL_CFG(), sequence_length=512):
    """Return ordered list of layer dicts with param counts/bytes, FLOPs, and KV cache bytes."""
    d = cfg.emb_dim
    head_dim = d // cfg.q_heads
    kv_total = cfg.kv_heads * head_dim

    attn_params = (d * d) + (d * head_dim * cfg.q_heads) + (d * kv_total) + (d * d)  # Q, K, V, O
    mlp_params = (d * cfg.mlp_hidden) + (cfg.mlp_hidden * d) + (d * cfg.mlp_hidden)  # Gate, Up, Down
    block_params = attn_params + mlp_params

    embed_params = cfg.vocab_size * d
    lmhead_params = cfg.vocab_size * d
    finalnorm_params = d

    # KV cache: 2 (key+value) * kv_heads * head_dim * sequence_length * bytes_per_param
    kv_cache_bytes = 2 * cfg.kv_heads * head_dim * sequence_length * BYTES_PER_PARAM

    layers = []
    layers.append({"name": "embed_tokens", "kind": "Embedding",
                   "params": embed_params, "bytes": int(embed_params * BYTES_PER_PARAM), "flops": 0, "kv_cache_bytes": 0})
    for i in range(cfg.num_blocks):
        p = block_params
        scaled_flops = int(2 * p * FLOP_PROXY_SCALE)
        layers.append({"name": f"decoder_{i}", "kind": "DecoderBlock",
                      "params": p, "bytes": int(p * BYTES_PER_PARAM), "flops": scaled_flops,
                      "kv_cache_bytes": int(kv_cache_bytes), "head_dim": head_dim, "kv_heads": cfg.kv_heads})
    layers.append({"name": "final_norm", "kind": "RMSNorm",
                  "params": finalnorm_params, "bytes": int(finalnorm_params * BYTES_PER_PARAM), "flops": 0, "kv_cache_bytes": 0})
    layers.append({"name": "lm_head", "kind": "LMHead",
                   "params": lmhead_params, "bytes": int(lmhead_params * BYTES_PER_PARAM), "flops": 0, "kv_cache_bytes": 0})
    return layers

# Hot layers to pin in Host DRAM first (modify for placement experiments)
HOT_LAYERS_BY_NAME = ("lm_head", "final_norm")