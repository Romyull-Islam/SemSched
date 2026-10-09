from dataclasses import dataclass


# Weight precision for all simulators: "fp32", "fp16", "int8" or "int4".
QUANT = "fp32" # patched per run by run_paper_tables.py


# Multiplier on per-layer FLOPs (2 x params per token); 1.0 leaves them unscaled.
FLOP_PROXY_SCALE = 1.0


BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.5,  # packed
}[QUANT]


# Mistral 7B
@dataclass(frozen=True)
class Mistral7BCfg:
    num_blocks: int = 32
    vocab_size: int = 32000
    emb_dim: int = 4096
    mlp_hidden: int = 14336
    q_heads: int = 32
    kv_heads: int = 8  # grouped-query attention (GQA)


# Llama 2 13B
@dataclass(frozen=True)
class Llama13BCfg:
    num_blocks: int = 40
    vocab_size: int = 32000
    emb_dim: int = 5120
    mlp_hidden: int = 13824
    q_heads: int = 40
    kv_heads: int = 40  # multi-head attention


# Qwen3-style 20B configuration (~20B params)
@dataclass(frozen=True)
class Qwen3_20BCfg:
    num_blocks: int = 40
    vocab_size: int = 151936  # Qwen vocabulary
    emb_dim: int = 6144
    mlp_hidden: int = 22016
    q_heads: int = 40
    kv_heads: int = 8  # GQA



# Qwen2.5 72B, dense decoder: 80 layers, 8192 emb dim, 64 Q heads / 8 KV heads
# (GQA), ~152K vocab. The default model.
@dataclass(frozen=True)
class Qwen2_5_72BCfg:
    num_blocks: int = 80
    vocab_size: int = 152064
    emb_dim: int = 8192
    mlp_hidden: int = 28672         # SwiGLU intermediate dim
    q_heads: int = 64
    kv_heads: int = 8               # GQA


# Llama 3.1 405B, for a weight footprint far larger than every DRAM tier.
# 126 layers, 16384 emb dim, 128 Q heads / 8 KV heads (GQA), 53248 MLP hidden,
# 128256 vocab. FP16 footprint ~810 GB against 64 GB of DRAM: 92% NAND-resident.
@dataclass(frozen=True)
class Llama3_1_405BCfg:
    num_blocks: int = 126
    vocab_size: int = 128256
    emb_dim: int = 16384
    mlp_hidden: int = 53248
    q_heads: int = 128
    kv_heads: int = 8


# Model used by all simulators.
DEFAULT_MODEL_CFG = Qwen2_5_72BCfg # patched per run by run_paper_tables.py


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


def decomposed_build_layers(cfg=DEFAULT_MODEL_CFG(), sequence_length=512):
    base_layers = build_layers(cfg, sequence_length)
    decomposed = []
    d = cfg.emb_dim
    head_dim = d // cfg.q_heads
    kv_total = cfg.kv_heads * head_dim

    attn_params = (d * d) + (d * head_dim * cfg.q_heads) + (d * kv_total) + (d * d)  # Q, K, V, O
    mlp_params = (d * cfg.mlp_hidden) + (cfg.mlp_hidden * d) + (d * cfg.mlp_hidden)  # Gate, Up, Down

    # Only the attention sublayer carries KV cache
    kv_cache_bytes = 2 * cfg.kv_heads * head_dim * sequence_length * BYTES_PER_PARAM

    for L in base_layers:
        if L["kind"] == "DecoderBlock":
            # Attention sublayer
            attn_layer = dict(L)
            attn_layer["name"] = L["name"] + "_attn"
            attn_layer["kind"] = "Attention"
            attn_layer["params"] = attn_params
            attn_layer["bytes"] = int(attn_params * BYTES_PER_PARAM)
            attn_layer["flops"] = int(2 * attn_params * FLOP_PROXY_SCALE)
            attn_layer["kv_cache_bytes"] = int(kv_cache_bytes)
            attn_layer["head_dim"] = head_dim
            attn_layer["kv_heads"] = cfg.kv_heads

            # MLP sublayer
            mlp_layer = dict(L)
            mlp_layer["name"] = L["name"] + "_mlp"
            mlp_layer["kind"] = "MLP"
            mlp_layer["params"] = mlp_params
            mlp_layer["bytes"] = int(mlp_params * BYTES_PER_PARAM)
            mlp_layer["flops"] = int(2 * mlp_params * FLOP_PROXY_SCALE)
            mlp_layer["kv_cache_bytes"] = 0  # No KV cache for MLP

            decomposed.extend([attn_layer, mlp_layer])
        else:
            decomposed.append(L)

    return decomposed



# Layers FlexGen pins whole in the fastest tier with room
HOT_LAYERS_BY_NAME = ("lm_head", "final_norm")


