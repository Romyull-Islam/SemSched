# model_cfg.py
from dataclasses import dataclass

# ---- QUANT PRESET (change this one line to switch all sims) ----
QUANT = "fp32"  # options: "fp32", "fp16", "int8", "int4"
# ----------------------------------------------------------------

BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.5,   # packed
}[QUANT]

@dataclass(frozen=True)
class LlamaLikeCfg:
    num_blocks: int = 32
    vocab_size: int = 32000
    emb_dim: int = 4096
    mlp_hidden: int = 14336
    q_heads: int = 32
    kv_heads: int = 8

def build_layers(cfg = LlamaLikeCfg()):
    """Return ordered list of layer dicts with param counts/bytes and a simple FLOP proxy."""
    d = cfg.emb_dim
    head_dim = d // cfg.q_heads
    kv_total = cfg.kv_heads * head_dim

    attn_params = (2 * d * d) + (2 * d * kv_total)
    mlp_params  = (2 * d * cfg.mlp_hidden) + (cfg.mlp_hidden * d)
    block_params = attn_params + mlp_params

    embed_params = cfg.vocab_size * d
    lmhead_params = cfg.vocab_size * d
    finalnorm_params = d

    layers = []
    layers.append({"name":"embed_tokens","kind":"Embedding",
                   "params":embed_params, "bytes":int(embed_params * BYTES_PER_PARAM), "flops":0})
    for i in range(cfg.num_blocks):
        p = block_params
        layers.append({"name":f"decoder_{i}","kind":"DecoderBlock",
                       "params":p, "bytes":int(p * BYTES_PER_PARAM), "flops":2*p})
    layers.append({"name":"final_norm","kind":"RMSNorm",
                   "params":finalnorm_params, "bytes":int(finalnorm_params * BYTES_PER_PARAM), "flops":0})
    layers.append({"name":"lm_head","kind":"LMHead",
                   "params":lmhead_params, "bytes":int(lmhead_params * BYTES_PER_PARAM), "flops":0})
    return layers

# Hot layers to pin in DRAM first (same frequency, but lm_head has huge bytes)
HOT_LAYERS_BY_NAME = ("lm_head", "final_norm")
