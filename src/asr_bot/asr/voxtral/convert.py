"""Convert an HF-format Voxtral Mini 4B Realtime checkpoint into one flat, fused file in a chosen dtype.

Changes vs the source layout: q/k/v and gate/up are fused, the encoder k projection gets a zero
bias, tensors are cast to ``dtype``, and the model dimensions are stored in the file metadata.
"""
from __future__ import annotations

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .net import Config

CONVERTER_VERSION = "1"
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def config_from_hf(src_dir: str, ada_dim: int) -> Config:
    with open(os.path.join(src_dir, "config.json")) as f:
        c = json.load(f)
    a, t = c["audio_config"], c["text_config"]
    return Config(
        enc_layers=a["num_hidden_layers"], enc_dim=a["hidden_size"], enc_heads=a["num_attention_heads"],
        enc_head_dim=a["head_dim"], enc_ffn=a["intermediate_size"], enc_window=a["sliding_window"],
        enc_theta=a["rope_parameters"]["rope_theta"],
        dec_layers=t["num_hidden_layers"], dec_dim=t["hidden_size"], dec_heads=t["num_attention_heads"],
        dec_kv_heads=t["num_key_value_heads"], dec_head_dim=t["head_dim"], dec_ffn=t["intermediate_size"],
        dec_window=t["sliding_window"], dec_theta=t["rope_parameters"]["rope_theta"],
        vocab_size=t["vocab_size"], norm_eps=t["rms_norm_eps"], ada_dim=ada_dim)


def convert(src_dir: str, out_path: str, dtype: str, meta: dict | None = None, tekken: str | None = None) -> str:
    """Write ``out_path`` (atomically). ``src_dir`` holds model.safetensors and config.json; the tokenizer is
    ``src_dir/tekken.json`` unless ``tekken`` names another file."""
    dt = DTYPES[dtype]
    src = safe_open(os.path.join(src_dir, "model.safetensors"), "pt", device="cpu")
    g = src.get_tensor

    def cv(t):
        return t.to(dt).contiguous()

    cfg = config_from_hf(src_dir, ada_dim=g("language_model.model.layers.0.ada_rms_norm.linear1.weight").shape[0])
    o: dict[str, torch.Tensor] = {}
    for n in ("conv1", "conv2"):
        o[f"encoder.{n}_w"] = cv(g(f"audio_tower.embedder.{n}.weight"))
        o[f"encoder.{n}_b"] = cv(g(f"audio_tower.embedder.{n}.bias"))
    for i in range(cfg.enc_layers):
        p, q = f"audio_tower.layers.{i}.", f"encoder.layers.{i}."
        qw, kw, vw = (g(p + f"self_attn.{n}_proj.weight") for n in "qkv")
        o[q + "wqkv"] = cv(torch.cat([qw, kw, vw]))
        o[q + "bqkv"] = cv(torch.cat([g(p + "self_attn.q_proj.bias"), torch.zeros(kw.shape[0], dtype=qw.dtype),
                                      g(p + "self_attn.v_proj.bias")]))
        o[q + "wo"], o[q + "bo"] = cv(g(p + "self_attn.o_proj.weight")), cv(g(p + "self_attn.o_proj.bias"))
        o[q + "attn_norm"] = cv(g(p + "self_attn_layer_norm.weight"))
        o[q + "ffn_norm"] = cv(g(p + "final_layer_norm.weight"))
        o[q + "w13"] = cv(torch.cat([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")]))
        o[q + "w2"], o[q + "b2"] = cv(g(p + "mlp.down_proj.weight")), cv(g(p + "mlp.down_proj.bias"))
    o["encoder.norm"] = cv(g("audio_tower.norm.weight"))
    o["encoder.proj1"] = cv(g("multi_modal_projector.linear_1.weight"))
    o["encoder.proj2"] = cv(g("multi_modal_projector.linear_2.weight"))
    o["decoder.embed"] = cv(g("language_model.model.embed_tokens.weight"))
    o["decoder.norm"] = cv(g("language_model.model.norm.weight"))
    for i in range(cfg.dec_layers):
        p, q = f"language_model.model.layers.{i}.", f"decoder.layers.{i}."
        o[q + "attn_norm"] = cv(g(p + "input_layernorm.weight"))
        o[q + "ffn_norm"] = cv(g(p + "post_attention_layernorm.weight"))
        o[q + "wqkv"] = cv(torch.cat([g(p + f"self_attn.{n}_proj.weight") for n in "qkv"]))
        o[q + "wo"] = cv(g(p + "self_attn.o_proj.weight"))
        o[q + "w13"] = cv(torch.cat([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")]))
        o[q + "w2"] = cv(g(p + "mlp.down_proj.weight"))
        o[q + "ada_down"] = cv(g(p + "ada_rms_norm.linear1.weight"))
        o[q + "ada_up"] = cv(g(p + "ada_rms_norm.linear2.weight"))
    # tokenizer byte table + guard bias, so loading never scans the 131k-token vocabulary
    import numpy as np
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer
    from .guard import Utf8Guard
    tek = Tekkenizer.from_file(tekken or os.path.join(src_dir, "tekken.json"))
    tb = [b"" if tek.is_special(i) else tek.id_to_byte_piece(i) for i in range(cfg.vocab_size)]
    offs = np.zeros(len(tb) + 1, dtype=np.int64)
    offs[1:] = np.cumsum([len(b) for b in tb])
    o["tokenizer.blob"] = torch.from_numpy(np.frombuffer(b"".join(tb), dtype=np.uint8).copy())
    o["tokenizer.offs"] = torch.from_numpy(offs)
    o["guard.boundary"] = torch.from_numpy(Utf8Guard(tb, "cpu").boundary_np.copy())
    md = {"format": "voxtral-offline", "converter_version": CONVERTER_VERSION, "dtype": dtype,
          "config": json.dumps(cfg.__dict__), **(meta or {})}
    tmp = out_path + f".tmp{os.getpid()}"
    save_file(o, tmp, metadata=md)
    os.replace(tmp, out_path)
    return out_path
