"""Meta Omnilingual ASR (https://github.com/facebookresearch/omnilingual-asr)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .. import languages
from ..audio import SAMPLE_RATE, Audio, chunk
from .base import ASRBackend, BackendUnavailable, Transcript, free_torch_memory, register_backend

log = logging.getLogger(__name__)

# Non-"Unlimited" model cards only accept audio shorter than 40 s.
_SHORT_MODEL_MAX_SECONDS = 35.0


@register_backend("omnilingual")
class OmnilingualBackend(ASRBackend):
    display_name = "Meta Omnilingual"
    supported_languages = None  # 1600+ languages
    detects_language = False

    def __init__(self, name: str, options: dict[str, Any], device: str = "cuda"):
        super().__init__(name, options, device)
        self._pipeline = None
        self._codes: dict[str, str] | None = None

    @classmethod
    def check_available(cls) -> None:
        try:
            import omnilingual_asr  # noqa: F401
        except ImportError as e:
            raise BackendUnavailable(
                "omnilingual-asr is not installed (pip install 'asr-bot[omni]')"
            ) from e

    @property
    def unlimited(self) -> bool:
        return "Unlimited" in (self.model_name or "")

    def load(self) -> None:
        _init_fairseq2_thread()
        if self._pipeline is not None:
            return
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        card = self.model_name or "omniASR_LLM_Unlimited_3B_v2"
        device = self.device or None
        cache = self._weights_path(card)
        if cache is not None and cache.exists():
            try:
                self._pipeline = _load_from_bf16(card, cache, device)
                return
            except Exception:
                log.exception("fast weights %s unusable; loading the original checkpoint", cache)
        kwargs: dict[str, Any] = {"model_card": card}
        if device:
            kwargs["device"] = device
        self._pipeline = ASRInferencePipeline(**kwargs)
        if cache is not None and not cache.exists():
            try:
                _save_bf16(self._pipeline.model, cache)
            except Exception:
                log.exception("could not write fast weights %s", cache)

    def _weights_path(self, card: str) -> Path | None:
        """Where the bf16 safetensors copy of the checkpoint lives (None = feature off).

        The original checkpoint is fp32 (16 GB for the 3B model); a bf16 copy is half the size
        and skips the conversion, so a (re)load takes ~10 s instead of ~20 s."""
        if not self.options.get("fast_weights", True):
            return None
        base = self.options.get("weights_dir") or os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "asr-bot", "weights"
        )
        return Path(base).expanduser() / f"{card}.bf16.safetensors"

    def unload(self) -> None:
        self._pipeline = None
        free_torch_memory()

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def lang_code(self, language: str | None) -> str | None:
        """ISO-639-1 -> the model's code, e.g. 'hy' -> 'hye_Armn'."""
        if not language:
            return None
        if self._codes is None:
            self._codes = _supported_codes()
        iso3 = languages.to_iso3(language)
        if not iso3:
            return None
        if language in self.options.get("codes", {}):
            return self.options["codes"][language]
        matches = sorted(c for c in self._codes if c.split("_", 1)[0] == iso3)
        return matches[0] if matches else None

    def transcribe(self, audio: Audio, language: str | None = None) -> Transcript:
        self.load()
        code = self.lang_code(language)
        pieces = [audio.samples] if self.unlimited else chunk(audio.samples, _SHORT_MODEL_MAX_SECONDS)
        inputs = [{"waveform": p, "sample_rate": SAMPLE_RATE} for p in pieces]
        kwargs: dict[str, Any] = {"batch_size": int(self.options.get("batch_size", 1))}
        if code and "CTC" not in (self.model_name or ""):
            kwargs["lang"] = [code] * len(inputs)
        texts = self._pipeline.transcribe(inputs, **kwargs)
        text = " ".join(t.strip() for t in texts if t and t.strip())
        return Transcript(text=text, language=language if code else None)


def _load_from_bf16(card: str, path: Path, device: str | None):
    """Build the model empty on ``device`` and fill it from a bf16 safetensors file."""
    import torch
    from fairseq2.data.tokenizers.hub import load_tokenizer
    from fairseq2.nn.utils.module import reset_non_persistent_buffers, to_empty
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    from omnilingual_asr.models.wav2vec2_llama.hub import get_wav2vec2_llama_model_hub
    from safetensors.torch import load_file

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    hub = get_wav2vec2_llama_model_hub()
    model = hub.create_new_model(hub.get_model_config(card), device=dev, dtype=torch.bfloat16, meta=True)
    to_empty(model, device=dev)
    model.load_state_dict(load_file(str(path)), strict=True)
    reset_non_persistent_buffers(model)
    return ASRInferencePipeline(None, model=model, tokenizer=load_tokenizer(card), device=dev)


def _save_bf16(model, path: Path) -> None:
    from safetensors.torch import save_file

    state, seen = {}, set()
    for key, tensor in model.state_dict().items():
        t = tensor.detach().cpu().contiguous()
        state[key] = t.clone() if t.data_ptr() in seen else t  # safetensors rejects shared storage
        seen.add(t.data_ptr())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    save_file(state, str(tmp))
    os.replace(tmp, path)  # never leave a half-written file behind
    log.info("saved fast weights %s (%.1f GB)", path, path.stat().st_size / 1e9)


def _init_fairseq2_thread() -> None:
    """fairseq2 0.6 initializes its thread-local gang stack only in the importing thread;
    the bot runs models on a dedicated worker thread, so set it up there too."""
    from fairseq2 import gang

    if not hasattr(gang._thread_local, "current_gangs"):
        gang._thread_local.current_gangs = []


def _supported_codes() -> set[str]:
    try:
        from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

        return set(supported_langs)
    except ImportError:
        return set()
