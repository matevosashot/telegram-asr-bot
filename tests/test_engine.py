"""Routing and language-detection logic with fake engines."""

from __future__ import annotations

import numpy as np
import pytest

from asr_bot.audio import SAMPLE_RATE, Audio
from asr_bot.engine import ENGINE_SCORING, EngineManager, Plan, TranscriptionError

from .conftest import FakeBackend, FakeLID


def _audio(seconds: float = 2.0) -> Audio:
    rng = np.random.default_rng(1)
    return Audio(rng.normal(0, 0.1, int(SAMPLE_RATE * seconds)).astype(np.float32))


def plan(**kw) -> Plan:
    return Plan(**{"engine_choice": "auto", "detector": "fake", "candidates": ["hy", "ru", "en"],
                   "min_prob": 0.5, **kw})


def test_route_rules_and_fallback(config):
    em = EngineManager(config)
    assert em.route("hy") == "armo"           # explicit rule
    assert em.route("ru") == "ru"             # first fallback engine that supports it
    assert em.route("fr") == "general"        # 'ru' engine doesn't support fr
    assert em.route(None) == "general"        # unknown language -> all-language engine


async def test_auto_uses_detected_language(config):
    em = EngineManager(config)
    r = await em.transcribe(_audio(), plan())
    # ka is not a candidate -> hy wins with 0.7/0.8
    assert r.guess.language == "hy" and r.guess.probability == pytest.approx(0.875)
    assert r.engine == "armo" and r.hint == "hy" and r.transcript.text == "բարեւ"
    assert r.language == "hy"
    await em.shutdown()


async def test_low_confidence_gives_no_hint(config):
    FakeLID.probs = {"hy": 0.4, "ru": 0.35, "en": 0.25}
    em = EngineManager(config)
    r = await em.transcribe(_audio(), plan(min_prob=0.5))
    assert r.engine == "armo" and r.hint is None
    assert FakeBackend.calls[-1] == ("armo", None)
    await em.shutdown()


async def test_fixed_engine_unsupported_language_note(config):
    em = EngineManager(config)
    r = await em.transcribe(_audio(), plan(engine_choice="ru"))
    assert r.engine == "ru" and r.hint is None and "does not list" in r.note
    await em.shutdown()


async def test_no_candidates_means_any_language(config):
    FakeLID.probs = {"ka": 0.9, "hy": 0.1}
    em = EngineManager(config)
    r = await em.transcribe(_audio(), plan(candidates=[]))
    assert r.guess.language == "ka" and r.engine == "general"
    await em.shutdown()


async def test_engine_scoring(config, monkeypatch):
    scores = {"hy": -0.9, "ru": -0.3, "en": -1.5}
    monkeypatch.setattr(FakeBackend, "can_score", True, raising=False)
    monkeypatch.setattr(FakeBackend, "score", lambda self, audio, lang: scores[lang], raising=False)
    em = EngineManager(config)
    r = await em.transcribe(_audio(), plan(detector=ENGINE_SCORING))
    assert r.guess.method == ENGINE_SCORING and r.guess.language == "ru"
    assert r.engine == "ru" and r.hint == "ru"
    await em.shutdown()


async def test_silence_rejected(config):
    em = EngineManager(config)
    with pytest.raises(TranscriptionError):
        await em.transcribe(Audio(np.zeros(SAMPLE_RATE, dtype=np.float32)), plan())
    await em.shutdown()


async def test_unknown_engine_type_is_reported(tmp_path):
    from .conftest import make_config

    cfg = make_config(tmp_path, engines={"weird": {"type": "does-not-exist"}})
    em = EngineManager(cfg)
    assert "weird" in em.unavailable and "weird" not in em.engines


def test_idle_unload(config):
    em = EngineManager(config)
    em.warmup(["general"])
    assert em.engines["general"].is_loaded()
    assert em.unload_idle(3600) == []
    em._last_used["general"] = 0
    assert em.unload_idle(3600) == ["general"]
    assert not em.engines["general"].is_loaded()


async def test_oom_unloads_other_models_and_retries(config, monkeypatch):
    from asr_bot.asr import Transcript

    em = EngineManager(config)
    em.warmup(["general", "ru"])
    attempts = []

    def flaky(self, audio, language=None):
        attempts.append(self.name)
        if len(attempts) == 1:
            raise RuntimeError("CUDA failed with error out of memory")
        return Transcript("ok", "hy")

    monkeypatch.setattr(FakeBackend, "transcribe", flaky)
    r = await em.transcribe(_audio(), plan())
    assert r.transcript.text == "ok" and attempts == ["armo", "armo"]
    assert not em.engines["general"].is_loaded() and not em.engines["ru"].is_loaded()
    await em.shutdown()


def test_pinned_engines_are_never_unloaded(tmp_path):
    from .conftest import make_config

    em = EngineManager(make_config(tmp_path, transcription={"keep_loaded": ["armo"]}))
    em.warmup_pinned()
    em.warmup(["general"])
    assert em.engines["armo"].is_loaded() and em.engines["general"].is_loaded()
    assert em.unload_all() == ["general"]
    em._last_used["armo"] = 0
    assert em.unload_idle(3600) == []
    assert em.engines["armo"].is_loaded()
    assert em.unload_all(include_pinned=True) == ["armo"]


async def test_oom_retry_spares_pinned_models(tmp_path, monkeypatch):
    from asr_bot.asr import Transcript
    from .conftest import make_config

    em = EngineManager(make_config(tmp_path, transcription={"keep_loaded": ["ru"]}))
    em.warmup(["general", "ru"])
    attempts = []

    def flaky(self, audio, language=None):
        attempts.append(self.name)
        if len(attempts) == 1:
            raise RuntimeError("CUDA failed with error out of memory")
        return Transcript("ok", "hy")

    monkeypatch.setattr(FakeBackend, "transcribe", flaky)
    await em.transcribe(_audio(), plan())
    assert not em.engines["general"].is_loaded() and em.engines["ru"].is_loaded()
    await em.shutdown()
    assert not em.engines["ru"].is_loaded()  # shutdown still releases pinned models


async def test_detector_none_skips_detection(config):
    em = EngineManager(config)
    r = await em.transcribe(_audio(), plan(detector="none", engine_choice="armo"))
    assert r.guess is None and r.hint is None and r.engine == "armo"
    from asr_bot.bot.format import header

    r.transcript.language = None
    assert header(r).startswith("armo")  # no flag / language name
    await em.shutdown()


def test_keep_loaded_must_name_a_known_engine(tmp_path):
    from asr_bot.config import ConfigError
    from .conftest import make_config

    with pytest.raises(ConfigError):
        make_config(tmp_path, transcription={"keep_loaded": ["nope"]})
