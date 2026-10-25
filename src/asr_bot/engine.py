"""Engine manager: owns ASR engines and language detectors, routes requests, serializes
GPU work on a single worker thread, and unloads idle models."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .asr import ASRBackend, BackendUnavailable, Transcript, backend_types
from .asr.base import free_torch_memory
from .audio import Audio
from .config import Config
from .lid import LanguageDetector, LanguageGuess, detector_types

log = logging.getLogger(__name__)

ENGINE_SCORING = "engine-scoring"
NO_DETECTOR = "none"
SCORING_WINDOW = 15.0


def _is_oom(e: BaseException) -> bool:
    if type(e).__name__ == "OutOfMemoryError":  # torch.OutOfMemoryError
        return True
    return "out of memory" in str(e).lower()  # CTranslate2 / CUDA runtime errors


class TranscriptionError(Exception):
    """User-presentable failure."""


@dataclass
class Plan:
    """Per-request settings snapshot."""

    engine_choice: str
    detector: str
    candidates: list[str]
    min_prob: float


@dataclass
class Result:
    transcript: Transcript
    guess: LanguageGuess | None
    engine: str
    hint: str | None
    note: str | None = None

    @property
    def language(self) -> str | None:
        return self.transcript.language or (self.guess.language if self.guess else None)


class EngineManager:
    def __init__(self, config: Config):
        self.config = config
        self.engines: dict[str, ASRBackend] = {}
        self.detectors: dict[str, LanguageDetector] = {}
        self.unavailable: dict[str, str] = {}  # name -> reason (engines and detectors)
        self.pinned: set[str] = set(config.keep_loaded)  # never unloaded (engine or detector names)
        self._last_used: dict[str, float] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr-gpu")
        self._pending = 0
        self._idle_task: asyncio.Task | None = None
        self._preload_task: asyncio.Task | None = None
        self._build()

    # ---- construction ------------------------------------------------------------------

    def _build(self) -> None:
        btypes = backend_types()
        for name, spec in self.config.engines.items():
            cls = btypes.get(spec["type"])
            if cls is None:
                self.unavailable[name] = f"unknown engine type {spec['type']!r}"
                continue
            try:
                cls.check_available()
            except BackendUnavailable as e:
                self.unavailable[name] = str(e)
                continue
            self.engines[name] = cls(name, {k: v for k, v in spec.items() if k != "type"}, self.config.device)
        dtypes = detector_types()
        for name, spec in self.config.detectors.items():
            cls = dtypes.get(spec["type"])
            if cls is None:
                self.unavailable[f"detector:{name}"] = f"unknown detector type {spec['type']!r}"
                continue
            try:
                cls.check_available()
            except BackendUnavailable as e:
                self.unavailable[f"detector:{name}"] = str(e)
                continue
            self.detectors[name] = cls(name, {k: v for k, v in spec.items() if k != "type"}, self.config.device)
        for name, reason in self.unavailable.items():
            log.warning("%s unavailable: %s", name, reason)
        if not self.engines:
            log.error("no ASR engine is available - install at least one extra, e.g. asr-bot[whisper]")

    # ---- routing -----------------------------------------------------------------------

    def route(self, language: str | None) -> str:
        """Engine for 'auto' mode."""
        if language:
            rule = self.config.route_rules.get(language)
            if rule in self.engines:
                return rule
            for name in self.config.route_fallback:
                eng = self.engines.get(name)
                if eng and eng.supports(language):
                    return name
        # Unknown language: prefer an engine that supports everything and detects languages.
        for name in self.config.route_fallback:
            eng = self.engines.get(name)
            if eng and eng.supported_languages is None:
                return name
        for name, eng in self.engines.items():
            if eng.supported_languages is None:
                return name
        if self.engines:
            return next(iter(self.engines))
        raise TranscriptionError("No ASR engine is installed on the server.")

    def resolve_engine(self, choice: str, language: str | None) -> str:
        if choice == "auto" or choice not in self.engines:
            return self.route(language)
        return choice

    # ---- public API --------------------------------------------------------------------

    @property
    def queue_length(self) -> int:
        return self._pending

    async def transcribe(
        self,
        audio: Audio,
        plan: Plan,
        on_queued: Callable[[int], Awaitable[None]] | None = None,
    ) -> Result:
        ahead = self._pending
        self._pending += 1
        try:
            if ahead and on_queued:
                await on_queued(ahead)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self._run, audio, plan)
        finally:
            self._pending -= 1

    async def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run something on the GPU thread (e.g. unloading)."""
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    def is_pinned(self, key: str) -> bool:
        return key in self.pinned or key.removeprefix("detector:") in self.pinned

    def loaded(self) -> list[str]:
        names = [n for n, e in self.engines.items() if e.is_loaded()]
        names += [f"detector:{n}" for n, d in self.detectors.items() if d.is_loaded()]
        return names

    def unload_all(self, include_pinned: bool = False) -> list[str]:
        """Unload everything that is not pinned (or everything); returns the names unloaded."""
        unloaded = []
        for key, comp in self._components():
            if comp.is_loaded() and (include_pinned or not self.is_pinned(key)):
                comp.unload()
                unloaded.append(key)
        return unloaded

    def unload_idle(self, max_idle_seconds: float) -> list[str]:
        now = time.time()
        unloaded = []
        for key, comp in self._components():
            if comp.is_loaded() and not self.is_pinned(key) and now - self._last_used.get(key, 0) > max_idle_seconds:
                log.info("unloading idle %s", key)
                comp.unload()
                unloaded.append(key)
        return unloaded

    def start_preload(self) -> None:
        """Load pinned models in the background, on the GPU thread: requests that arrive while
        they load simply queue behind it."""
        if self.pinned and self._preload_task is None:
            self._preload_task = asyncio.get_running_loop().create_task(self.call(self.warmup_pinned))

    def start_idle_unloader(self) -> None:
        minutes = self.config.idle_unload_minutes
        if minutes <= 0 or self._idle_task is not None:
            return

        async def loop() -> None:
            while True:
                await asyncio.sleep(60)
                try:
                    await self.call(self.unload_idle, minutes * 60)
                except Exception:
                    log.exception("idle unload failed")

        self._idle_task = asyncio.get_running_loop().create_task(loop())

    async def shutdown(self) -> None:
        if self._preload_task:
            self._preload_task.cancel()
            self._preload_task = None
        if self._idle_task:
            self._idle_task.cancel()
            self._idle_task = None
        try:
            await self.call(self.unload_all, True)
        finally:
            self._executor.shutdown(wait=False)

    def warmup(self, names: list[str]) -> None:
        for key, comp in self._components():
            if key in names or key.removeprefix("detector:") in names:
                log.info("loading %s", key)
                comp.load()
                self._last_used[key] = time.time()
                log.info("%s is loaded", key)

    def warmup_pinned(self) -> None:
        """Load every pinned model; a failure is logged and the model loads on first use instead."""
        for key, comp in self._components():
            if self.is_pinned(key):
                try:
                    self.warmup([key])
                except Exception:
                    log.exception("could not preload %s", key)

    # ---- the actual work (runs on the GPU thread) --------------------------------------

    def _components(self):
        for name, eng in self.engines.items():
            yield name, eng
        for name, det in self.detectors.items():
            yield f"detector:{name}", det

    def _touch(self, key: str) -> None:
        self._last_used[key] = time.time()

    def _run(self, audio: Audio, plan: Plan) -> Result:
        if audio.is_silent():
            raise TranscriptionError("The audio is silent.")
        started = time.time()
        guess = self._detect(audio, plan)
        lang = guess.language if guess else None
        engine_name = self.resolve_engine(plan.engine_choice, lang)
        engine = self.engines[engine_name]
        note = None
        hint = None
        if lang:
            confident = guess is not None and (guess.probability >= plan.min_prob or guess.method == ENGINE_SCORING)
            if not engine.supports(lang):
                note = f"{engine_name} does not list this language; results may be poor."
            elif confident:
                hint = lang
        self._touch(engine_name)
        try:
            transcript = self._with_oom_retry(engine_name, lambda: engine.transcribe(audio, hint))
        except Exception as e:
            log.exception("engine %s failed", engine_name)
            raise TranscriptionError(f"Engine {engine_name} failed: {type(e).__name__}: {e}") from e
        finally:
            self._touch(engine_name)
        if hint:
            transcript.language = hint
        elif not transcript.language and lang:
            transcript.language = lang
        transcript.engine = engine_name
        transcript.duration = audio.duration
        transcript.elapsed = time.time() - started
        return Result(transcript=transcript, guess=guess, engine=engine_name, hint=hint, note=note)

    def _with_oom_retry(self, keep: str, fn: Callable[[], Any]) -> Any:
        """Run ``fn``; on GPU out-of-memory unload every other model and try once more."""
        try:
            return fn()
        except Exception as e:
            if not _is_oom(e):
                raise
            log.warning("GPU out of memory in %s; unloading other models and retrying", keep)
        for key, comp in self._components():
            if key != keep and comp.is_loaded() and not self.is_pinned(key):
                comp.unload()
        comp = self.engines.get(keep) or self.detectors.get(keep.removeprefix("detector:"))
        if comp is not None and comp.is_loaded():
            comp.unload()  # a half-loaded model may hold fragmented memory
        free_torch_memory()
        return fn()

    def _detect(self, audio: Audio, plan: Plan) -> LanguageGuess | None:
        if plan.detector == NO_DETECTOR:
            return None
        if plan.detector == ENGINE_SCORING and plan.candidates:
            scorer = self._scoring_engine(plan.engine_choice)
            if scorer is not None:
                return self._detect_by_scoring(scorer, audio, plan.candidates)
        det = self.detectors.get(plan.detector) or next(iter(self.detectors.values()), None)
        if det is None:
            return None
        key = f"detector:{det.name}"
        self._touch(key)
        try:
            return self._with_oom_retry(key, lambda: det.detect(audio, plan.candidates))
        except Exception:
            log.exception("language detector %s failed", det.name)
            return None
        finally:
            self._touch(key)

    def _scoring_engine(self, choice: str) -> ASRBackend | None:
        eng = self.engines.get(choice)
        if eng is not None and eng.can_score:
            return eng
        for name in ["whisper", *self.engines]:
            eng = self.engines.get(name)
            if eng is not None and eng.can_score and eng.supported_languages is None:
                return eng
        return None

    def _detect_by_scoring(self, scorer: ASRBackend, audio: Audio, candidates: list[str]) -> LanguageGuess:
        clip = audio.head(SCORING_WINDOW)
        self._touch(scorer.name)
        scores = {}
        for lang in candidates:
            if scorer.supports(lang):
                try:
                    scores[lang] = scorer.score(clip, lang)
                except Exception:
                    log.exception("scoring %s failed", lang)
        self._touch(scorer.name)
        finite = {k: v for k, v in scores.items() if math.isfinite(v)}
        if not finite:
            return LanguageGuess(None, 0.0, [], ENGINE_SCORING)
        # Turn mean log-probs into a distribution (temperature chosen so a 0.2 nat/token
        # difference is a clear preference).
        m = max(finite.values())
        exp = {k: math.exp((v - m) / 0.1) for k, v in finite.items()}
        z = sum(exp.values())
        probs = {k: v / z for k, v in exp.items()}
        return LanguageGuess.from_probs(probs, [], ENGINE_SCORING)
