"""Static configuration loaded from a TOML file.

Values that the owner can change at runtime (default engine, limits, ...) are only
*defaults* here; the live values are kept in the database (see ``settings.py``).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

DEFAULT_CONFIG_PATH = Path("~/.config/asr-bot/config.toml").expanduser()


class ConfigError(Exception):
    pass


@dataclass
class TelegramConfig:
    owner_id: int
    token_env: str = "TELEGRAM_BOT_TOKEN"
    token: str | None = None
    # Self-hosted Bot API server (lifts the 20 MB download limit), e.g. "http://localhost:8081"
    local_api_url: str | None = None

    def resolve_token(self) -> str:
        token = os.environ.get(self.token_env) or self.token
        if not token:
            raise ConfigError(
                f"No bot token: set the {self.token_env} environment variable "
                "or telegram.token in the config file"
            )
        return token


@dataclass
class Defaults:
    """Defaults for settings that can be changed from the bot's admin menu."""

    default_engine: str = "auto"
    language_detector: str = "whisper"
    candidate_languages: list[str] = field(default_factory=lambda: ["hy", "ru", "en"])
    min_language_prob: float = 0.5
    per_hour: int = 5
    per_day: int = 20
    max_audio_seconds: int = 180
    groups: bool = True
    guest: bool = True
    business: bool = True
    business_reply_in_chat: bool = False
    business_auto_transcribe: bool = False
    notify_unknown_users: bool = True


@dataclass
class Config:
    telegram: TelegramConfig
    data_dir: Path
    defaults: Defaults
    # Routing for engine "auto": explicit language -> engine rules, then the first engine
    # in ``fallback`` that lists the language as supported (or supports all languages).
    route_rules: dict[str, str]
    route_fallback: list[str]
    engines: dict[str, dict[str, Any]]
    detectors: dict[str, dict[str, Any]]
    device: str = "cuda"
    idle_unload_minutes: float = 15
    store_transcripts: bool = False
    log_level: str = "INFO"
    path: Path | None = None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "bot.db"


DEFAULT_ENGINES: dict[str, dict[str, Any]] = {
    "whisper": {"type": "faster-whisper", "model": "large-v3", "compute_type": "float16"},
    "qwen3": {"type": "qwen3-asr", "model": "Qwen/Qwen3-ASR-1.7B"},
    "omnilingual": {"type": "omnilingual", "model": "omniASR_LLM_Unlimited_3B_v2"},
}
DEFAULT_DETECTORS: dict[str, dict[str, Any]] = {
    "whisper": {"type": "whisper-lid", "model": "large-v3-turbo"},
    "voxlingua": {"type": "voxlingua107"},
}


def _defaults_from(data: dict[str, Any]) -> Defaults:
    d = Defaults()
    tr = data.get("transcription", {})
    lim = data.get("limits", {})
    feat = data.get("features", {})
    for src in (tr, lim, feat):
        for key, value in src.items():
            if hasattr(d, key):
                setattr(d, key, value)
    d.candidate_languages = [c.lower() for c in d.candidate_languages]
    return d


def load_config(path: Path | str | None = None) -> Config:
    path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"Config file not found: {path} (create one with `asr-bot init-config`)")
    with path.open("rb") as f:
        data = tomllib.load(f)
    return config_from_dict(data, path=path)


def config_from_dict(data: dict[str, Any], path: Path | None = None) -> Config:
    tg = data.get("telegram", {})
    if "owner_id" not in tg:
        raise ConfigError("telegram.owner_id is required (your numeric Telegram user id)")
    telegram = TelegramConfig(
        owner_id=int(tg["owner_id"]),
        token_env=tg.get("token_env", "TELEGRAM_BOT_TOKEN"),
        token=tg.get("token") or None,
        local_api_url=tg.get("local_api_url") or None,
    )
    tr = data.get("transcription", {})
    routing = data.get("routing", {})
    engines = data.get("engines") or DEFAULT_ENGINES
    detectors = data.get("detectors") or DEFAULT_DETECTORS
    for name, spec in {**engines, **detectors}.items():
        if "type" not in spec:
            raise ConfigError(f"engine/detector {name!r} has no 'type'")
    defaults = _defaults_from(data)
    route_rules = {k.lower(): v for k, v in routing.get("rules", {"hy": "omnilingual"}).items()}
    route_fallback = list(routing.get("fallback", ["qwen3", "whisper"]))
    cfg = Config(
        telegram=telegram,
        data_dir=Path(data.get("storage", {}).get("data_dir", "~/.local/share/asr-bot")).expanduser(),
        defaults=defaults,
        route_rules=route_rules,
        route_fallback=route_fallback,
        engines=engines,
        detectors=detectors,
        device=tr.get("device", "cuda"),
        idle_unload_minutes=float(tr.get("idle_unload_minutes", 15)),
        store_transcripts=bool(data.get("storage", {}).get("store_transcripts", False)),
        log_level=data.get("logging", {}).get("level", "INFO"),
        path=path,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    names = set(cfg.engines)
    for name in [*cfg.engines, *cfg.detectors]:
        # Names end up in inline-button callback data (max 64 bytes).
        if len(name.encode()) > 32 or ":" in name:
            raise ConfigError(f"engine/detector name {name!r} must be at most 32 bytes and contain no ':'")
    for lang, engine in cfg.route_rules.items():
        if engine not in names:
            raise ConfigError(f"routing.rules: {lang} -> unknown engine {engine!r}")
    for engine in cfg.route_fallback:
        if engine not in names:
            raise ConfigError(f"routing.fallback: unknown engine {engine!r}")
    d = cfg.defaults
    if d.default_engine != "auto" and d.default_engine not in names:
        raise ConfigError(f"default_engine {d.default_engine!r} is not a configured engine")
    if d.language_detector not in cfg.detectors and d.language_detector != "engine-scoring":
        raise ConfigError(f"language_detector {d.language_detector!r} is not configured")
