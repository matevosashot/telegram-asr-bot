"""Shared objects the handlers need, stored in ``application.bot_data``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from telegram.ext import ContextTypes

from ..auth import AccessControl
from ..config import Config
from ..db import Database
from ..engine import EngineManager
from ..limits import RateLimiter
from ..settings import Settings

KEY = "asr_bot_services"


@dataclass
class Services:
    config: Config
    db: Database
    settings: Settings
    access: AccessControl
    limiter: RateLimiter
    engines: EngineManager
    tmp_dir: Path
    bot_username: str = ""
    # Short-lived ids for callback data that would not fit in 64 bytes.
    scratch: dict[str, object] = field(default_factory=dict)

    @property
    def owner_id(self) -> int:
        return self.config.telegram.owner_id


def services(context: ContextTypes.DEFAULT_TYPE) -> Services:
    return context.application.bot_data[KEY]
