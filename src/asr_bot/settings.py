"""Runtime settings: config-file defaults overridden by values the owner set in the bot."""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any

from .config import Defaults
from .db import Database


class Settings:
    def __init__(self, defaults: Defaults, db: Database):
        self._defaults = defaults
        self._db = db
        self._overrides: dict[str, Any] = {}

    async def load(self) -> "Settings":
        known = {f.name for f in fields(Defaults)}
        self._overrides = {k: v for k, v in (await self._db.get_settings()).items() if k in known}
        return self

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._defaults, name)

    def is_overridden(self, name: str) -> bool:
        return name in self._overrides

    async def set(self, name: str, value: Any) -> None:
        if not hasattr(self._defaults, name):
            raise KeyError(name)
        if value == getattr(self._defaults, name):
            self._overrides.pop(name, None)
            await self._db.delete_setting(name)
        else:
            self._overrides[name] = value
            await self._db.set_setting(name, value)

    async def reset(self, name: str) -> None:
        self._overrides.pop(name, None)
        await self._db.delete_setting(name)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self._defaults), **self._overrides}
