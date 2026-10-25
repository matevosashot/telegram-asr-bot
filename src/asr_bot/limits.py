"""Per-user request limits (owner is exempt)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from .db import Database

HOUR = 3600
DAY = 86400


@dataclass
class LimitDecision:
    allowed: bool
    reason: str | None = None
    retry_after: float | None = None  # seconds


@dataclass
class Usage:
    last_hour: int
    last_day: int
    per_hour: int
    per_day: int


class RateLimiter:
    def __init__(self, db: Database):
        self.db = db
        self._locks: dict[int, asyncio.Lock] = {}

    def lock(self, user_id: int) -> asyncio.Lock:
        """Hold while checking the limit and recording the request, so concurrent
        updates from one user can't all pass the check before any is counted."""
        return self._locks.setdefault(user_id, asyncio.Lock())

    async def usage(self, user_id: int, per_hour: int, per_day: int, now: float | None = None) -> Usage:
        now = now or time.time()
        return Usage(
            last_hour=await self.db.count_requests(user_id, now - HOUR),
            last_day=await self.db.count_requests(user_id, now - DAY),
            per_hour=per_hour,
            per_day=per_day,
        )

    async def check(
        self,
        user_id: int,
        *,
        per_hour: int,
        per_day: int,
        max_audio_seconds: float,
        duration: float | None,
        now: float | None = None,
    ) -> LimitDecision:
        now = now or time.time()
        if duration is not None and max_audio_seconds and duration > max_audio_seconds:
            return LimitDecision(
                False,
                f"Audio is too long ({fmt_duration(duration)}); the limit is {fmt_duration(max_audio_seconds)}.",
            )
        for window, limit, label in ((HOUR, per_hour, "hour"), (DAY, per_day, "day")):
            if limit <= 0:
                continue
            since = now - window
            used = await self.db.count_requests(user_id, since)
            if used >= limit:
                oldest = await self.db.oldest_request_since(user_id, since)
                retry = (oldest + window - now) if oldest else window
                return LimitDecision(
                    False,
                    f"Limit reached: {limit} requests per {label}. Try again in {fmt_duration(retry)}.",
                    retry,
                )
        return LimitDecision(True)


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
