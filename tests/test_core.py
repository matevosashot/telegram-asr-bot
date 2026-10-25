"""Database, access control, limits, settings, languages, formatting, audio."""

from __future__ import annotations

import time

import numpy as np
import pytest

from asr_bot import languages
from asr_bot.asr import Transcript
from asr_bot.audio import SAMPLE_RATE, Audio, chunk, probe_duration
from asr_bot.auth import AccessControl, Role, parse_target
from asr_bot.bot import format as fmt
from asr_bot.config import ConfigError, config_from_dict
from asr_bot.db import Database
from asr_bot.engine import Result
from asr_bot.limits import RateLimiter, fmt_duration
from asr_bot.settings import Settings

from .conftest import OWNER_ID, make_config


@pytest.fixture
async def db(tmp_path):
    d = await Database(tmp_path / "t.db").open()
    yield d
    await d.close()


async def test_owner_and_unknown(db):
    ac = AccessControl(db, OWNER_ID)
    assert (await ac.check(OWNER_ID, "me", "Me", None)).role == Role.OWNER
    res = await ac.check(5, "stranger", "S", None)
    assert res.role == Role.UNKNOWN and not res.role.can_transcribe


async def test_allow_known_user_by_username(db):
    ac = AccessControl(db, OWNER_ID)
    await ac.check(5, "Friend", "F", None)
    outcome, user = await ac.allow("@friend")
    assert outcome == "allowed" and user.id == 5
    assert (await ac.check(5, "Friend", "F", None)).role == Role.ALLOWED


async def test_pending_username_auto_approves_on_first_contact(db):
    ac = AccessControl(db, OWNER_ID)
    outcome, _ = await ac.allow("@NewGuy")
    assert outcome == "pending"
    assert await db.list_pending_usernames() == ["newguy"]
    res = await ac.check(7, "newguy", "N", None)
    assert res.role == Role.ALLOWED and res.newly_approved
    assert await db.list_pending_usernames() == []
    # second contact is not "newly approved"
    assert not (await ac.check(7, "newguy", "N", None)).newly_approved


async def test_deny_and_block(db):
    ac = AccessControl(db, OWNER_ID)
    await ac.allow(8)
    assert (await ac.check(8, None, "X", None)).role == Role.ALLOWED
    assert (await ac.deny(8))[0] == "removed"
    assert (await ac.check(8, None, "X", None)).role == Role.UNKNOWN
    assert (await ac.block(8))[0] == "blocked"
    assert (await ac.check(8, None, "X", None)).role == Role.BLOCKED
    assert (await ac.block(OWNER_ID))[0] in ("owner", "not_found")
    await ac.allow("@later")
    assert (await ac.deny("@later"))[0] == "pending_removed"


async def test_blocked_user_is_not_auto_approved(db):
    ac = AccessControl(db, OWNER_ID)
    await ac.check(9, "bad", "B", None)
    await ac.block(9)
    await ac.allow("@bad")  # known -> allowed explicitly; check the pending path instead
    await ac.block(9)
    await db.add_pending_username("bad")
    assert (await ac.check(9, "bad", "B", None)).role == Role.BLOCKED


def test_parse_target():
    assert parse_target("123") == 123
    assert parse_target("@Some_User") == "@some_user"
    assert parse_target("bad name!") is None


async def test_rate_limits(db):
    lim = RateLimiter(db)
    now = time.time()
    for i in range(2):
        await db.log_interaction(user_id=5, chat_id=5, chat_type="private", source="private", kind="transcribe",
                                 ts=now - 100 - i)
    await db.log_interaction(user_id=5, chat_id=5, chat_type="private", source="private", kind="denied", ts=now)
    d = await lim.check(5, per_hour=2, per_day=10, max_audio_seconds=60, duration=10, now=now)
    assert not d.allowed and "per hour" in d.reason and 3400 < d.retry_after < 3600
    d = await lim.check(5, per_hour=3, per_day=10, max_audio_seconds=60, duration=10, now=now)
    assert d.allowed
    d = await lim.check(5, per_hour=0, per_day=2, max_audio_seconds=60, duration=10, now=now)
    assert not d.allowed and "per day" in d.reason
    d = await lim.check(5, per_hour=10, per_day=10, max_audio_seconds=60, duration=61, now=now)
    assert not d.allowed and "too long" in d.reason
    # outside the window
    d = await lim.check(5, per_hour=2, per_day=10, max_audio_seconds=60, duration=1, now=now + 3700)
    assert d.allowed


async def test_settings_override_and_reset(db, tmp_path):
    cfg = make_config(tmp_path)
    s = await Settings(cfg.defaults, db).load()
    assert s.per_hour == 2
    await s.set("per_hour", 9)
    assert s.per_hour == 9
    s2 = await Settings(cfg.defaults, db).load()
    assert s2.per_hour == 9 and s2.is_overridden("per_hour")
    await s2.set("per_hour", 2)  # back to default -> override removed
    assert not s2.is_overridden("per_hour")
    with pytest.raises(KeyError):
        await s.set("nope", 1)


def test_config_validation(tmp_path):
    with pytest.raises(ConfigError):
        config_from_dict({"telegram": {}})
    with pytest.raises(ConfigError):
        make_config(tmp_path, routing={"rules": {"hy": "missing"}})
    cfg = config_from_dict({"telegram": {"owner_id": 1}})
    assert set(cfg.engines) == {"whisper", "qwen3", "omnilingual"}
    assert cfg.route_rules == {"hy": "omnilingual"}


def test_languages():
    assert languages.normalize("hye_Armn") == "hy"
    assert languages.normalize("hy: Armenian") == "hy"
    assert languages.normalize("Russian") == "ru"
    assert languages.normalize("zho") == "zh"
    assert languages.to_iso3("hy") == "hye"
    r = languages.restrict({"hy": 0.2, "ka": 0.6, "en": 0.2}, ["hy", "en"])
    assert r == {"hy": 0.5, "en": 0.5}
    assert languages.restrict({"ka": 1.0}, ["hy"]) == {"ka": 1.0}


def test_split_text():
    text = ("Sentence number one. " * 400).strip()
    chunks = fmt.split_text(text, 1000)
    assert all(len(c) <= 1000 for c in chunks)
    assert " ".join(chunks).replace("  ", " ") == text
    assert fmt.split_text("short") == ["short"]


def _result(text: str) -> Result:
    return Result(transcript=Transcript(text=text, language="hy", engine="armo", duration=42), guess=None,
                  engine="armo", hint="hy")


def test_render():
    out = fmt.render(_result("<b>hi</b>"))
    assert len(out) == 1 and "🇦🇲 Armenian · armo · 0:42" in out[0] and "&lt;b&gt;" in out[0]
    assert "(no speech recognized)" in fmt.render(_result(""))[0]
    assert len(fmt.render(_result("word " * 2000))) == 3
    assert fmt.render(_result("word " * 5000)) == []
    assert fmt_duration(3725) == "1:02:05"


def test_audio_conversion(tone_ogg, tmp_path):
    assert 1.9 < probe_duration(tone_ogg) < 2.2
    a = Audio.from_file(tone_ogg, tmp_path)
    assert a.path.exists() and 1.9 < a.duration < 2.2 and not a.is_silent()
    assert a.head(1.0).duration == 1.0
    assert Audio(np.zeros(SAMPLE_RATE, dtype=np.float32)).is_silent()


def test_chunking():
    samples = np.random.default_rng(0).normal(0, 0.1, SAMPLE_RATE * 100).astype(np.float32)
    samples[SAMPLE_RATE * 31: SAMPLE_RATE * 32] = 0  # a pause where the cut should go
    parts = chunk(samples, 35)
    assert sum(len(p) for p in parts) == len(samples)
    assert all(len(p) <= 35 * SAMPLE_RATE for p in parts)
    assert 31 * SAMPLE_RATE <= len(parts[0]) <= 32 * SAMPLE_RATE


async def test_concurrent_pending_approval_wins_once(db):
    import asyncio

    await db.add_pending_username("twin")
    results = await asyncio.gather(*(db.touch_user(80, "twin", "T", None) for _ in range(3)))
    assert sum(approved for _, approved in results) == 1


async def test_username_moves_to_new_account(db):
    ac = AccessControl(db, OWNER_ID)
    await ac.check(81, "handle", "Old", None)
    await ac.check(82, "handle", "New", None)
    outcome, user = await ac.allow("@handle")
    assert user.id == 82


async def test_block_unseen_username_drops_pending(db):
    ac = AccessControl(db, OWNER_ID)
    await ac.allow("@ghost")
    assert (await ac.block("@ghost"))[0] == "not_found"
    assert await db.list_pending_usernames() == []


def test_language_aliases():
    assert languages.normalize("iw") == "he"
    assert languages.normalize("jv") == "jw"


def test_engine_name_length_validated(tmp_path):
    with pytest.raises(ConfigError):
        make_config(tmp_path, engines={"x" * 40: {"type": "fake"}})
