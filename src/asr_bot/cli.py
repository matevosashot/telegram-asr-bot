"""Command line interface: ``asr-bot run | check | transcribe | init-config``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
import tempfile
import time
from importlib.resources import files
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, ConfigError, load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="asr-bot", description=__doc__)
    parser.add_argument("-c", "--config", type=Path, default=None,
                        help=f"config file (default: $ASR_BOT_CONFIG or {DEFAULT_CONFIG_PATH})")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="run the bot (long polling)")
    sub.add_parser("check", help="check config, token, ffmpeg and which engines are available")
    p = sub.add_parser("transcribe", help="transcribe local files (no Telegram), for testing engines")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("-e", "--engine", default=None, help="engine name or 'auto' (default from config)")
    p.add_argument("-d", "--detector", default=None, help="language detector name or 'engine-scoring'")
    p.add_argument("-l", "--languages", default=None, help="candidate languages, e.g. hy,ru,en ('' = any)")
    p = sub.add_parser("init-config", help="write an example config file")
    p.add_argument("--owner-id", type=int, default=None)
    p.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.cmd == "init-config":
        return _init_config(args)
    config_path = args.config or _env_config()
    try:
        config = load_config(config_path)
    except ConfigError as e:
        sys.exit(f"error: {e}")
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else config.log_level)

    if args.cmd == "run":
        from .bot.app import run

        try:
            token = config.telegram.resolve_token()
        except ConfigError as e:
            sys.exit(f"error: {e}")
        run(config, token)
    elif args.cmd == "check":
        sys.exit(_check(config))
    elif args.cmd == "transcribe":
        _transcribe(config, args)


def _env_config() -> Path | None:
    import os

    value = os.environ.get("ASR_BOT_CONFIG")
    return Path(value) if value else None


def _init_config(args) -> None:
    dest = args.config or DEFAULT_CONFIG_PATH
    if dest.exists() and not args.force:
        sys.exit(f"{dest} already exists (use --force to overwrite)")
    text = files("asr_bot").joinpath("config.example.toml").read_text()
    if args.owner_id:
        import re

        text = re.sub(r"^owner_id = \d+", f"owner_id = {args.owner_id}", text, flags=re.M)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    print(f"wrote {dest} - edit telegram.owner_id and export TELEGRAM_BOT_TOKEN, then run `asr-bot run`")


def _check(config) -> int:
    from .engine import EngineManager

    ok = True
    print(f"config: {config.path}\nowner id: {config.telegram.owner_id}\ndatabase: {config.db_path}")
    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        ok &= bool(found)
        print(f"{tool}: {found or 'MISSING'}")
    try:
        token = config.telegram.resolve_token()
        import httpx

        r = httpx.get(f"https://api.telegram.org/bot{token}/getMe", timeout=20).json()
        if r.get("ok"):
            me = r["result"]
            print(f"bot: @{me['username']} (guest mode: {me.get('supports_guest_queries')}, "
                  f"business: {me.get('can_connect_to_business')}, "
                  f"reads all group messages: {me.get('can_read_all_group_messages')})")
        else:
            ok = False
            print(f"bot token rejected: {r.get('description')}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"bot: {e}")
    em = EngineManager(config)
    for name, eng in em.engines.items():
        print(f"engine {name}: {eng.display_name} ({eng.model_name}) - available")
    for name, det in em.detectors.items():
        print(f"detector {name}: {det.display_name} - available")
    for name, reason in em.unavailable.items():
        print(f"{name}: UNAVAILABLE - {reason}")
    ok &= bool(em.engines)
    return 0 if ok else 1


def _transcribe(config, args) -> None:
    from . import languages
    from .audio import Audio
    from .engine import EngineManager, Plan

    em = EngineManager(config)
    d = config.defaults
    cands = d.candidate_languages if args.languages is None else [c for c in args.languages.split(",") if c]
    plan = Plan(engine_choice=args.engine or d.default_engine, detector=args.detector or d.language_detector,
                candidates=cands, min_prob=d.min_language_prob)

    workdir = Path(tempfile.mkdtemp(prefix="asr-bot-"))

    async def go() -> None:
        try:
            for path in args.files:
                audio = Audio.from_file(path, workdir=workdir)
                started = time.time()
                r = await em.transcribe(audio, plan)
                g = r.guess
                guess = f"{g.language} {g.probability:.2f} via {g.method} {g.top}" if g else "-"
                print(f"== {path} [{languages.name(r.language)}; engine {r.engine}; "
                      f"{audio.duration:.1f}s audio in {time.time() - started:.1f}s; lid: {guess}]")
                print(r.transcript.text)
                if r.note:
                    print(f"note: {r.note}")
                audio.cleanup()
        finally:
            await em.shutdown()
            shutil.rmtree(workdir, ignore_errors=True)

    asyncio.run(go())
