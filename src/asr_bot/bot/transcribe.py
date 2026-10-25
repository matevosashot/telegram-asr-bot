"""The common path for every entry point: check access and limits, download, transcribe."""

from __future__ import annotations

import asyncio
import html
import io
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Message, User
from telegram.constants import FileSizeLimit, ParseMode
from telegram.error import TelegramError

from ..audio import Audio, AudioError, probe_duration
from ..auth import AccessResult, Role
from ..engine import Plan, Result, TranscriptionError
from ..limits import fmt_duration
from . import format as fmt
from .services import Services

log = logging.getLogger(__name__)


class UserFacingError(Exception):
    """An error whose message is shown to the user as-is."""


@dataclass
class Media:
    file_id: str
    kind: str
    file_size: int | None
    duration: float | None
    file_name: str | None = None


def extract_media(message: Message | None) -> Media | None:
    """Find transcribable media in a message (voice, audio, round video, video, audio/video file)."""
    if message is None:
        return None
    if message.voice:
        v = message.voice
        return Media(v.file_id, "voice", v.file_size, _secs(v.duration))
    if message.audio:
        a = message.audio
        return Media(a.file_id, "audio", a.file_size, _secs(a.duration), a.file_name)
    if message.video_note:
        v = message.video_note
        return Media(v.file_id, "video_note", v.file_size, _secs(v.duration))
    if message.video:
        v = message.video
        return Media(v.file_id, "video", v.file_size, _secs(v.duration), v.file_name)
    doc = message.document
    if doc and doc.mime_type and doc.mime_type.split("/")[0] in ("audio", "video"):
        return Media(doc.file_id, "document", doc.file_size, None, doc.file_name)
    if doc and doc.file_name and doc.file_name.lower().endswith(
        (".mp3", ".m4a", ".ogg", ".oga", ".opus", ".wav", ".flac", ".aac", ".wma", ".webm", ".mp4", ".mkv", ".mov", ".amr")
    ):
        return Media(doc.file_id, "document", doc.file_size, None, doc.file_name)
    return None


def _secs(value) -> float | None:
    """PTB returns durations as int or timedelta depending on PTB_TIMEDELTA."""
    if value is None:
        return None
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    return float(value)


# ---- access ------------------------------------------------------------------------------


async def check_access(svc: Services, bot: Bot, user: User | None, *, source: str, chat_id: int | None = None,
                       chat_type: str | None = None) -> AccessResult | None:
    """Record the user, and notify the owner about strangers. Returns None for no user."""
    if user is None or user.is_bot:
        return None
    res = await svc.access.check(user.id, user.username, user.first_name, user.last_name)
    if res.newly_approved:
        await _safe_send(bot, svc.owner_id, f"✅ {res.user.display} (id {user.id}) was pre-authorized by "
                                            "username and has now started using the bot.")
    if res.role == Role.UNKNOWN:
        await svc.db.log_interaction(user_id=user.id, chat_id=chat_id, chat_type=chat_type, source=source,
                                     kind="denied")
        if svc.settings.notify_unknown_users and await svc.db.claim_notification(user.id):
            await notify_owner_about(svc, bot, res, source)
    return res


async def notify_owner_about(svc: Services, bot: Bot, res: AccessResult, source: str) -> None:
    u = res.user
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Allow", callback_data=f"acc:allow:{u.id}"),
        InlineKeyboardButton("⛔ Block", callback_data=f"acc:block:{u.id}"),
        InlineKeyboardButton("🙈 Ignore", callback_data=f"acc:ignore:{u.id}"),
    ]])
    text = f"👤 New user tried the bot ({source}):\n{html.escape(u.display)}\nid: <code>{u.id}</code>"
    await _safe_send(bot, svc.owner_id, text, reply_markup=kb, parse_mode=ParseMode.HTML)


async def _safe_send(bot: Bot, chat_id: int, text: str, **kwargs) -> Message | None:
    try:
        return await bot.send_message(chat_id, text, **kwargs)
    except TelegramError as e:
        log.warning("could not send to %s: %s", chat_id, e)
        return None


# ---- transcription -----------------------------------------------------------------------

StatusCallback = Callable[[str], Awaitable[None]]


async def transcribe_media(
    svc: Services,
    bot: Bot,
    media: Media,
    *,
    access: AccessResult,
    chat_id: int | None,
    chat_type: str | None,
    source: str,
    status: StatusCallback | None = None,
) -> Result:
    """Check limits, download and transcribe. Raises UserFacingError with a friendly message."""
    user = access.user
    s = svc.settings
    is_owner = access.role == Role.OWNER

    async def log_kind(kind: str, **kw) -> int:
        return await svc.db.log_interaction(user_id=user.id, chat_id=chat_id, chat_type=chat_type,
                                            source=source, kind=kind, **kw)

    if (media.file_size and media.file_size > FileSizeLimit.FILESIZE_DOWNLOAD
            and not svc.config.telegram.local_api_url):
        await log_kind("too_big", duration=media.duration, detail=str(media.file_size))
        raise UserFacingError(
            f"📦 The file is {media.file_size / 1e6:.0f} MB; Telegram lets bots download at most 20 MB. "
            "Send a shorter clip or a compressed audio file.")

    # Check the limit and record the request atomically per user; the row is updated
    # with the outcome later (and stays counted, unless rejected for length).
    async with svc.limiter.lock(user.id):
        if not is_owner:
            decision = await svc.limiter.check(user.id, per_hour=s.per_hour, per_day=s.per_day,
                                               max_audio_seconds=s.max_audio_seconds, duration=media.duration)
            if not decision.allowed:
                await log_kind("limited", duration=media.duration, detail=decision.reason)
                raise UserFacingError("⏳ " + decision.reason)
        row = await log_kind("transcribe", duration=media.duration, detail="pending")

    max_seconds = None if is_owner or not s.max_audio_seconds else float(s.max_audio_seconds)
    too_long = UserFacingError(f"⏳ Audio is too long; the limit is {fmt_duration(max_seconds or 0)}.")
    workdir = Path(tempfile.mkdtemp(prefix="asr-", dir=svc.tmp_dir))
    audio: Audio | None = None
    try:
        try:
            tg_file = await bot.get_file(media.file_id)
            src = workdir / ("input" + _suffix(media, tg_file.file_path))
            await tg_file.download_to_drive(src)
        except TelegramError as e:
            await svc.db.update_interaction(row, kind="error", detail=f"download: {e}")
            raise UserFacingError(f"Could not download the file from Telegram: {e}") from e

        duration = await asyncio.to_thread(probe_duration, src) or media.duration
        if max_seconds and duration and duration > max_seconds + 1:
            await svc.db.update_interaction(row, kind="limited", duration=duration, detail="too long")
            raise too_long
        try:
            # Decoding is capped too: some containers don't report a duration.
            cap = max_seconds + 2 if max_seconds else None
            audio = await asyncio.to_thread(Audio.from_file, src, workdir, cap)
        except AudioError as e:
            await svc.db.update_interaction(row, kind="error", duration=duration, detail=str(e))
            raise UserFacingError("🙉 " + str(e)) from e
        if max_seconds and audio.duration > max_seconds + 1:
            await svc.db.update_interaction(row, kind="limited", duration=audio.duration, detail="too long")
            raise too_long

        plan = Plan(
            engine_choice=user.engine if user.engine in ("auto", *svc.engines.engines) else s.default_engine,
            detector=s.language_detector,
            candidates=list(s.candidate_languages),
            min_prob=float(s.min_language_prob),
        )

        async def on_queued(ahead: int) -> None:
            if status:
                await status(f"⏳ Queued ({ahead} ahead of you)…")

        started = time.time()
        try:
            result = await svc.engines.transcribe(audio, plan, on_queued=on_queued)
        except TranscriptionError as e:
            await svc.db.update_interaction(row, kind="error", duration=audio.duration, detail=str(e))
            raise UserFacingError("⚠️ " + str(e)) from e
        except BaseException:
            await svc.db.update_interaction(row, kind="error", duration=audio.duration, detail="crashed")
            raise
        await svc.db.update_interaction(
            row, duration=audio.duration, language=result.language, engine=result.engine,
            latency=time.time() - started,
            detail=result.transcript.text if svc.config.store_transcripts else None,
        )
        return result
    finally:
        if audio is not None:
            audio.cleanup()
        shutil.rmtree(workdir, ignore_errors=True)


def _suffix(media: Media, file_path: str | None) -> str:
    for name in (media.file_name, file_path):
        if name and "." in name.rsplit("/", 1)[-1]:
            return "." + name.rsplit(".", 1)[-1][:8]
    return {"voice": ".ogg", "video_note": ".mp4", "video": ".mp4"}.get(media.kind, ".bin")


# ---- delivery ----------------------------------------------------------------------------


async def deliver(
    bot: Bot,
    result: Result,
    *,
    chat_id: int,
    placeholder: Message | None = None,
    reply_to: int | None = None,
    business_connection_id: str | None = None,
) -> None:
    """Send a transcript: edit the placeholder with the first part, send the rest as new
    messages, or a .txt file when it is very long."""
    chunks = fmt.render(result)
    kwargs = {"business_connection_id": business_connection_id} if business_connection_id else {}
    if not chunks:
        data = fmt.as_file_text(result).encode()
        caption = f"<i>{html.escape(fmt.header(result))}</i>\n📄 Long transcript attached."
        doc = InputFile(io.BytesIO(data), filename="transcript.txt")
        if placeholder is not None:
            await _edit_or_send(bot, placeholder, caption, chat_id, reply_to, kwargs)
            await bot.send_document(chat_id, doc, reply_to_message_id=reply_to, allow_sending_without_reply=True,
                                    **kwargs)
        else:
            await bot.send_document(chat_id, doc, caption=caption, parse_mode=ParseMode.HTML,
                                    reply_to_message_id=reply_to, allow_sending_without_reply=True, **kwargs)
        return
    first, rest = chunks[0], chunks[1:]
    if placeholder is not None:
        await _edit_or_send(bot, placeholder, first, chat_id, reply_to, kwargs)
    else:
        await bot.send_message(chat_id, first, parse_mode=ParseMode.HTML, reply_to_message_id=reply_to,
                               allow_sending_without_reply=True, **kwargs)
    thread = placeholder.message_thread_id if placeholder is not None and placeholder.is_topic_message else None
    for chunk in rest:
        await bot.send_message(chat_id, chunk, parse_mode=ParseMode.HTML, reply_to_message_id=reply_to,
                               allow_sending_without_reply=True, message_thread_id=thread, **kwargs)


async def _edit_or_send(bot: Bot, placeholder: Message, text: str, chat_id: int, reply_to: int | None,
                        kwargs: dict) -> None:
    try:
        await placeholder.edit_text(text, parse_mode=ParseMode.HTML)
    except TelegramError:
        await bot.send_message(chat_id, text, parse_mode=ParseMode.HTML, reply_to_message_id=reply_to,
                               allow_sending_without_reply=True, **kwargs)
