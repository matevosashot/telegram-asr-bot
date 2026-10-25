"""Handlers for regular users: private chats, /t replies in groups, user commands."""

from __future__ import annotations

import asyncio
import contextlib
import html
import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ChatAction, ChatType, ParseMode
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from ..auth import Role
from ..limits import fmt_duration
from .services import Services, services
from .transcribe import UserFacingError, check_access, deliver, extract_media, transcribe_media

log = logging.getLogger(__name__)

PRIVATE_MSG = ("🔒 This bot is private. I've let the owner know you'd like access; "
               "you'll be able to use it once they approve you.")


def help_text(svc: Services, role: Role) -> str:
    me = f"@{svc.bot_username}" if svc.bot_username else "the bot"
    text = (
        "🎙 <b>Voice → text</b>\n\n"
        "• Send or forward me a voice message, audio file (mp3, m4a, ogg…), round video or video "
        "and I'll reply with the transcript. The language is detected automatically.\n"
        f"• <b>Groups</b>: add {me} to the group and reply to a voice message with /t.\n"
        f"• <b>Any chat</b> (Guest Mode): reply to a voice message and mention {me}.\n\n"
        "Commands: /engine – choose recognition engine · /usage – your limits · /whoami"
    )
    if role == Role.OWNER:
        text += ("\n\n<b>Owner</b>: /admin – settings & users menu · /allow @user|id · /deny @user|id · "
                 "/block @user|id · /users · /recent · /stats · /languages hy ru en · /unload")
    return text


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    svc = services(context)
    msg = update.effective_message
    res = await check_access(svc, context.bot, update.effective_user, source="command",
                             chat_id=msg.chat_id, chat_type=msg.chat.type)
    if res is None or res.role == Role.BLOCKED:
        return
    if not res.role.can_transcribe:
        await msg.reply_text(PRIVATE_MSG)
        return
    await msg.reply_text(help_text(svc, res.role), parse_mode=ParseMode.HTML)


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    svc = services(context)
    u = update.effective_user
    res = await check_access(svc, context.bot, u, source="command")
    if res is None or res.role == Role.BLOCKED:
        return
    await update.effective_message.reply_text(
        f"id: <code>{u.id}</code>\nusername: @{html.escape(u.username or '-')}\nrole: {res.role.value}",
        parse_mode=ParseMode.HTML)


async def usage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    svc = services(context)
    msg = update.effective_message
    res = await check_access(svc, context.bot, update.effective_user, source="command")
    if res is None or not res.role.can_transcribe:
        if res and res.role == Role.UNKNOWN:
            await msg.reply_text(PRIVATE_MSG)
        return
    s = svc.settings
    if res.role == Role.OWNER:
        await msg.reply_text("👑 You're the owner: no limits apply.")
        return
    u = await svc.limiter.usage(res.user.id, s.per_hour, s.per_day)
    lim = lambda n: "∞" if n <= 0 else str(n)  # noqa: E731
    await msg.reply_text(
        f"Requests in the last hour: {u.last_hour}/{lim(s.per_hour)}\n"
        f"Requests in the last 24h: {u.last_day}/{lim(s.per_day)}\n"
        f"Max audio length: {fmt_duration(s.max_audio_seconds) if s.max_audio_seconds else '∞'}")


# ---- personal engine choice ----------------------------------------------------------------


def engine_keyboard(svc: Services, current: str | None) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(("✅ " if current is None else "") + f"Default ({svc.settings.default_engine})",
                                  callback_data="usr:eng:")]]
    for name in ["auto", *svc.engines.engines]:
        rows.append([InlineKeyboardButton(("✅ " if current == name else "") + _engine_label(svc, name),
                                          callback_data=f"usr:eng:{name}")])
    return InlineKeyboardMarkup(rows)


def _engine_label(svc: Services, name: str) -> str:
    if name == "auto":
        return "auto (pick by language)"
    eng = svc.engines.engines[name]
    return f"{name} – {eng.display_name}"


async def engine_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    svc = services(context)
    res = await check_access(svc, context.bot, update.effective_user, source="command")
    if res is None or not res.role.can_transcribe:
        if res and res.role == Role.UNKNOWN:
            await update.effective_message.reply_text(PRIVATE_MSG)
        return
    await update.effective_message.reply_text(
        "Choose the recognition engine for your messages:",
        reply_markup=engine_keyboard(svc, res.user.engine))


async def engine_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    svc = services(context)
    q = update.callback_query
    res = await check_access(svc, context.bot, q.from_user, source="command")
    if res is None or not res.role.can_transcribe:
        await q.answer("Not allowed", show_alert=True)
        return
    choice = q.data.split(":", 2)[2] or None
    if choice is not None and choice != "auto" and choice not in svc.engines.engines:
        await q.answer("Unknown engine", show_alert=True)
        return
    await svc.db.set_user_engine(res.user.id, choice)
    await q.answer(f"Engine: {choice or 'default'}")
    with contextlib.suppress(TelegramError):
        await q.edit_message_reply_markup(engine_keyboard(svc, choice))


# ---- transcription entry points ------------------------------------------------------------


class _Typing:
    """Keep the 'typing…' indicator alive while working."""

    def __init__(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int):
        self.context, self.chat_id, self.task = context, chat_id, None

    async def _loop(self) -> None:
        while True:
            with contextlib.suppress(TelegramError):
                await self.context.bot.send_chat_action(self.chat_id, ChatAction.TYPING)
            await asyncio.sleep(4.5)

    async def __aenter__(self):
        self.task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc):
        self.task.cancel()


async def _transcribe_and_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, source_msg: Message,
                                reply_to: Message, source: str, res) -> None:
    svc = services(context)
    media = extract_media(source_msg)
    chat = reply_to.chat
    placeholder = await reply_to.reply_text("🎧 Transcribing…", do_quote=True)

    async def status(text: str) -> None:
        with contextlib.suppress(TelegramError):
            await placeholder.edit_text(text)

    try:
        async with _Typing(context, chat.id):
            result = await transcribe_media(svc, context.bot, media, access=res, chat_id=chat.id,
                                            chat_type=chat.type, source=source, status=status)
    except UserFacingError as e:
        await status(str(e))
        return
    except Exception:
        log.exception("transcription failed")
        await status("⚠️ Something went wrong while transcribing. The owner can check the logs.")
        return
    await deliver(context.bot, result, chat_id=chat.id, placeholder=placeholder, reply_to=reply_to.message_id)


async def private_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Anything sent to the bot in a private chat: media (sent or forwarded) is transcribed;
    a message replying to media transcribes that media; anything else gets a hint."""
    svc = services(context)
    msg = update.effective_message
    res = await check_access(svc, context.bot, update.effective_user, source="private",
                             chat_id=msg.chat_id, chat_type=msg.chat.type)
    if res is None or res.role == Role.BLOCKED:
        return
    if not res.role.can_transcribe:
        await msg.reply_text(PRIVATE_MSG)
        return
    if extract_media(msg):
        await _transcribe_and_reply(update, context, msg, msg, "private", res)
    elif extract_media(msg.reply_to_message):
        await _transcribe_and_reply(update, context, msg.reply_to_message, msg, "private", res)
    else:
        await msg.reply_text("Send or forward me a voice message or an audio file and I'll transcribe it. /help")


async def t_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/t as a reply to a voice message (groups, or private)."""
    svc = services(context)
    msg = update.effective_message
    chat = msg.chat
    is_group = chat.type in (ChatType.GROUP, ChatType.SUPERGROUP)
    if is_group and not svc.settings.groups:
        return
    res = await check_access(svc, context.bot, update.effective_user, source="group" if is_group else "private",
                             chat_id=chat.id, chat_type=chat.type)
    if res is None or res.role == Role.BLOCKED:
        return
    if not res.role.can_transcribe:
        await msg.reply_text("🔒 You're not allowed to use this bot." if is_group else PRIVATE_MSG)
        return
    target = msg.reply_to_message
    if extract_media(target) is None:
        await msg.reply_text("Reply to a voice message (or audio/video) with /t to transcribe it.")
        return
    await _transcribe_and_reply(update, context, target, target, "group" if is_group else "private", res)
