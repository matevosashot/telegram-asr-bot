"""Builds the python-telegram-bot Application."""

from __future__ import annotations

import logging

from telegram import (
    BotCommand,
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeChat,
    Update,
)
from telegram.error import TelegramError
from telegram.request import BaseRequest
from telegram.ext import (
    AIORateLimiter,
    Application,
    ApplicationBuilder,
    BusinessConnectionHandler,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters as F,
)

from ..auth import AccessControl
from ..config import Config
from ..db import Database
from ..engine import EngineManager
from ..limits import RateLimiter
from ..settings import Settings
from . import admin, business, guest, user
from .services import KEY, Services

log = logging.getLogger(__name__)

USER_COMMANDS = [
    BotCommand("start", "How to use the bot"),
    BotCommand("t", "Transcribe the voice message you reply to"),
    BotCommand("engine", "Choose the recognition engine"),
    BotCommand("usage", "Your usage and limits"),
    BotCommand("whoami", "Your Telegram id"),
    BotCommand("help", "Help"),
]
OWNER_COMMANDS = USER_COMMANDS + [
    BotCommand("admin", "Settings and users menu"),
    BotCommand("allow", "Allow @username or id"),
    BotCommand("deny", "Remove @username or id from the whitelist"),
    BotCommand("block", "Block @username or id"),
    BotCommand("users", "List users"),
    BotCommand("recent", "Recent activity"),
    BotCommand("stats", "Usage statistics"),
    BotCommand("languages", "Candidate languages, e.g. /languages hy ru en"),
    BotCommand("unload", "Unload models from GPU memory"),
]
GROUP_COMMANDS = [BotCommand("t", "Reply to a voice message to transcribe it")]


def build_application(
    config: Config,
    token: str,
    engines: EngineManager | None = None,
    request: BaseRequest | None = None,
) -> Application:
    builder: ApplicationBuilder = (
        Application.builder()
        .token(token)
        .concurrent_updates(True)
        .rate_limiter(AIORateLimiter(max_retries=2))
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
    )
    if request is not None:  # tests inject a fake transport
        builder = builder.request(request).get_updates_request(request)
    else:
        builder = builder.read_timeout(60).write_timeout(60).media_write_timeout(120)
    if config.telegram.local_api_url:
        base = config.telegram.local_api_url.rstrip("/")
        builder = builder.base_url(f"{base}/bot").base_file_url(f"{base}/file/bot").local_mode(True)
    app = builder.build()

    db = Database(config.db_path)
    tmp_dir = config.data_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    app.bot_data[KEY] = Services(
        config=config,
        db=db,
        settings=Settings(config.defaults, db),
        access=AccessControl(db, config.telegram.owner_id),
        limiter=RateLimiter(db),
        engines=engines or EngineManager(config),
        tmp_dir=tmp_dir,
    )
    register_handlers(app)
    return app


def register_handlers(app: Application) -> None:
    msg = F.UpdateType.MESSAGE
    private = msg & F.ChatType.PRIVATE
    owner_cmd = lambda name, cb: CommandHandler(name, cb, filters=private)  # noqa: E731

    # Updates that are not ordinary messages.
    app.add_handler(MessageHandler(F.UpdateType.GUEST_MESSAGE, guest.guest_message, block=False))
    app.add_handler(BusinessConnectionHandler(business.business_connection))
    app.add_handler(MessageHandler(F.UpdateType.BUSINESS_MESSAGE, business.business_message, block=False))

    # User commands.
    app.add_handler(CommandHandler(["start", "help"], user.start, filters=msg))
    app.add_handler(CommandHandler("whoami", user.whoami, filters=msg))
    app.add_handler(CommandHandler("usage", user.usage, filters=msg))
    app.add_handler(CommandHandler("engine", user.engine_cmd, filters=msg))
    app.add_handler(CommandHandler(["t", "transcribe"], user.t_command, filters=msg, block=False))

    # Owner commands (private chat only).
    app.add_handler(owner_cmd(["admin", "settings"], admin.admin_cmd))
    app.add_handler(owner_cmd("allow", admin.allow_cmd))
    app.add_handler(owner_cmd("deny", admin.deny_cmd))
    app.add_handler(owner_cmd("block", admin.block_cmd))
    app.add_handler(owner_cmd("users", admin.users_cmd))
    app.add_handler(owner_cmd("recent", admin.recent_cmd))
    app.add_handler(owner_cmd("stats", admin.stats_cmd))
    app.add_handler(owner_cmd("languages", admin.languages_cmd))
    app.add_handler(owner_cmd("unload", admin.unload_cmd))
    app.add_handler(MessageHandler(private & F.StatusUpdate.USERS_SHARED, admin.users_shared))

    # Inline keyboard callbacks.
    app.add_handler(CallbackQueryHandler(admin.access_callback, pattern=r"^acc:"))
    app.add_handler(CallbackQueryHandler(admin.admin_callback, pattern=r"^adm:"))
    app.add_handler(CallbackQueryHandler(user.engine_callback, pattern=r"^usr:eng:"))

    # Everything else in private chats: media is transcribed, other messages get help.
    app.add_handler(MessageHandler(private & ~F.COMMAND & ~F.StatusUpdate.ALL, user.private_message, block=False))

    app.add_error_handler(_on_error)


async def _post_init(app: Application) -> None:
    svc: Services = app.bot_data[KEY]
    await svc.db.open()
    await svc.settings.load()
    me = await app.bot.get_me()
    svc.bot_username = me.username or ""
    log.info("bot @%s (id %s) starting; owner id %s; engines: %s; detectors: %s",
             me.username, me.id, svc.owner_id, ", ".join(svc.engines.engines) or "none",
             ", ".join(svc.engines.detectors) or "none")
    if not me.supports_guest_queries:
        log.info("Guest Mode is off for this bot (enable it in @BotFather to use @mentions in any chat)")
    try:
        await app.bot.set_my_commands(USER_COMMANDS, scope=BotCommandScopeAllPrivateChats())
        await app.bot.set_my_commands(GROUP_COMMANDS, scope=BotCommandScopeAllGroupChats())
        await app.bot.set_my_commands(OWNER_COMMANDS, scope=BotCommandScopeChat(svc.owner_id))
    except TelegramError as e:
        log.warning("could not set bot commands: %s", e)
    svc.engines.start_idle_unloader()
    svc.engines.start_preload()


async def _post_shutdown(app: Application) -> None:
    svc: Services = app.bot_data[KEY]
    await svc.engines.shutdown()
    await svc.db.close()


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("error while handling update %s", getattr(update, "update_id", update), exc_info=context.error)


def run(config: Config, token: str) -> None:
    app = build_application(config, token)
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=False)
