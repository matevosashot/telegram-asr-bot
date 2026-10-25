"""Who may use the bot: the single owner from the config, plus a whitelist kept in the DB."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .db import Database, UserRow, normalize_username


class Role(enum.Enum):
    OWNER = "owner"
    ALLOWED = "allowed"
    UNKNOWN = "unknown"
    BLOCKED = "blocked"

    @property
    def can_transcribe(self) -> bool:
        return self in (Role.OWNER, Role.ALLOWED)


@dataclass
class AccessResult:
    role: Role
    user: UserRow
    newly_approved: bool  # user was pre-authorized by username and is seen for the first time


class AccessControl:
    def __init__(self, db: Database, owner_id: int):
        self.db = db
        self.owner_id = owner_id

    def is_owner(self, user_id: int | None) -> bool:
        return user_id == self.owner_id

    async def check(
        self, user_id: int, username: str | None, first_name: str | None, last_name: str | None
    ) -> AccessResult:
        user, approved = await self.db.touch_user(user_id, username, first_name, last_name)
        if user_id == self.owner_id:
            role = Role.OWNER
        else:
            role = Role(user.status)
        return AccessResult(role=role, user=user, newly_approved=approved)

    async def lookup(self, user_id: int) -> AccessResult | None:
        """Access of a user we have seen before, without recording a new contact."""
        user = await self.db.get_user(user_id)
        if user is None:
            return None
        role = Role.OWNER if user_id == self.owner_id else Role(user.status)
        return AccessResult(role=role, user=user, newly_approved=False)

    async def allow(self, target: str | int) -> tuple[str, UserRow | None]:
        """Whitelist a user by numeric id or @username.

        Returns ("allowed", user) when the user is known (or given by id), or
        ("pending", None) when only the username is known: the user is approved
        automatically the first time they write to the bot.
        """
        user_id = _as_id(target)
        if user_id is not None:
            await self.db.set_status(user_id, "allowed")
            return "allowed", await self.db.get_user(user_id)
        user = await self.db.find_user_by_username(str(target))
        if user is not None:
            await self.db.set_status(user.id, "allowed")
            return "allowed", await self.db.get_user(user.id)
        await self.db.add_pending_username(str(target))
        return "pending", None

    async def deny(self, target: str | int) -> tuple[str, UserRow | None]:
        """Remove a user from the whitelist (back to 'unknown'). Also drops a pending username."""
        user_id = _as_id(target)
        user = await self.db.get_user(user_id) if user_id is not None else await self.db.find_user_by_username(str(target))
        removed_pending = False
        if user_id is None:
            removed_pending = await self.db.remove_pending_username(str(target))
        if user is not None:
            await self.db.set_status(user.id, "unknown")
            return "removed", await self.db.get_user(user.id)
        return ("pending_removed" if removed_pending else "not_found"), None

    async def block(self, target: str | int) -> tuple[str, UserRow | None]:
        user_id = _as_id(target)
        user = await self.db.get_user(user_id) if user_id is not None else await self.db.find_user_by_username(str(target))
        if user_id == self.owner_id:
            return "owner", user
        if user_id is not None and user is None:
            await self.db.set_status(user_id, "blocked")
            return "blocked", await self.db.get_user(user_id)
        if user is None:
            await self.db.remove_pending_username(str(target))
            return "not_found", None
        if user.id == self.owner_id:
            return "owner", user
        await self.db.set_status(user.id, "blocked")
        return "blocked", await self.db.get_user(user.id)


def _as_id(target: str | int) -> int | None:
    if isinstance(target, int):
        return target
    t = target.strip()
    if t.lstrip("-").isdigit():
        return int(t)
    return None


def parse_target(text: str) -> str | int | None:
    t = text.strip()
    if not t:
        return None
    if t.lstrip("-").isdigit():
        return int(t)
    name = normalize_username(t)
    if not name or not all(c.isalnum() or c == "_" for c in name):
        return None
    return "@" + name
