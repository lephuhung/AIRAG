"""
Invite link consumption — shared, race-safe use-count claiming.
=================================================================
Both the public registration flow (`POST /auth/register`) and the
authenticated join flow (`POST /tenants/invite/{token}/accept`) need to claim
exactly one "use" of an invite link. Doing a Python-side
`use_count < max_uses` check followed by `use_count += 1` is NOT atomic: two
concurrent redemptions of a `max_uses=1` link can both pass the check and both
increment, blowing past the cap. This module centralises the atomic claim.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import update, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.invite_token import InviteToken


async def consume_invite_use(db: AsyncSession, invite: InviteToken) -> bool:
    """Atomically claim one use of an invite link.

    Issues a single conditional UPDATE that only matches while the invite is
    still active, unexpired, and below its ``max_uses`` cap. Postgres re-evaluates
    the WHERE clause against the latest committed row under READ COMMITTED, so a
    concurrent redemption blocks on the row lock and then correctly fails once the
    cap is reached — closing the read-check-then-increment race.

    Returns ``True`` if a use was claimed (caller may proceed), ``False`` if the
    invite is exhausted / expired / revoked.

    Does NOT commit: the caller commits within the same transaction that creates
    the tenant membership, so if that fails the claimed use rolls back too. After
    this call the passed-in ``invite`` ORM instance's ``use_count`` is stale — do
    not read it; re-fetch if an up-to-date value is needed.
    """
    stmt = (
        update(InviteToken)
        .where(
            InviteToken.id == invite.id,
            InviteToken.is_active.is_(True),
            InviteToken.expires_at > datetime.utcnow(),
            or_(
                InviteToken.max_uses.is_(None),
                InviteToken.use_count < InviteToken.max_uses,
            ),
        )
        .values(use_count=InviteToken.use_count + 1)
    )
    result = await db.execute(stmt)
    return (result.rowcount or 0) > 0
