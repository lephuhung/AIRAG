"""
Security utilities — JWT token creation/validation and password hashing.
"""
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from jose import jwt, JWTError
import bcrypt

from app.core.config import settings

# Prefix for third-party API keys (NexusRAG Key) so they're recognisable in logs.
API_KEY_PREFIX = "nrk_"


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


def create_access_token(user_id: uuid.UUID, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token."""
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode = {
        "sub": str(user_id),
        "type": "access",
        "exp": expire,
    }
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(user_id: uuid.UUID, expires_delta: timedelta | None = None) -> str:
    """Create a JWT refresh token."""
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    )
    to_encode = {
        "sub": str(user_id),
        "type": "refresh",
        "exp": expire,
    }
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token. Returns payload dict or raises JWTError."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        raise


def hash_api_key(plaintext: str) -> str:
    """Return the sha256 hex digest used to store/look up an API key."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str, str]:
    """Mint a new API key.

    Returns (plaintext, key_hash, prefix). The plaintext is shown to the user
    exactly once; only key_hash is persisted. `prefix` is a short, safe-to-store
    fragment for display in the UI.
    """
    plaintext = API_KEY_PREFIX + secrets.token_urlsafe(32)
    return plaintext, hash_api_key(plaintext), plaintext[: len(API_KEY_PREFIX) + 6]


def generate_link_code() -> str:
    """Short, unambiguous one-time code for linking a Telegram chat (no 0/O/1/I)."""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(8))
