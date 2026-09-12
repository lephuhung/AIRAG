"""Canonical source identity and build-profile resolution — Phase 1C.

The identity and profile helpers in this module are the **only**
authoritative derivation of an ingestion event's identity. Every
trigger path — S3 webhook, direct upload, chat upload, reindex,
queue retry, ``/confirm`` callback — MUST funnel through these
functions so that two callers that observed the same storage event
always converge on the same identity.

Four canonicalization rules (every trigger applies them identically):

1. **S3 event keys are form-encoded, storage keys are not.**
   ``object_key_from_s3_event`` form-decodes exactly once
   (``+`` → space, ``%XX`` → UTF-8 byte) and then normalizes.
   Every other trigger passes an already-decoded storage/list key
   through ``normalize_object_key``. Never decode twice; never
   reconstruct the key from a user filename.

2. **The webhook is metadata-only.** A webhook receives a HEAD for
   ``bucket``/``key``/``versionId``/``etag``/``size`` from MinIO
   and writes ``source_arrivals.arrival_identity``. It never
   reads the object body and never creates a revision.

3. **ETag is a version selector, not a content identity.**
   ``versionId`` wins when storage versioning is on; otherwise
   the lowercased etag is a best-effort version surrogate. A
   multipart etag ``<md5>-<n>`` is kept verbatim and is still NOT
   a content hash. ``content_sha256`` — streamed by ``/confirm``,
   direct upload, chat upload, and reindex — is the ONLY content
   discriminator and is always part of ``compute_source_object_identity``.

4. **Only ``compute_source_object_identity`` is an attempt key.**
   ``arrival_identity`` deliberately omits ``sha256`` and exists
   only to match a webhook to its ``/confirm``. An arrival-key
   collision from an identical ``etag+size`` (versioning off) is
   acceptable: it groups one ingest event, while differing bytes
   still yield a different attempt identity through ``sha256``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal
from urllib.parse import unquote_plus


# ---------------------------------------------------------------------------
# Public constants / enums
# ---------------------------------------------------------------------------


SOURCE_SCHEME: str = "s3v1"


class RevisionBuildProfile(str, Enum):
    """Build profiles that the v2 pipeline can apply to a revision.

    Each profile imposes a different set of required artifacts before
    ``verify_draft`` can transition the revision to ``verified``. The
    profile is determined at allocation time and is immutable for the
    lifetime of the revision (a retry allocates a NEW revision with
    ``retry_of_revision_id`` pointing at the prior terminal revision;
    the new revision may have a different profile if the trigger
    changed).
    """

    FULL = "FULL"
    CHAT_UPLOAD = "CHAT_UPLOAD"
    PARSE_ONLY = "PARSE_ONLY"


@dataclass(frozen=True)
class IngestFlags:
    """Per-call flags that override the prefix-based profile default.

    ``parse_only`` always wins — even over the ``chat_file_`` prefix —
    because it represents an explicit operator action (reindex,
    parse-only debug). All other profile resolution is purely
    deterministic from the canonical object key.
    """

    parse_only: bool = False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InvalidSourceObjectKey(ValueError):
    """The supplied object key is empty, absolute, escapes the bucket,
    or otherwise cannot be safely normalized."""


class MissingObjectVersion(ValueError):
    """A version selector (``versionId`` or ``etag``) is required to
    derive an identity — neither was supplied. This is a hard error:
    a webhook/confirm with no version info cannot be safely keyed."""


# ---------------------------------------------------------------------------
# Key normalization
# ---------------------------------------------------------------------------


def normalize_object_key(raw_key: str) -> str:
    """Normalize a key that is already decoded (storage/list API or
    parsed request).

    Rules:
    - Strip exactly ONE leading ``/`` (storage APIs sometimes prefix
      keys with ``/``).
    - Reject empty keys (``""``).
    - Reject keys that escape the bucket via ``..`` segments.
    - Reject keys that are still absolute (``/foo`` is fine — the
      single ``/`` is the leading-slash case; ``///foo`` becomes
      ``//foo`` after one strip, which is invalid by the same rule).
    - NO case folding (storage keys are case-sensitive).
    - NO URL decoding (the caller is responsible for form-decoding
      S3 event keys via :func:`object_key_from_s3_event` first).

    :raises InvalidSourceObjectKey: if the key fails any of the above
      invariants.
    """
    if not isinstance(raw_key, str):
        raise InvalidSourceObjectKey(f"key must be a string, got {type(raw_key).__name__}")
    key = raw_key[1:] if raw_key.startswith("/") else raw_key
    if not key:
        raise InvalidSourceObjectKey("key is empty after normalization")
    if key.startswith("/") or key.startswith(".."):
        # Either still-absolute (multiple leading slashes) or escapes
        # the bucket prefix.
        raise InvalidSourceObjectKey(f"key is invalid after normalization: {key!r}")
    # Reject ``..`` segments anywhere — a key like ``dir/../escape``
    # would alias another valid key after path resolution.
    for segment in key.split("/"):
        if segment == "..":
            raise InvalidSourceObjectKey(f"key escapes bucket via '..': {raw_key!r}")
    # ``|`` is the canonical identity field separator; a key containing it
    # would corrupt the positional identity parse (and permanently block
    # ingestion of that object).
    if "|" in key:
        raise InvalidSourceObjectKey(f"key contains reserved '|': {raw_key!r}")
    return key


def object_key_from_s3_event(event_key: str) -> str:
    """Decode an S3/MinIO ``ObjectCreated`` event key with FORM-decoding
    semantics, then normalize.

    S3 webhook payloads are application/x-www-form-urlencoded: spaces
    arrive as ``+`` and non-ASCII characters are UTF-8 percent-encoded
    (``%C3%A9`` for ``é``). We decode exactly ONCE with form semantics
    (so ``+`` becomes a space), then normalize. Double-decoding would
    turn ``%25`` (a literal percent) into ``%`` and then back into a
    bogus decode on the second pass — silently changing the identity.

    :raises InvalidSourceObjectKey: if the decoded key fails
      :func:`normalize_object_key` invariants.
    """
    if not isinstance(event_key, str):
        raise InvalidSourceObjectKey(
            f"event_key must be a string, got {type(event_key).__name__}"
        )
    # form-decode: ``+`` → space, ``%XX`` → UTF-8 byte.
    decoded = unquote_plus(event_key, encoding="utf-8", errors="strict")
    return normalize_object_key(decoded)


# ---------------------------------------------------------------------------
# Version-token derivation
# ---------------------------------------------------------------------------


def _version_token(version_id: str | None, etag: str | None) -> str:
    """Derive a version-SELECTOR token.

    - ``version_id`` wins when storage versioning is on (MinIO sets
      ``"v-<uuid>"`` on every write). It is a true version identifier.
    - Otherwise the lowercased etag is a best-effort version surrogate.
      Multipart-upload etags (``<md5>-<n>``) are preserved verbatim and
      are STILL only a version selector — NOT a content hash.
    - If neither is present, raise :class:`MissingObjectVersion`. We
      never fall back to ``size`` alone because two writes of the same
      byte length would otherwise collide.

    :raises MissingObjectVersion: when both ``version_id`` and ``etag``
      are ``None``.
    """
    if version_id:
        return f"version:{version_id}"
    if etag:
        # Multipart etags are kept verbatim (lowercased). The MD5 prefix
        # in a multipart etag is a part-MD5, not a content hash.
        return f"etag:{etag.lower()}"
    raise MissingObjectVersion(
        "either version_id or etag is required to derive an identity"
    )


# ---------------------------------------------------------------------------
# Identity derivation
# ---------------------------------------------------------------------------


def compute_source_object_identity(
    bucket: str,
    object_key: str,
    version_id: str | None,
    etag: str | None,
    size_bytes: int,
    content_sha256: str,
) -> str:
    """Derive the canonical attempt identity for a storage object.

    This is the ONLY string that should be persisted to
    ``revision_ingestion_attempts.source_*`` (well, broken out into
    components). Two storage objects that share a version selector
    (e.g., both versioning-off re-uploads of the same byte stream)
    still separate when their ``content_sha256`` differs.

    The canonical form is::

        {SOURCE_SCHEME}|{bucket}|{normalize_object_key(object_key)}|{_version_token(version_id, etag)}|{size_bytes}|{content_sha256}

    No component is optional. ``bucket`` is treated as opaque; it is
    lowercased by convention but not enforced.
    """
    key = normalize_object_key(object_key)
    return (
        f"{SOURCE_SCHEME}"
        f"|{bucket}"
        f"|{key}"
        f"|{_version_token(version_id, etag)}"
        f"|{size_bytes}"
        f"|{content_sha256}"
    )


def arrival_identity(
    bucket: str,
    object_key: str,
    version_id: str | None,
    etag: str | None,
    size_bytes: int,
) -> str:
    """Derive the canonical webhook-stage identity.

    This key MATCHES a webhook to its ``/confirm`` callback. It
    deliberately OMITS ``sha256`` so that a webhook (which only knows
    the object's metadata) can produce the same key that ``/confirm``
    (which streams the body) would derive BEFORE the streamed hash is
    known.

    An arrival-key collision (same ``etag+size`` for versioning-off
    re-uploads) is acceptable: it groups one ingest event. Differing
    bytes still yield a different attempt identity via :func:
    `compute_source_object_identity`.
    """
    key = normalize_object_key(object_key)
    return (
        f"{SOURCE_SCHEME}"
        f"|{bucket}"
        f"|{key}"
        f"|{_version_token(version_id, etag)}"
        f"|{size_bytes}"
    )


# ---------------------------------------------------------------------------
# Build-profile resolution
# ---------------------------------------------------------------------------


# Re-export ``RevisionBuildProfile`` and ``IngestFlags`` for convenience.
# They live at module scope above; the alias block below documents
# their public-API status.
__all__ = [
    "SOURCE_SCHEME",
    "RevisionBuildProfile",
    "IngestFlags",
    "InvalidSourceObjectKey",
    "MissingObjectVersion",
    "normalize_object_key",
    "object_key_from_s3_event",
    "_version_token",
    "compute_source_object_identity",
    "arrival_identity",
    "resolve_build_profile",
]


def resolve_build_profile(
    object_key: str, flags: IngestFlags
) -> RevisionBuildProfile:
    """Determine the canonical build profile for an object key.

    Rules (applied identically by every caller — so the profile is
    deterministic from the object alone):

    1. ``flags.parse_only`` ALWAYS wins — even over the ``chat_file_``
       prefix. ``parse_only`` is an explicit operator/reindex action.
    2. ``chat_file_*`` keys → ``CHAT_UPLOAD`` (skip caption/KG after
       parse; vectors are still REQUIRED — ``embed_skipped`` stays false,
       while ``captions_skipped`` and ``kg_skipped`` must be true).
    3. Anything else → ``FULL``.

    :raises InvalidSourceObjectKey: if the key fails normalization.
    """
    if flags.parse_only:
        return RevisionBuildProfile.PARSE_ONLY
    key = normalize_object_key(object_key)
    if key.startswith("chat_file_"):
        return RevisionBuildProfile.CHAT_UPLOAD
    return RevisionBuildProfile.FULL
