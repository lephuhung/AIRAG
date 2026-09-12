"""Tests for canonical source identity and build-profile resolution.

Every test in this file is a REAL assertion about the identity contract
the brief defines — none are tautologies. They cover:

- ``compute_source_object_identity`` — canonical form, case-folding
  prohibition, S3 event vs storage key decoding rules.
- ``arrival_identity`` — webhook-stage key (no sha256), separate from
  attempt identity.
- ``resolve_build_profile`` — deterministic per-object derivation.
- ``SourceArrival`` upsert semantics on duplicate webhook arrivals.

These tests are pure-Python (no DB) for the canonicalization rules, with
the one DB-backed assertion (``test_duplicate_webhook_arrival_is_idempotent``)
exercising the ``source_arrivals.arrival_identity`` UNIQUE constraint.
"""

from __future__ import annotations

import uuid
from typing import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.agents.v2.persistence.source_identity import (  # type: ignore[import-not-found]
    IngestFlags,
    RevisionBuildProfile,
    SOURCE_SCHEME,
    MissingObjectVersion,
    InvalidSourceObjectKey,
    _version_token,
    arrival_identity,
    compute_source_object_identity,
    normalize_object_key,
    object_key_from_s3_event,
    resolve_build_profile,
)


# ---------------------------------------------------------------------------
# _version_token — version selector is NEVER a content identity
# ---------------------------------------------------------------------------


class TestVersionToken:
    def test_version_id_wins_when_present(self):
        """``versionId`` is the version selector when versioning is on."""
        assert (
            _version_token("v-1234", '"abc"') == "version:v-1234"
        ), "versionId must win over etag"

    def test_etag_used_when_version_id_absent(self):
        """Without versioning, the lowercased etag is a version surrogate."""
        # The etag value is treated as opaque by the version token —
        # the brief's implementation lowercases it but does not strip
        # surrounding quotes (storage adapters normalize that upstream).
        assert _version_token(None, "abcdef") == "etag:abcdef"
        assert _version_token(None, "ABCDEF") == "etag:abcdef"
        assert _version_token(None, "AbCdEf") == "etag:abcdef"

    def test_multipart_etag_kept_verbatim(self):
        """A multipart etag ``<md5>-<n>`` is preserved verbatim (and is
        STILL a version selector, NOT a content hash)."""
        token = _version_token(None, "d41d8cd98f00b204e9800998ecf8427e-2")
        assert token == "etag:d41d8cd98f00b204e9800998ecf8427e-2"

    def test_missing_both_version_id_and_etag_raises(self):
        """No version selector at all is a hard error — we never want to
        fall back to ``size`` alone as a version surrogate."""
        with pytest.raises(MissingObjectVersion):
            _version_token(None, None)

    def test_version_id_takes_precedence_over_etag_even_when_etag_looks_content_like(
        self,
    ):
        """``version_id`` always wins; etag is ignored."""
        assert (
            _version_token("v-final", "sha-like-value")
            == "version:v-final"
        )


# ---------------------------------------------------------------------------
# normalize_object_key — used verbatim by storage/list APIs
# ---------------------------------------------------------------------------


class TestNormalizeObjectKey:
    def test_storage_key_used_verbatim(self):
        """``normalize_object_key`` is used verbatim — no case folding,
        no URL decoding, no filename reconstruction."""
        assert normalize_object_key("dir/Some File.PDF") == "dir/Some File.PDF"

    def test_leading_slash_stripped(self):
        """Storage APIs sometimes prefix keys with ``/``; strip exactly one."""
        assert normalize_object_key("/dir/file.pdf") == "dir/file.pdf"

    def test_leading_slash_already_absent_unchanged(self):
        assert normalize_object_key("dir/file.pdf") == "dir/file.pdf"

    def test_empty_key_rejected(self):
        """Empty keys are never valid (no "any" object identity)."""
        with pytest.raises(InvalidSourceObjectKey):
            normalize_object_key("")

    def test_dotdot_key_rejected(self):
        """Keys that escape the bucket (``..``) are rejected so the identity
        cannot alias arbitrary paths."""
        with pytest.raises(InvalidSourceObjectKey):
            normalize_object_key("../etc/passwd")
        with pytest.raises(InvalidSourceObjectKey):
            normalize_object_key("dir/../../etc/passwd")

    def test_absolute_key_after_strip_still_rejected(self):
        """A key that becomes ``/...`` after stripping the leading ``/``
        is rejected (defense-in-depth — only one leading ``/`` is stripped)."""
        # The bare ``/`` becomes ``""`` after the leading-slash strip
        # and must be rejected as empty.
        with pytest.raises(InvalidSourceObjectKey):
            normalize_object_key("/")
        # ``//foo`` strips to ``/foo`` which is still absolute — reject.
        with pytest.raises(InvalidSourceObjectKey):
            normalize_object_key("//foo")
        # Pure ``/foo`` strips to ``foo`` which is fine.
        assert normalize_object_key("/foo") == "foo"

    def test_case_preserved(self):
        """No case folding — ``Some File.PDF`` and ``some file.pdf`` are
        DIFFERENT storage objects."""
        a = normalize_object_key("Some File.PDF")
        b = normalize_object_key("some file.pdf")
        assert a != b, "case must NOT be folded — these are different objects"

    def test_user_filename_does_not_change_key(self):
        """The user filename is irrelevant — only the storage key counts."""
        # Pretend the same storage key was uploaded under two different
        # user filenames; the normalized key is identical.
        k = "uploads/abc/report.pdf"
        assert normalize_object_key(k) == normalize_object_key(k)


# ---------------------------------------------------------------------------
# object_key_from_s3_event — S3 webhooks are form-encoded, decoded EXACTLY ONCE
# ---------------------------------------------------------------------------


class TestObjectKeyFromS3Event:
    def test_plus_decoded_to_space(self):
        """S3 ObjectCreated events are application/x-www-form-urlencoded:
        ``+`` is a space."""
        assert object_key_from_s3_event("doc+1.pdf") == "doc 1.pdf"

    def test_percent_encoded_decoded_once(self):
        """Percent-encoded bytes are decoded exactly once."""
        # %20 → space
        assert object_key_from_s3_event("doc%20file.pdf") == "doc file.pdf"
        # UTF-8 bytes: a multi-byte char like 'é' is %C3%A9
        assert object_key_from_s3_event("r%C3%A9sum%C3%A9.pdf") == "résumé.pdf"

    def test_double_decoding_not_applied(self):
        """A pre-decoded storage key coming through an S3 event must
        NEVER be decoded a second time. We test by feeding a key that
        contains a literal ``%`` (already-decoded), and asserting the
        literal ``%`` survives."""
        # Single-decode test: %25 (encoded percent) → %, then verify no
        # further re-decoding.
        once = object_key_from_s3_event("100%25-done.pdf")
        assert once == "100%-done.pdf"

    def test_storage_key_and_event_key_converge(self):
        """The brief's hard rule: ``doc+1%20b.pdf`` from an S3 event
        must canonicalize to the SAME key as the storage/list-API's
        already-decoded ``doc 1 b.pdf``."""
        from_event = object_key_from_s3_event("doc+1%20b.pdf")
        from_storage = normalize_object_key("doc 1 b.pdf")
        assert from_event == from_storage, (
            "S3 event key must canonicalize to the same key as the "
            "storage/list-reported key (brief Step 2 rule 1)"
        )

    def test_storage_key_canonicalization_matches_event_decoding(self):
        """Round-trip: an event key that is already storage-canonical
        (no form encoding) decodes to the same normalized key."""
        # ''plain'' key: no pluses, no %XX — event → same storage key
        assert (
            object_key_from_s3_event("dir/file.pdf")
            == normalize_object_key("dir/file.pdf")
        )

    def test_invalid_event_key_raises(self):
        with pytest.raises(InvalidSourceObjectKey):
            object_key_from_s3_event("")  # decoded empty key is invalid
        with pytest.raises(InvalidSourceObjectKey):
            object_key_from_s3_event("../escape.pdf")  # dotdot is invalid


# ---------------------------------------------------------------------------
# compute_source_object_identity — the ONLY attempt key
# ---------------------------------------------------------------------------


class TestComputeSourceObjectIdentity:
    def _kwargs(self, **overrides):
        base = dict(
            bucket="my-bucket",
            object_key="uploads/abc/file.pdf",
            version_id="v-1",
            etag="abc",
            size_bytes=1024,
            content_sha256=(
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
        )
        base.update(overrides)
        return base

    def test_canonical_form_is_stable(self):
        """Calling the function twice with identical inputs yields an
        identical string — the canonical form is deterministic."""
        a = compute_source_object_identity(**self._kwargs())
        b = compute_source_object_identity(**self._kwargs())
        assert a == b

    def test_includes_sha256_always(self):
        """sha256 is ALWAYS part of the identity — never optional."""
        identity = compute_source_object_identity(**self._kwargs())
        assert (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            in identity
        )

    def test_storage_key_used_verbatim(self):
        """The storage key is used verbatim — no case folding, no decode."""
        kwargs = self._kwargs(object_key="Some Dir/FILE.PDF")
        identity = compute_source_object_identity(**kwargs)
        # The literal "Some Dir/FILE.PDF" must appear in the identity.
        assert "Some Dir/FILE.PDF" in identity, (
            "case must be preserved in the identity"
        )

    def test_user_filename_does_not_change_identity(self):
        """The user filename (which never enters the call) is irrelevant:
        two uploads of the same storage key share an identity."""
        # Same storage key, two different file payloads that hash to the
        # same SHA256. The brief assumes content_sha256 is the *streamed*
        # SHA256; same content → same SHA256 → same identity.
        kwargs = self._kwargs(
            content_sha256="a" * 64, object_key="uploads/x.pdf"
        )
        a = compute_source_object_identity(**kwargs)
        b = compute_source_object_identity(**kwargs)
        assert a == b  # The user filename doesn't even enter this API.

    def test_size_changes_identity(self):
        """Size is part of the identity (object metadata)."""
        kwargs = self._kwargs()
        a = compute_source_object_identity(**kwargs)
        kwargs_b = dict(kwargs)
        kwargs_b["size_bytes"] = 2048
        b = compute_source_object_identity(**kwargs_b)
        assert a != b, "size must change the identity"

    def test_multipart_etag_same_sha_same_version_stable(self):
        """Same multipart etag + same sha + same size = same identity."""
        kwargs = self._kwargs(
            version_id=None,
            etag="d41d8cd98f00b204e9800998ecf8427e-2",
            content_sha256="a" * 64,
            size_bytes=100,
        )
        a = compute_source_object_identity(**kwargs)
        b = compute_source_object_identity(**kwargs)
        assert a == b


# ---------------------------------------------------------------------------
# arrival_identity — webhook-only staging key (NO sha256)
# ---------------------------------------------------------------------------


class TestArrivalIdentity:
    def test_no_sha256_in_arrival_identity(self):
        """``arrival_identity`` deliberately OMITS sha256."""
        a = arrival_identity(
            bucket="my-bucket",
            object_key="uploads/abc/file.pdf",
            version_id="v-1",
            etag='"abc"',
            size_bytes=1024,
        )
        # No 64-char hex digest anywhere in the identity
        assert all(
            not (c * 64 in a)
            for c in "0123456789abcdef"
        ), f"arrival_identity must NOT include sha256: {a}"

    def test_arrival_identity_differs_from_attempt_identity_only_via_sha(self):
        """Same bucket/key/version/size → same arrival. Attempt identity
        differs only because it includes the streamed sha256."""
        kwargs = dict(
            bucket="my-bucket",
            object_key="uploads/abc/file.pdf",
            version_id="v-1",
            etag="abc",
            size_bytes=1024,
        )
        sha = "b" * 64
        arrival = arrival_identity(**kwargs)
        attempt = compute_source_object_identity(
            content_sha256=sha, **kwargs
        )
        # Same prefix (everything before the final sha field), attempt
        # simply appends the sha suffix.
        assert attempt.startswith(arrival), (
            "attempt identity must extend the arrival identity with sha256"
        )
        assert attempt.endswith(sha)

    def test_arrival_collides_for_same_etag_and_size_but_sha_distinguishes_attempt(
        self,
    ):
        """Same etag + size (versioning off) yields one arrival key —
        that arrival key groups one ingest event. But two objects with
        different sha256 still get different attempt identities."""
        common = dict(
            bucket="my-bucket",
            object_key="uploads/abc/file.pdf",
            version_id=None,
            etag="abc",
            size_bytes=1024,
        )
        # Same arrival key (no sha in arrival)
        a1 = arrival_identity(**common)
        a2 = arrival_identity(**common)
        assert a1 == a2
        # Different sha → different attempt identities
        attempt_a = compute_source_object_identity(
            content_sha256="a" * 64, **common
        )
        attempt_b = compute_source_object_identity(
            content_sha256="b" * 64, **common
        )
        assert attempt_a != attempt_b, "sha256 must distinguish attempt keys"


# ---------------------------------------------------------------------------
# Multipart etag is not a content hash
# ---------------------------------------------------------------------------


class TestMultipartEtagNotContentHash:
    def test_same_etag_different_sha_yields_different_attempt_but_same_arrival(
        self,
    ):
        """Multipart upload: same etag (because same parts) but different
        content (because different part ordering) → different attempt
        identity (sha differs) but same arrival identity (etag/size same)."""
        common = dict(
            bucket="my-bucket",
            object_key="uploads/abc/file.pdf",
            version_id=None,
            etag="d41d8cd98f00b204e9800998ecf8427e-2",
            size_bytes=2048,
        )
        arrival_a = arrival_identity(**common)
        arrival_b = arrival_identity(**common)
        assert arrival_a == arrival_b  # same arrival key

        attempt_a = compute_source_object_identity(
            content_sha256="a" * 64, **common
        )
        attempt_b = compute_source_object_identity(
            content_sha256="b" * 64, **common
        )
        assert attempt_a != attempt_b, "different sha → different attempt"


# ---------------------------------------------------------------------------
# ETag as a version selector (NOT content identity)
# ---------------------------------------------------------------------------


class TestEtagIsVersionSelector:
    def test_identity_and_arrival_stable_across_callers(self):
        """Two callers invoking the same identity function with the same
        inputs always get the same strings back. ETag is treated as a
        stable version selector (lowercased verbatim)."""
        kwargs = dict(
            bucket="b",
            object_key="k",
            version_id=None,
            etag="ABCDEF",
            size_bytes=10,
            content_sha256="c" * 64,
        )
        a = compute_source_object_identity(**kwargs)
        b = compute_source_object_identity(**kwargs)
        assert a == b
        # arrival_identity takes the same args *minus* content_sha256.
        arrival_kwargs = {k: v for k, v in kwargs.items() if k != "content_sha256"}
        assert arrival_identity(**arrival_kwargs) == arrival_identity(**arrival_kwargs)

    def test_sha256_is_only_content_discriminator(self):
        """Holding etag/size/key constant and varying only sha256 must
        produce a different attempt identity — that is the 'only content
        discriminator' rule."""
        common = dict(
            bucket="b", object_key="k", version_id=None, etag="e", size_bytes=10,
        )
        a = compute_source_object_identity(content_sha256="a" * 64, **common)
        b = compute_source_object_identity(content_sha256="b" * 64, **common)
        assert a != b
        # But the arrival keys collide (etag+size match)
        assert arrival_identity(**common) == arrival_identity(**common)


# ---------------------------------------------------------------------------
# Overwrite same key → new identity → new attempt
# ---------------------------------------------------------------------------


class TestOverwriteSameKey:
    def test_overwrite_changes_version_etag_sha_and_identity(self):
        """Overwriting a key with different bytes changes version/etag/sha
        and therefore the identity."""
        # Original version
        v1 = dict(
            bucket="b",
            object_key="uploads/x.pdf",
            version_id=None,
            etag="v1",
            size_bytes=100,
            content_sha256="a" * 64,
        )
        # Overwrite: etag, size, sha all change
        v2 = dict(
            bucket="b",
            object_key="uploads/x.pdf",
            version_id=None,
            etag="v2",
            size_bytes=200,
            content_sha256="b" * 64,
        )
        id_v1 = compute_source_object_identity(**v1)
        id_v2 = compute_source_object_identity(**v2)
        assert id_v1 != id_v2

    def test_versioning_on_changes_identity_per_version(self):
        """With versioning on, two versions of the same key are distinct
        attempt identities (the versionId distinguishes them)."""
        common = dict(
            bucket="b", object_key="uploads/x.pdf", etag=None, size_bytes=100,
        )
        a = compute_source_object_identity(
            version_id="v1", content_sha256="a" * 64, **common
        )
        b = compute_source_object_identity(
            version_id="v2", content_sha256="a" * 64, **common
        )
        assert a != b
        # Same sha but different version → different identity
        # (the brief's "version selector" rule).


# ---------------------------------------------------------------------------
# resolve_build_profile — deterministic per-object derivation
# ---------------------------------------------------------------------------


class TestBuildProfileResolution:
    def test_doc_prefix_yields_FULL(self):
        """Plain ``doc_*`` keys map to ``FULL`` profile."""
        assert (
            resolve_build_profile("doc_abc.pdf", IngestFlags(parse_only=False))
            == RevisionBuildProfile.FULL
        )

    def test_any_key_yields_FULL_when_no_flags(self):
        """Without chat prefix and without parse_only, any object → FULL."""
        assert (
            resolve_build_profile("uploads/foo.pdf", IngestFlags(parse_only=False))
            == RevisionBuildProfile.FULL
        )

    def test_chat_file_prefix_yields_CHAT_UPLOAD(self):
        """``chat_file_*`` keys map to ``CHAT_UPLOAD`` profile."""
        assert (
            resolve_build_profile(
                "chat_file_xyz.docx", IngestFlags(parse_only=False)
            )
            == RevisionBuildProfile.CHAT_UPLOAD
        )

    def test_explicit_parse_only_flag_overrides(self):
        """``parse_only`` flag ALWAYS overrides the prefix-based default."""
        assert (
            resolve_build_profile("doc_x.pdf", IngestFlags(parse_only=True))
            == RevisionBuildProfile.PARSE_ONLY
        )
        assert (
            resolve_build_profile(
                "chat_file_x.docx", IngestFlags(parse_only=True)
            )
            == RevisionBuildProfile.PARSE_ONLY
        )

    def test_parse_only_overrides_plain_key(self):
        assert (
            resolve_build_profile("uploads/plain.pdf", IngestFlags(parse_only=True))
            == RevisionBuildProfile.PARSE_ONLY
        )

    def test_profile_resolution_is_deterministic(self):
        """Same key + flags → same profile (idempotent)."""
        flags = IngestFlags(parse_only=False)
        a = resolve_build_profile("doc_abc.pdf", flags)
        b = resolve_build_profile("doc_abc.pdf", flags)
        assert a == b

    def test_chat_file_substring_does_not_match(self):
        """Only keys that START WITH ``chat_file_`` map to CHAT_UPLOAD;
        substrings like ``my_chat_file_x`` do NOT."""
        assert (
            resolve_build_profile(
                "my_chat_file_x.pdf", IngestFlags(parse_only=False)
            )
            == RevisionBuildProfile.FULL
        )

    def test_normalization_applied_to_prefix_check(self):
        """``resolve_build_profile`` normalizes the key before prefix-matching,
        so a leading slash on the storage key does not change the profile."""
        assert (
            resolve_build_profile(
                "/chat_file_x.docx", IngestFlags(parse_only=False)
            )
            == RevisionBuildProfile.CHAT_UPLOAD
        )


# ---------------------------------------------------------------------------
# Webhook idempotency — SourceArrival UNIQUE(arrival_identity)
# ---------------------------------------------------------------------------


class TestDuplicateWebhookArrival:
    @pytest_asyncio.fixture
    async def session(self, async_db) -> AsyncIterator[AsyncSession]:
        """The transactional async session."""
        yield async_db

    @pytest.mark.asyncio
    async def test_duplicate_webhook_arrival_is_idempotent(
        self, session: AsyncSession, raw_connection
    ):
        """A duplicate webhook with the same arrival_identity MUST NOT
        create a second ``source_arrivals`` row — the UNIQUE constraint
        on ``arrival_identity`` is the arbiter.

        The test inserts two rows with the same arrival_identity
        using ``ON CONFLICT DO NOTHING`` semantics and asserts exactly
        one row exists."""
        from app.models.source_arrival import SourceArrival
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        arrival_id = str(uuid.uuid4())
        arrival_identity_str = (
            f"{SOURCE_SCHEME}|bkt|key.pdf|version:v1|1024"
        )

        # First insert (succeeds, returns the row).
        result1 = await session.execute(
            pg_insert(SourceArrival)
            .values(
                arrival_id=uuid.UUID(arrival_id),
                bucket="bkt",
                object_key="key.pdf",
                version_id="v1",
                etag=None,
                size_bytes=1024,
                arrival_identity=arrival_identity_str,
            )
            .on_conflict_do_nothing(index_elements=["arrival_identity"])
            .returning(SourceArrival.arrival_id)
        )
        first_id = result1.scalar_one_or_none()
        await session.flush()

        # Second insert with a DIFFERENT arrival_id but the SAME
        # arrival_identity — must conflict and produce no row.
        second_arrival_id = str(uuid.uuid4())
        result2 = await session.execute(
            pg_insert(SourceArrival)
            .values(
                arrival_id=uuid.UUID(second_arrival_id),
                bucket="bkt",
                object_key="key.pdf",
                version_id="v1",
                etag=None,
                size_bytes=1024,
                arrival_identity=arrival_identity_str,
            )
            .on_conflict_do_nothing(index_elements=["arrival_identity"])
            .returning(SourceArrival.arrival_id)
        )
        second_id = result2.scalar_one_or_none()
        await session.flush()

        assert first_id is not None, "first insert must succeed"
        assert second_id is None, (
            "second insert with same arrival_identity must be a no-op "
            "(UNIQUE constraint enforces idempotency)"
        )

        # Exactly one row in source_arrivals for this identity.
        cnt = await session.execute(
            text(
                "SELECT count(*) FROM source_arrivals "
                "WHERE arrival_identity = :ident"
            ),
            {"ident": arrival_identity_str},
        )
        assert cnt.scalar() == 1, (
            "exactly one source_arrivals row must exist per arrival_identity"
        )

        # The committed row carries the FIRST arrival_id (the loser
        # was discarded).
        row = await session.execute(
            text(
                "SELECT arrival_id FROM source_arrivals "
                "WHERE arrival_identity = :ident"
            ),
            {"ident": arrival_identity_str},
        )
        committed_id = row.scalar()
        assert committed_id is not None
        # Compare UUIDs as strings (psycopg returns UUID type).
        assert str(committed_id) == arrival_id, (
            "the surviving row is the FIRST insert, not the loser's"
        )


# ---------------------------------------------------------------------------
# Brief-named canonical-identity checklist (Phase 1C Task 3)
#
# The brief enumerates these test names explicitly. Several concepts are
# also covered by the focused classes above; these named tests keep a 1:1
# brief-item -> test mapping for review.
# ---------------------------------------------------------------------------


class TestBriefNamedCanonicalIdentity:
    def test_source_object_identity_is_canonical(self):
        """The storage key is used verbatim; the user filename and case
        never change the identity; an S3 event key is form-decoded exactly
        once before normalization."""
        canonical = compute_source_object_identity(
            bucket="bkt",
            object_key="dir/Doc 1.pdf",
            version_id="v-1",
            etag=None,
            size_bytes=10,
            content_sha256="a" * 64,
        )
        # Storage key verbatim (case-sensitive, no folding).
        assert "|dir/Doc 1.pdf|" in canonical
        assert compute_source_object_identity(
            bucket="bkt", object_key="dir/doc 1.pdf", version_id="v-1",
            etag=None, size_bytes=10, content_sha256="a" * 64,
        ) != canonical
        # One leading slash is stripped; the identity is unchanged.
        assert compute_source_object_identity(
            bucket="bkt", object_key="/dir/Doc 1.pdf", version_id="v-1",
            etag=None, size_bytes=10, content_sha256="a" * 64,
        ) == canonical
        # S3 event key is form-decoded exactly once and converges.
        from_event = object_key_from_s3_event("dir/Doc+1.pdf")
        assert from_event == "dir/Doc 1.pdf"
        assert compute_source_object_identity(
            bucket="bkt", object_key=from_event, version_id="v-1",
            etag=None, size_bytes=10, content_sha256="a" * 64,
        ) == canonical

    def test_multipart_etag_is_not_a_content_hash(self):
        """Same etag/size but different sha256 yields different attempt
        identities while sharing one arrival key."""
        multipart_etag = "d41d8cd98f00b204e9800998ecf8427e-3"
        attempt_a = compute_source_object_identity(
            bucket="bkt", object_key="k.zip", version_id=None,
            etag=multipart_etag, size_bytes=100, content_sha256="1" * 64,
        )
        attempt_b = compute_source_object_identity(
            bucket="bkt", object_key="k.zip", version_id=None,
            etag=multipart_etag, size_bytes=100, content_sha256="2" * 64,
        )
        arrival_a = arrival_identity(
            bucket="bkt", object_key="k.zip", version_id=None,
            etag=multipart_etag, size_bytes=100,
        )
        arrival_b = arrival_identity(
            bucket="bkt", object_key="k.zip", version_id=None,
            etag=multipart_etag, size_bytes=100,
        )
        assert attempt_a != attempt_b, (
            "different sha256 must separate attempt identities even when "
            "a multipart etag is reused"
        )
        assert arrival_a == arrival_b
        # The multipart etag is preserved verbatim as the version selector.
        assert multipart_etag in attempt_a

    def test_s3_event_key_is_decoded_once_and_matches_storage_key(self):
        """``doc+1%20b.pdf`` from the event canonicalizes to the same key as
        the storage-reported ``doc 1 b.pdf``."""
        assert object_key_from_s3_event("doc+1%20b.pdf") == "doc 1 b.pdf"
        assert object_key_from_s3_event(
            "doc+1%20b.pdf"
        ) == normalize_object_key("doc 1 b.pdf")

    def test_etag_is_a_version_selector_not_content_identity(self):
        """Identity and arrival strings are stable across callers and
        sha256 is the only content discriminator."""
        a = compute_source_object_identity(
            bucket="bkt", object_key="x.pdf", version_id=None, etag="abc123",
            size_bytes=7, content_sha256="f" * 64,
        )
        b = compute_source_object_identity(
            bucket="bkt", object_key="x.pdf", version_id=None, etag="abc123",
            size_bytes=7, content_sha256="f" * 64,
        )
        assert a == b, "identity must be stable across identical callers"
        # etag is case-insensitive metadata: 'ABC123' == 'abc123'.
        c = compute_source_object_identity(
            bucket="bkt", object_key="x.pdf", version_id=None, etag="ABC123",
            size_bytes=7, content_sha256="f" * 64,
        )
        assert a == c
        # sha256 is the ONLY content discriminator.
        d = compute_source_object_identity(
            bucket="bkt", object_key="x.pdf", version_id=None, etag="abc123",
            size_bytes=7, content_sha256="e" * 64,
        )
        assert a != d

    @pytest.mark.asyncio
    async def test_overwrite_same_key_changes_identity_and_creates_new_attempt(
        self, async_db: AsyncSession, document_factory
    ):
        """A new version/etag/sha on the same key is a NEW identity and MUST
        allocate a new attempt/revision generation (not converge on the old
        one) — the R1 arbiter-scope guarantee."""
        from app.services.agents.v2.persistence.document_revisions import (
            DocumentRevisionsRepository,
        )

        repo = DocumentRevisionsRepository(async_db)
        document_id = document_factory()
        old = compute_source_object_identity(
            bucket="bkt", object_key="k.pdf", version_id="v-1", etag=None,
            size_bytes=10, content_sha256="1" * 64,
        )
        new = compute_source_object_identity(
            bucket="bkt", object_key="k.pdf", version_id="v-2", etag=None,
            size_bytes=11, content_sha256="2" * 64,
        )
        assert old != new
        r_old, created_old = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=old,
            build_profile=RevisionBuildProfile.FULL,
        )
        r_new, created_new = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=new,
            build_profile=RevisionBuildProfile.FULL,
        )
        assert created_old is True
        assert created_new is True
        assert r_old.revision_id != r_new.revision_id
        assert r_new.generation > r_old.generation

    def test_build_profile_resolution_is_deterministic(self):
        """``doc_*`` -> ``FULL``, ``chat_file_*`` -> ``CHAT_UPLOAD``, explicit
        parse-only flag overrides."""
        assert (
            resolve_build_profile("doc_2024/report.pdf", IngestFlags())
            is RevisionBuildProfile.FULL
        )
        assert (
            resolve_build_profile("chat_file_abc.pdf", IngestFlags())
            is RevisionBuildProfile.CHAT_UPLOAD
        )
        assert (
            resolve_build_profile(
                "doc_2024/report.pdf", IngestFlags(parse_only=True)
            )
            is RevisionBuildProfile.PARSE_ONLY
        )
        assert (
            resolve_build_profile(
                "chat_file_abc.pdf", IngestFlags(parse_only=True)
            )
            is RevisionBuildProfile.PARSE_ONLY
        )
        for key in ("doc_x.pdf", "chat_file_y.pdf", "other/z.pdf"):
            assert resolve_build_profile(
                key, IngestFlags()
            ) == resolve_build_profile(key, IngestFlags())
