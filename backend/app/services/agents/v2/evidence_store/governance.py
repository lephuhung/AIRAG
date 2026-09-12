"""Evidence Store governance: minimization, classification, encryption, hydration.

Spec §15.3/§24/§26. Before persistence::

    raw capability result
    → Evidence Builder/Minimizer
    → deterministic PII classification
    → EvidenceRecord + StoragePolicy
    → Evidence Store

This module owns that pipeline and the governed hydration gate:

- :func:`minimize_people_record` — People evidence stores only the task-required
  fields (deterministic canonical JSON, extra fields dropped).
- :func:`classify_evidence` — deterministic ``normal`` / ``personal`` /
  ``sensitive_personal`` floor derived from the source identity and the field
  names, which a model/detector can only *raise*, never downgrade.
- :class:`EvidenceKeyring` + :func:`encrypt_payload` / :func:`decrypt_content` —
  AES-256-GCM with the keyring supplied by runtime secret management
  (``EVIDENCE_ENCRYPTION_KEYS`` + ``EVIDENCE_ENCRYPTION_ACTIVE_KEY_ID``). A
  record persists only ciphertext, key id, nonce, and algorithm; there is no
  plaintext column. A missing key id or unavailable key fails closed with
  :class:`EvidenceKeyUnavailable` and never returns plaintext. Rotation adds a
  new active key id while retaining prior ids, so already-retained evidence
  stays readable with its recorded key id; in-place re-encryption is out of
  scope.
- :class:`EvidenceGovernor` — persists minimized/classified/encrypted records
  idempotently and hydrates an admitted ``EvidenceUse`` only after the
  authorization-bound checks (run scope, purpose/target, ACL through the
  authoritative revision, expiry, tombstone, revision requirement, recursive
  derived validation) pass, decrypting last. Every allow/deny read is audited.

Runtime/security metadata never enters a semantic contract: the governor reads
trusted authorization from the request-scoped ``CapabilityRuntimeContext`` and
resolves workspace ownership from the immutable revision row, never from copied
evidence metadata (spec §24).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Collection,
    Final,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
)

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.services.agents.v2.contracts.base import CONTRACT_VERSION
from app.services.agents.v2.contracts.binding import ScopedDocument
from app.services.agents.v2.contracts.capability import CapabilityRuntimeContext
from app.services.agents.v2.contracts.evidence import (
    DerivedSourceIdentity,
    DocumentSourceIdentity,
    EvidenceClassification,
    EvidenceRecord,
    EvidenceSourceIdentity,
    EvidenceStoreRow,
    PeopleSourceIdentity,
    Provenance,
    StoragePolicy,
)
from app.services.agents.v2.contracts.validation import (
    validate_evidence_store_row,
)
from app.services.agents.v2.persistence.evidence import (
    EncryptedEvidenceRecord,
    EvidenceRepository,
)


# ---------------------------------------------------------------------------
# Constants / errors
# ---------------------------------------------------------------------------

#: The only encryption algorithm persisted by this module.
ENCRYPTION_ALGORITHM: Final[str] = "AES-256-GCM"

#: AES-GCM nonce length in bytes (96-bit nonce, the recommended size).
GCM_NONCE_BYTES: Final[int] = 12

#: AES-256 key length in bytes.
KEY_BYTES: Final[int] = 32

#: The only value of ``validation_state`` that makes derived evidence eligible.
DERIVED_VALIDATED: Final[str] = "validated"

#: The deterministic derived-evidence validation states.
DERIVED_VALIDATION_STATES: frozenset[str] = frozenset(
    {"validated", "unvalidated", "failed"}
)

#: Lexical (deterministic) names that raise a People record to
#: ``sensitive_personal``. The rule is deliberately lexical: classification is
#: computed here from the stored field names, never taken from the model.
SENSITIVE_PERSON_FIELDS: frozenset[str] = frozenset(
    {
        "national_id",
        "citizen_id",
        "health_status",
        "medical_record",
        "religion",
        "ethnicity",
        "criminal_record",
        "biometric_id",
    }
)

_CLASSIFICATION_RANK: dict[str, int] = {
    "normal": 0,
    "personal": 1,
    "sensitive_personal": 2,
}


class EvidenceGovernanceError(Exception):
    """Base class for evidence-governance failures."""


class EvidenceKeyUnavailable(EvidenceGovernanceError):
    """The requested key id is not configured / the keyring is unavailable.

    Fail closed: the caller never receives plaintext.
    """


class EvidenceDecryptionError(EvidenceGovernanceError):
    """Ciphertext failed authentication (wrong key, tamper, or hash mismatch)."""


class EvidenceAccessDenied(EvidenceGovernanceError):
    """Hydration was refused; the reason is also written to the audit sink."""


class EvidenceMinimizationError(EvidenceGovernanceError):
    """A raw record cannot be minimized to the task-required fields."""


class EvidenceValidationError(EvidenceGovernanceError):
    """The evidence row violates a governance invariant before persistence."""


# ---------------------------------------------------------------------------
# Deterministic minimization / classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MinimizedEvidence:
    """The minimized content plus the field names that survived minimization."""

    content: str
    field_names: tuple[str, ...]


def _normalize_field_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def minimize_people_record(
    raw_record: Mapping[str, object],
    *,
    required_fields: Collection[str],
) -> MinimizedEvidence:
    """Keep only the task-required People fields (spec §15.3).

    Every field the task needs must be present — a missing required field is an
    error rather than a silently dropped fact. The content is canonical JSON
    (sorted keys, no whitespace) so the same record minimizes to the same bytes
    and the same content hash.
    """
    required = tuple(sorted({_normalize_field_name(str(f)) for f in required_fields}))
    missing = [name for name in required if name not in raw_record]
    if missing:
        raise EvidenceMinimizationError(
            "People evidence is missing task-required field(s): "
            f"{missing}"
        )
    kept = {name: raw_record[name] for name in required}
    content = json.dumps(
        kept,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return MinimizedEvidence(content=content, field_names=required)


def classify_evidence(
    source: EvidenceSourceIdentity,
    *,
    field_names: Collection[str] = (),
    detected: Optional[EvidenceClassification] = None,
) -> EvidenceClassification:
    """Deterministic evidence classification (spec §15.3).

    The floor is a pure function of the source identity and the stored field
    names: People evidence is at least ``personal`` (``sensitive_personal`` when
    a sensitive field is among the kept fields); other sources are ``normal``.
    ``detected`` is an optional external detector result that may only *raise*
    the classification — a model can never downgrade it.
    """
    if isinstance(source, PeopleSourceIdentity):
        sensitive = any(
            _normalize_field_name(str(name)) in SENSITIVE_PERSON_FIELDS
            for name in field_names
        )
        floor: EvidenceClassification = (
            "sensitive_personal" if sensitive else "personal"
        )
    else:
        floor = "normal"

    if detected is None:
        return floor
    if detected not in _CLASSIFICATION_RANK:
        raise EvidenceValidationError(
            f"unknown evidence classification {detected!r}"
        )
    if _CLASSIFICATION_RANK[detected] > _CLASSIFICATION_RANK[floor]:
        return detected
    return floor


def sha256_content_hash(content: str) -> str:
    """The deterministic ``EvidenceRecord.content_hash``."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Encryption (AES-256-GCM, runtime keyring)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncryptedPayload:
    """The four persisted encryption fields; no plaintext by construction."""

    ciphertext: bytes
    nonce: bytes
    encryption_key_id: str
    encryption_algorithm: str


class EvidenceKeyring:
    """Runtime keyring backed by ``EVIDENCE_ENCRYPTION_KEYS``.

    ``keys`` maps a key id to a 32-byte AES-256 key. New writes use
    ``active_key_id``; reads use the key id recorded on the row, so rotation
    (a new active id with the prior ids retained) keeps old evidence readable.
    """

    def __init__(
        self, keys: Mapping[str, bytes], active_key_id: Optional[str]
    ) -> None:
        self._keys: dict[str, bytes] = dict(keys)
        self._active_key_id = (
            active_key_id.strip() if active_key_id else None
        ) or None

    @classmethod
    def from_json(
        cls, keys_json: Optional[str], active_key_id: Optional[str]
    ) -> "EvidenceKeyring":
        """Parse the JSON keyring env value ``{"key_id": "<base64 32B>"}``."""
        keys: dict[str, bytes] = {}
        if keys_json is not None and keys_json.strip():
            try:
                raw = json.loads(keys_json)
            except json.JSONDecodeError as exc:
                raise EvidenceKeyUnavailable(
                    f"EVIDENCE_ENCRYPTION_KEYS is not valid JSON: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise EvidenceKeyUnavailable(
                    "EVIDENCE_ENCRYPTION_KEYS must be a JSON object mapping "
                    "key id to base64 key material"
                )
            for key_id, encoded in raw.items():
                if (
                    not isinstance(key_id, str)
                    or not key_id.strip()
                    or not isinstance(encoded, str)
                ):
                    raise EvidenceKeyUnavailable(
                        "EVIDENCE_ENCRYPTION_KEYS entries must map a non-blank "
                        "key id to a base64 string"
                    )
                try:
                    keys[key_id.strip()] = base64.b64decode(
                        encoded, validate=True
                    )
                except (binascii.Error, ValueError) as exc:
                    raise EvidenceKeyUnavailable(
                        f"EVIDENCE_ENCRYPTION_KEYS entry {key_id!r} is not "
                        f"valid base64: {exc}"
                    ) from exc
        return cls(keys, active_key_id)

    @classmethod
    def from_settings(cls, settings=None) -> "EvidenceKeyring":
        """Build the keyring from the runtime configuration."""
        if settings is None:
            from app.core.config import settings as runtime_settings

            settings = runtime_settings
        return cls.from_json(
            getattr(settings, "EVIDENCE_ENCRYPTION_KEYS", ""),
            getattr(settings, "EVIDENCE_ENCRYPTION_ACTIVE_KEY_ID", ""),
        )

    def active(self) -> str:
        """The key id new writes must use."""
        if not self._active_key_id:
            raise EvidenceKeyUnavailable(
                "no evidence encryption key id is active "
                "(EVIDENCE_ENCRYPTION_ACTIVE_KEY_ID is unset)"
            )
        return self._active_key_id

    def key(self, key_id: str) -> bytes:
        """The 32-byte key for ``key_id``; fail closed when unavailable."""
        if not key_id:
            raise EvidenceKeyUnavailable(
                "evidence record carries no encryption key id"
            )
        try:
            key = self._keys[key_id]
        except KeyError as exc:
            raise EvidenceKeyUnavailable(
                f"evidence encryption key id {key_id!r} is not in "
                "EVIDENCE_ENCRYPTION_KEYS"
            ) from exc
        if len(key) != KEY_BYTES:
            raise EvidenceKeyUnavailable(
                f"evidence encryption key {key_id!r} must be {KEY_BYTES} bytes, "
                f"got {len(key)}"
            )
        return key

    def __contains__(self, key_id: object) -> bool:
        return isinstance(key_id, str) and key_id in self._keys


def _associated_data(evidence_id: uuid.UUID) -> bytes:
    """Bind each ciphertext to its own evidence id."""
    return str(evidence_id).encode("utf-8")


def encrypt_payload(
    keyring: EvidenceKeyring, *, evidence_id: uuid.UUID, plaintext: str
) -> EncryptedPayload:
    """Encrypt ``plaintext`` with the active key (AES-256-GCM)."""
    key_id = keyring.active()
    key = keyring.key(key_id)
    nonce = os.urandom(GCM_NONCE_BYTES)
    ciphertext = AESGCM(key).encrypt(
        nonce, plaintext.encode("utf-8"), _associated_data(evidence_id)
    )
    return EncryptedPayload(
        ciphertext=ciphertext,
        nonce=nonce,
        encryption_key_id=key_id,
        encryption_algorithm=ENCRYPTION_ALGORITHM,
    )


def decrypt_content(
    keyring: EvidenceKeyring, record: EncryptedEvidenceRecord
) -> str:
    """Decrypt one stored row with its recorded key id; fail closed.

    Raises :class:`EvidenceKeyUnavailable` when the recorded key id is not in
    the keyring and :class:`EvidenceDecryptionError` when authentication or the
    content-hash check fails. Neither path returns plaintext.
    """
    if record.encryption_algorithm != ENCRYPTION_ALGORITHM:
        raise EvidenceDecryptionError(
            f"unsupported evidence encryption algorithm "
            f"{record.encryption_algorithm!r}"
        )
    key = keyring.key(record.encryption_key_id)
    try:
        plaintext = AESGCM(key).decrypt(
            record.nonce, record.ciphertext, _associated_data(record.evidence_id)
        )
    except InvalidTag as exc:
        raise EvidenceDecryptionError(
            f"evidence {record.evidence_id} failed AES-256-GCM authentication"
        ) from exc
    try:
        content = plaintext.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceDecryptionError(
            f"evidence {record.evidence_id} did not decrypt to UTF-8 text"
        ) from exc
    if sha256_content_hash(content) != record.content_hash:
        raise EvidenceDecryptionError(
            f"evidence {record.evidence_id} content hash does not match the "
            "stored hash"
        )
    return content


# ---------------------------------------------------------------------------
# Audited access
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceAccessDecision:
    """One audited evidence-read decision (allow or deny)."""

    use_id: Optional[uuid.UUID]
    evidence_id: Optional[uuid.UUID]
    run_id: Optional[str]
    allowed: bool
    reason: str
    occurred_at: datetime


@runtime_checkable
class EvidenceAccessAuditor(Protocol):
    """Sink for every allow/deny evidence read."""

    def record(self, decision: EvidenceAccessDecision) -> None: ...


class LoggingEvidenceAccessAuditor:
    """Default auditor: structured ``logging`` records, never silent."""

    def __init__(self) -> None:
        self._log = logging.getLogger(__name__)

    def record(self, decision: EvidenceAccessDecision) -> None:
        self._log.log(
            logging.INFO if decision.allowed else logging.WARNING,
            "evidence access %s use_id=%s evidence_id=%s run_id=%s reason=%s",
            "allowed" if decision.allowed else "denied",
            decision.use_id,
            decision.evidence_id,
            decision.run_id,
            decision.reason,
        )


# ---------------------------------------------------------------------------
# Governed store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceHydrationRequest:
    """Runtime inputs for one hydration (never checkpointed business state)."""

    runtime: CapabilityRuntimeContext
    #: The binding the admitted use resolved through, when the caller has one.
    #: Enforces the §15.2/§26 revision-match rule on reuse.
    required_binding: Optional[ScopedDocument] = None
    #: Deterministic clock override for expiry decisions.
    occurred_at: Optional[datetime] = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EvidenceGovernor:
    """The governed Evidence Store boundary (persist + hydrate).

    The governor only mutates and ``flush`` es through
    :class:`~app.services.agents.v2.persistence.evidence.EvidenceRepository`; the
    unit of work owns the transaction.
    """

    def __init__(
        self,
        session,
        *,
        keyring: Optional[EvidenceKeyring] = None,
        auditor: Optional[EvidenceAccessAuditor] = None,
        clock=None,
    ) -> None:
        self._repository = EvidenceRepository(session)
        self._keyring = (
            keyring if keyring is not None else EvidenceKeyring.from_settings()
        )
        self._auditor: EvidenceAccessAuditor = (
            auditor if auditor is not None else LoggingEvidenceAccessAuditor()
        )
        self._clock = clock if clock is not None else _utcnow

    @property
    def repository(self) -> EvidenceRepository:
        return self._repository

    @property
    def keyring(self) -> EvidenceKeyring:
        return self._keyring

    # -- persistence -------------------------------------------------------

    async def persist_record(
        self,
        *,
        source: EvidenceSourceIdentity,
        content: str,
        provenance: Provenance,
        field_names: Collection[str] = (),
        expires_at: Optional[datetime] = None,
        revision_id: Optional[uuid.UUID] = None,
        validation_state: Optional[str] = None,
        evidence_id: Optional[uuid.UUID] = None,
        detected_classification: Optional[EvidenceClassification] = None,
    ) -> uuid.UUID:
        """Minimize → classify → hash → encrypt → idempotently persist.

        Returns the persisted ``evidence_id`` (the existing one when the same
        source identity + content hash is already stored).
        """
        self._require_valid_governance_fields(
            source=source,
            revision_id=revision_id,
            validation_state=validation_state,
        )
        classification = classify_evidence(
            source, field_names=field_names, detected=detected_classification
        )
        content_hash = sha256_content_hash(content)
        row = EvidenceStoreRow(
            contract_version=CONTRACT_VERSION,
            record=EvidenceRecord(
                evidence_id=evidence_id or uuid.uuid4(),
                source=source,
                content=content,
                content_hash=content_hash,
                provenance=provenance,
            ),
            storage_policy=StoragePolicy(
                classification=classification, expires_at=expires_at
            ),
        )
        validate_evidence_store_row(row)

        payload = encrypt_payload(
            self._keyring,
            evidence_id=row.record.evidence_id,
            plaintext=content,
        )
        stored = EncryptedEvidenceRecord(
            evidence_id=row.record.evidence_id,
            contract_version=row.contract_version,
            content_hash=content_hash,
            classification=classification,
            expires_at=expires_at,
            revision_id=revision_id,
            source=source,
            provenance=provenance,
            validation_state=validation_state,
            ciphertext=payload.ciphertext,
            encryption_key_id=payload.encryption_key_id,
            nonce=payload.nonce,
            encryption_algorithm=payload.encryption_algorithm,
        )
        return await self._repository.insert_record(stored)

    async def persist_people_evidence(
        self,
        *,
        record_id: str,
        raw_record: Mapping[str, object],
        required_fields: Collection[str],
        provenance: Provenance,
        expires_at: Optional[datetime] = None,
        detected_classification: Optional[EvidenceClassification] = None,
        evidence_id: Optional[uuid.UUID] = None,
    ) -> uuid.UUID:
        """Minimize a raw People record and persist only task-required fields."""
        minimized = minimize_people_record(
            raw_record, required_fields=required_fields
        )
        return await self.persist_record(
            source=PeopleSourceIdentity(kind="people", record_id=record_id),
            content=minimized.content,
            field_names=minimized.field_names,
            provenance=provenance,
            expires_at=expires_at,
            detected_classification=detected_classification,
            evidence_id=evidence_id,
        )

    # -- hydration ---------------------------------------------------------

    async def hydrate_use(
        self,
        *,
        use_id: uuid.UUID,
        request: EvidenceHydrationRequest,
    ) -> str:
        """Return the decrypted evidence content for an authorized current-run use.

        Authorization is the same ACL/expiry/source/revision/derived-validation
        check as hydration; decryption happens only after every check passes.
        Every allow/deny outcome is written to the audit sink.
        """
        runtime = request.runtime
        occurred_at = request.occurred_at or self._clock()

        envelope = await self._repository.load_use(use_id)
        if envelope is None:
            self._deny(
                use_id=use_id,
                evidence_id=None,
                run_id=runtime.run_id,
                reason="unknown_use",
                occurred_at=occurred_at,
            )
        use = envelope.use
        if envelope.run_id != runtime.run_id:
            self._deny(
                use_id=use.use_id,
                evidence_id=use.evidence_id,
                run_id=runtime.run_id,
                reason="cross_run",
                occurred_at=occurred_at,
            )
        if use.purpose == "discovery":
            self._deny(
                use_id=use.use_id,
                evidence_id=use.evidence_id,
                run_id=runtime.run_id,
                reason="purpose_not_hydratable",
                occurred_at=occurred_at,
            )
        if use.purpose == "coverage" and use.target_id is None:
            self._deny(
                use_id=use.use_id,
                evidence_id=use.evidence_id,
                run_id=runtime.run_id,
                reason="purpose_not_hydratable",
                occurred_at=occurred_at,
            )

        record = await self._repository.load_record(use.evidence_id)
        if record is None:
            self._deny(
                use_id=use.use_id,
                evidence_id=use.evidence_id,
                run_id=runtime.run_id,
                reason="unknown_evidence",
                occurred_at=occurred_at,
            )
        if record.payload_purged_at is not None:
            self._deny(
                use_id=use.use_id,
                evidence_id=use.evidence_id,
                run_id=runtime.run_id,
                reason="purged",
                occurred_at=occurred_at,
            )
        if record.expires_at is not None and record.expires_at <= occurred_at:
            self._deny(
                use_id=use.use_id,
                evidence_id=use.evidence_id,
                run_id=runtime.run_id,
                reason="expired",
                occurred_at=occurred_at,
            )

        failure = self._revision_requirement_failure(record, request)
        if failure is None:
            failure = await self._authorization_failure(record, runtime)
        if failure is not None:
            self._deny(
                use_id=use.use_id,
                evidence_id=use.evidence_id,
                run_id=runtime.run_id,
                reason=failure,
                occurred_at=occurred_at,
            )

        try:
            content = decrypt_content(self._keyring, record)
        except (EvidenceKeyUnavailable, EvidenceDecryptionError) as exc:
            # Fail closed with the specific typed error (never a wrapped denial
            # that could hide a missing key), but still audit the denied read.
            self._auditor.record(
                EvidenceAccessDecision(
                    use_id=use.use_id,
                    evidence_id=use.evidence_id,
                    run_id=runtime.run_id,
                    allowed=False,
                    reason=(
                        "key_unavailable"
                        if isinstance(exc, EvidenceKeyUnavailable)
                        else "decryption_failed"
                    ),
                    occurred_at=occurred_at,
                )
            )
            raise

        self._auditor.record(
            EvidenceAccessDecision(
                use_id=use.use_id,
                evidence_id=use.evidence_id,
                run_id=runtime.run_id,
                allowed=True,
                reason="allowed",
                occurred_at=occurred_at,
            )
        )
        return content

    # -- internal ----------------------------------------------------------

    @staticmethod
    def _require_valid_governance_fields(
        *,
        source: EvidenceSourceIdentity,
        revision_id: Optional[uuid.UUID],
        validation_state: Optional[str],
    ) -> None:
        if isinstance(source, DocumentSourceIdentity):
            if revision_id is None:
                raise EvidenceValidationError(
                    "document evidence must reference the authoritative "
                    "revision_id it was read from (workspace resolves through "
                    "the revision, never a copied allowlist)"
                )
        elif revision_id is not None:
            raise EvidenceValidationError(
                f"{source.kind!r} evidence must not carry a document revision_id"
            )

        if isinstance(source, DerivedSourceIdentity):
            if validation_state not in DERIVED_VALIDATION_STATES:
                raise EvidenceValidationError(
                    "derived evidence must declare a validation_state in "
                    f"{sorted(DERIVED_VALIDATION_STATES)}, got "
                    f"{validation_state!r}"
                )
        elif validation_state is not None:
            raise EvidenceValidationError(
                "validation_state is derived-evidence metadata and must be "
                "NULL for non-derived evidence"
            )

    @staticmethod
    def _revision_requirement_failure(
        record: EncryptedEvidenceRecord,
        request: EvidenceHydrationRequest,
    ) -> Optional[str]:
        binding = request.required_binding
        if binding is None:
            return None
        source = record.source
        if not isinstance(source, DocumentSourceIdentity):
            return "revision_mismatch"
        if (
            source.document_id != binding.document_id
            or source.document_revision != binding.document_revision
        ):
            return "revision_mismatch"
        return None

    async def _authorization_failure(
        self,
        record: EncryptedEvidenceRecord,
        runtime: CapabilityRuntimeContext,
    ) -> Optional[str]:
        source = record.source
        if isinstance(source, DocumentSourceIdentity):
            if record.revision_id is None:
                return "revision_unresolved"
            resolved = await self._repository.resolve_revision_workspace(
                record.revision_id
            )
            if resolved is None:
                return "revision_unresolved"
            if resolved.document_id != source.document_id:
                return "revision_mismatch"
            if resolved.source_deleted_at is not None:
                return "document_tombstoned"
            if resolved.workspace_id not in runtime.workspace_ids:
                return "workspace_not_authorized"
            return None
        if isinstance(source, PeopleSourceIdentity):
            if not runtime.can_read_people:
                return "people_not_authorized"
            return None
        if isinstance(source, DerivedSourceIdentity):
            if record.validation_state != DERIVED_VALIDATED:
                return "derived_not_validated"
            return await self._recursive_source_failure(source)
        return None

    async def _recursive_source_failure(
        self, source: DerivedSourceIdentity
    ) -> Optional[str]:
        """Every recursively resolved source must exist, be live, and be validated."""
        pending = list(source.source_evidence_ids)
        visited: set[uuid.UUID] = set()
        while pending:
            evidence_id = pending.pop()
            if evidence_id in visited:
                continue
            visited.add(evidence_id)
            row = await self._repository.load_record(evidence_id)
            if row is None or row.payload_purged_at is not None:
                return "derived_source_unresolved"
            if isinstance(row.source, DerivedSourceIdentity):
                if row.validation_state != DERIVED_VALIDATED:
                    return "derived_source_unresolved"
                pending.extend(row.source.source_evidence_ids)
        return None

    def _deny(self, **decision) -> None:
        self._auditor.record(
            EvidenceAccessDecision(allowed=False, **decision)
        )
        raise EvidenceAccessDenied(
            f"evidence hydration denied: {decision['reason']}"
        )
