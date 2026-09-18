"""Side-effect-free v2 shadow execution (Phase 3, Task 6).

A shadow v2 run replays a primary turn through the SAME supervisor
topology — ``create_supervisor_v2_graph(checkpointer=isolated_saver)`` —
while proving it can never write production state or emit outbound events.

Global constraints honored here:

- Shadow compiles its OWN graph with an isolated saver bundle
  (``v2.persistence.shadow_checkpoint.create_shadow_checkpointer``); it
  NEVER calls the production ``get_supervisor_v2_graph()``, NEVER
  constructs the production saver (``v2.persistence.checkpoint`` is not
  imported on any path), and never touches the production graph selector,
  evidence governor, chat persistence, lease tables, or outbound sinks
  (proven by AST + live-spy + DB-count tests).
- Read-only source adapters only: every shadow source exposes reads and
  raises :class:`ShadowIsolationError` on any write surface. The shadow
  people directory, document view, semantic adapter, binding resolver,
  evidence builder/hydrator, lease repo, chat-messages stub, and
  authorization stub are all constructed over per-run ISOLATED stores.
- Factual shadow queries reach the SHARED ``TaskScheduler`` and the REAL
  ``people.lookup`` capability (built by the real
  ``build_v2_capability_registry``); document/section reads stay gated out
  and fail closed to typed outcomes. Authorization is NEVER hardcoded:
  ``can_read_people`` and ``allowed_capabilities`` are required caller
  inputs mirroring the primary turn (R64). Cancellation follows the
  primary run (``asyncio`` cancellation propagates; :func:`_stop_shadow_task`
  cancels twice over shield-bounded waits and reports ``False`` instead of
  hanging on a persistently resistant shadow, R65/R66.4).
- Output is discarded except redacted metrics (:class:`ShadowMetrics` —
  status/route/evaluation/counts/timing only, never response content).
- Exactly one ownership chain: the shadow graph is the same
  ``create_supervisor_v2_graph`` topology — no alternate graph/agent.
- Frozen contract types are imported, never redefined.
- Only the shared ``TaskScheduler`` dispatches a capability.
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Mapping
from uuid import UUID, uuid4

from app.services.agent.runtime_selector import PlanBindingResolver
from app.services.agents.supervisor_v2 import (
    V1PeopleLookupService,
    V1ServiceBundle,
    build_v2_capability_registry,
    create_supervisor_v2_graph,
)
from app.services.agents.v2.adapters.document import binding_id_for_ref
from app.services.agents.v2.capabilities import CapabilityRuntimeContext
from app.services.agents.v2.contracts.base import CONTRACT_VERSION
from app.services.agents.v2.contracts.binding import DocumentBindingSet, ScopedDocument
from app.services.agents.v2.contracts.conversation import (
    ConversationContext,
    ConversationTurn,
    EntityReference,
)
from app.services.agents.v2.contracts.evidence import (
    EvidencePurpose,
    EvidenceUse,
    EvidenceUseRef,
    PeopleSourceIdentity,
    Provenance,
)
from app.services.agents.v2.contracts.request import KnownDocumentResource, RequestContext
from app.services.agents.v2.contracts.semantic import (
    DocumentReference,
    SemanticContext,
    SemanticDraft,
)
from app.services.agents.v2.contracts.state import (
    CHECKPOINT_SCHEMA_REVISION,
    ExecutionState,
    GraphRuntimeContext,
    RuntimeServices,
    SupervisorV2State,
)
from app.services.agents.v2.contracts.validation import validate_evidence_use
from app.services.agents.v2.nodes.evaluate import AnswerDraftChannel, HydratedEvidence
from app.services.agents.v2.synthesis.adapter import StructuredLLMDraftBuilder
from app.services.agents.v2.synthesis.citations import StoreCitationResolver
from app.services.llm import get_main_provider_for_synthesis
from app.services.agents.v2.planning import AdaptivePlanner, AdaptiveReplanner
from app.services.agents.v2.semantic.intent import IntentClassifier
from app.services.agents.v2.persistence.shadow_checkpoint import (
    ShadowCheckpointBundle,
    create_shadow_checkpointer,
    is_shadow_saver,
)


class ShadowDependencyGap(Exception):
    """A shadow read-only dependency failed (typed gap, R66.1).

    Raised when a shadow source the run depends on (currently the
    read-only People source) fails instead of answering. Carries the
    ``reason`` fragment; :meth:`ShadowBundle.run` maps it to a typed
    ``unavailable`` / ``dependency-gap:`` outcome — never a generic
    zero-task error.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


__all__ = [
    "ShadowIsolationError",
    "ShadowDependencyGap",
    "ShadowMetrics",
    "ReadOnlySourceAdapter",
    "IsolatedShadowStores",
    "ShadowPeopleDirectory",
    "ReadOnlyProductionPeopleSource",
    "ShadowSemanticAdapter",
    "ShadowBindingResolver",
    "ShadowEvidenceBuilder",
    "ShadowEvidenceHydrator",
    "ShadowLeaseRepo",
    "ShadowChatMessages",
    "ShadowAuthorization",
    "ShadowBundle",
    "build_shadow_bundle",
    "should_run_shadow",
    "shadow_sampling_enabled",
]


logger = logging.getLogger(__name__)


class ShadowIsolationError(AttributeError):
    """A shadow run reached for a forbidden write or outbound surface.

    Subclasses ``AttributeError`` on purpose: write attributes raise this
    instead of existing, so ``hasattr(adapter, "write")`` is ``False`` —
    no write path is even reachable from the shadow bundle, let alone
    callable.
    """


#: Attribute names that count as a write/outbound path on a shadow object.
#: Any access to one of these raises :class:`ShadowIsolationError`.
_WRITE_SURFACE = frozenset(
    {
        "write",
        "save",
        "commit",
        "persist",
        "persist_record",
        "persist_people_evidence",
        "persist_use",
        "insert",
        "insert_record",
        "append",
        "append_use",
        "upsert",
        "update",
        "delete",
        "remove",
        "put",
        "post",
        "send",
        "send_message",
        "send_chat_action",
        "emit",
        "publish",
        "notify",
    }
)


class ReadOnlySourceAdapter:
    """Read-only wrapper around one shadow source.

    Only the wrapped ``read``/``lookup``/``fetch`` behavior is exposed;
    every write-surface attribute raises :class:`ShadowIsolationError`.
    The wrapper keeps no reference to any production session, connection,
    or store — it is constructed over the isolated shadow stores only.
    """

    def __init__(self, name: str, reader: Any = None) -> None:
        self._name = name
        self._reader = reader

    @property
    def adapter_name(self) -> str:
        return self._name

    @property
    def read_only(self) -> bool:
        return True

    async def read(self, *args: Any, **kwargs: Any) -> Any:
        """The only data path: delegate a read, or return no rows."""
        if self._reader is None:
            return None
        read = getattr(self._reader, "read", None)
        if callable(read):
            result = read(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return None

    async def lookup(self, *args: Any, **kwargs: Any) -> Any:
        """Read-only lookup alias (same delegation as :meth:`read`)."""
        return await self.read(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in _WRITE_SURFACE:
            raise ShadowIsolationError(
                f"shadow source adapter {self.__dict__.get('_name', '?')!r} "
                f"is read-only; write path {name!r} is not reachable"
            )
        raise AttributeError(
            f"shadow source adapter has no attribute {name!r}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _WRITE_SURFACE:
            raise ShadowIsolationError(
                "shadow source adapter is read-only; cannot set "
                f"{name!r}"
            )
        object.__setattr__(self, name, value)


@dataclass
class IsolatedShadowStores:
    """Per-run isolated stores: fresh dicts, never production handles.

    Keys mirror the production tables the isolation proof watches
    (checkpoint/evidence/use/audit/chat/memory/title) so tests can assert
    every shadow write landed here while production mappings/rows are
    untouched. ``leases`` records isolated retention-lease acquisitions
    (proving the scheduler's lease path ran without touching the lease
    table).
    """

    checkpoint: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    evidence_use: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    chat: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    title: dict[str, Any] = field(default_factory=dict)
    leases: list[dict[str, Any]] = field(default_factory=list)

    def write_count(self) -> int:
        """Total isolated entries written (records + uses + leases)."""
        return (
            len(self.evidence)
            + len(self.evidence_use)
            + len(self.leases)
            + len(self.checkpoint)
        )


@dataclass(frozen=True)
class ShadowMetrics:
    """Redacted shadow-run metrics — the ONLY shadow output.

    Carries status/route/evaluation/counts/timing. Response content,
    citations, evidence payloads, and user text are deliberately absent:
    there is no field that could carry them.
    """

    status: str
    route: str | None
    evaluation: str | None
    task_count: int
    isolated_writes: int
    duration_ms: int
    reason: str | None = None

    def redacted(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "route": self.route,
            "evaluation": self.evaluation,
            "task_count": self.task_count,
            "isolated_writes": self.isolated_writes,
            "duration_ms": self.duration_ms,
            "reason": self.reason,
        }


class ShadowPeopleDirectory:
    """Read-only isolated people directory (the ``people.lookup`` backing).

    Constructed over an isolated ``{name: record}`` snapshot — never over
    Mongo or any production store. The ONLY method is ``lookup``; there is
    no write surface at all.
    """

    def __init__(self, records: Mapping[str, Mapping[str, object]]) -> None:
        self._records = {
            str(name).strip().lower(): dict(record)
            for name, record in dict(records).items()
        }

    async def lookup(self, query: str) -> Mapping[str, object] | None:
        """Return the isolated record whose name appears in ``query``."""
        needle = str(query).strip().lower()
        for name, record in self._records.items():
            if name and name in needle:
                return dict(record)
        return None


class ReadOnlyProductionPeopleSource:
    """The production people lookup service used in read-only mode (R64).

    Wraps :class:`V1PeopleLookupService` and exposes ONLY ``lookup`` — the
    service itself performs reads (Mongo) and offers no write method, and
    the wrapper adds no surface beyond ``lookup``/``candidate_names``.
    ``candidate_names`` is empty (the production service has no listing
    API); name confirmation happens per-span through ``lookup``.
    """

    def __init__(self, service: Any | None = None) -> None:
        self._service = (
            service if service is not None else V1PeopleLookupService()
        )

    @property
    def read_only(self) -> bool:
        return True

    def candidate_names(self) -> tuple[str, ...]:
        return ()

    async def lookup(self, query: str) -> Mapping[str, object] | None:
        """Delegate one read to the production people lookup."""
        result = self._service.lookup(str(query))
        if asyncio.iscoroutine(result):
            return await result
        return result


#: Runs of 2+ uppercase-initial words (Vietnamese-aware): candidate
#: person-name spans for read-confirmed extraction. Each word MUST start
#: uppercase (remaining letters any case); a span becomes a person
#: reference ONLY after the read-only people source confirms a record.
_PERSON_SPAN_RE = re.compile(
    r"(?:[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐÊÔƠƯ]["
    r"A-Za-zÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐÊÔƠƯ"
    r"a-zàáâãèéêìíòóôõùúýăđêôơưạ-ỹ]*\s*){2,}"
)

#: Cap on spans confirmed per draft (bounds read-only lookup fan-out).
_MAX_PERSON_SPANS = 5


def _looks_like_name(span: str) -> bool:
    """Keep spans with at least one Titlecase word (len >= 2).

    Rejects pure-acronym runs (``CCCD``) and lone initials (``A B``) that
    the span regex self-splits into two uppercase-initial words — they
    would only waste a read-only lookup that cannot confirm a person.
    """
    for word in span.split():
        if (
            len(word) >= 2
            and word[0].isupper()
            and any(char.islower() for char in word[1:])
        ):
            return True
    return False


class ShadowSemanticAdapter:
    """Deterministic read-only semantic adapter that PRESERVES references.

    Unlike a stripping adapter, the draft keeps person references (from
    ``person_names``), document references (from ``known_documents``), and
    the verbatim normalized query — so factual shadow queries route to the
    real fast-domain topology instead of collapsing to ``direct``. It reads
    only its isolated inputs and writes nothing.
    """

    def __init__(
        self,
        raw_query: str,
        *,
        person_names: tuple[str, ...] = (),
        known_documents: tuple[UUID, ...] = (),
        document_view: Mapping[Any, Mapping[str, Any]] | None = None,
        people_source: Any | None = None,
        can_read_people: bool = False,
    ) -> None:
        self._query = raw_query
        self._person_names = tuple(person_names)
        self._known_documents = tuple(known_documents)
        self._view = dict(document_view or {})
        self._people_source = people_source
        # Mirrored primary authorization (R66.1): the read-only People
        # source is consulted ONLY when the primary turn could read
        # people. Safe default False — never consult unless granted.
        self._can_read_people = bool(can_read_people)
        # Cause of the last people-source failure, read by ShadowBundle.run
        # to report a typed dependency gap (reset on every draft attempt).
        self.last_gap_reason: str | None = None

    async def _confirmed_person_names(self) -> tuple[str, ...]:
        """Person names confirmed read-only against the people source.

        Explicit ``person_names`` (bundle-level/test input) win. Otherwise
        the read-only source is consulted without any write — but ONLY
        when ``can_read_people`` mirrors a granted primary turn (R66.1): a
        denied turn consults nothing and yields no names. A directory
        source contributes substring matches over its isolated snapshot; a
        production source confirms capitalized spans via ``lookup``
        (capped). Unconfirmed spans never become references; a source that
        FAILS (rather than misses) records a dependency-gap reason and
        raises :class:`ShadowDependencyGap` so the run reports a typed
        gap instead of a generic zero-task outcome.
        """
        self.last_gap_reason = None
        if self._person_names:
            return self._person_names
        if not self._can_read_people:
            return ()
        source = self._people_source
        if source is None:
            return ()
        lowered = self._query.strip().lower()
        candidates = getattr(source, "candidate_names", None)
        if callable(candidates):
            try:
                names = tuple(candidates()) or ()
            except Exception as exc:  # noqa: BLE001
                self.last_gap_reason = f"people source listing failed: {exc}"
                raise ShadowDependencyGap(self.last_gap_reason) from exc
            if names:
                return tuple(
                    name for name in names if name and name.lower() in lowered
                )
        lookup = getattr(source, "lookup", None)
        if not callable(lookup):
            return ()
        confirmed: list[str] = []
        seen: set[str] = set()
        for match in _PERSON_SPAN_RE.findall(self._query):
            span = " ".join(match.split())
            if span in seen or not _looks_like_name(span):
                continue
            seen.add(span)
            if len(confirmed) >= _MAX_PERSON_SPANS:
                break
            try:
                record = lookup(span)
                if asyncio.iscoroutine(record):
                    record = await record
            except ShadowDependencyGap:
                raise
            except Exception as exc:  # noqa: BLE001 — source failure: gap
                self.last_gap_reason = (
                    f"people source lookup failed for span {span!r}: {exc}"
                )
                raise ShadowDependencyGap(self.last_gap_reason) from exc
            if isinstance(record, Mapping) and record.get("name"):
                confirmed.append(str(record["name"]))
        return tuple(confirmed)

    def _document_ref(
        self, index: int, document_id: UUID
    ) -> DocumentReference:
        """Resolve one known document against the isolated view (read-only).

        Mirrors the production adapter's read path: a known document with
        exactly one isolated candidate resolves (status + canonical id);
        an unknown document stays unresolved so the router clarifies — a
        typed outcome, never a guess.
        """
        if document_id in self._view:
            return DocumentReference(
                ref_id=f"doc-{index}",
                original_span=str(document_id),
                normalized_reference=str(document_id).lower(),
                requested_role="target",
                revision_requirement=None,
                resolution_status="resolved",
                resolved_document_id=document_id,
                candidate_document_ids=(),
            )
        return DocumentReference(
            ref_id=f"doc-{index}",
            original_span=str(document_id),
            normalized_reference=str(document_id).lower(),
            requested_role="target",
            revision_requirement=None,
            resolution_status="unresolved",
            resolved_document_id=None,
            candidate_document_ids=(document_id,),
        )

    async def build_draft(
        self, request: RequestContext, conversation: ConversationContext
    ) -> SemanticDraft:
        normalized = self._query.strip()
        person_names = await self._confirmed_person_names()
        return SemanticDraft(
            provisional_contextualized_query=normalized.lower(),
            abbreviations=(),
            coreferences=(),
            document_refs=tuple(
                self._document_ref(index, document_id)
                for index, document_id in enumerate(self._known_documents)
            ),
            person_refs=tuple(
                EntityReference(ref_id=f"p{index}", kind="person", label=name)
                for index, name in enumerate(person_names)
            ),
            section_refs=(),
            preliminary_ambiguities=(),
        )


class ShadowBindingResolver:
    """Read-only binding resolver over an isolated document view.

    ``document_view`` maps ``document_id -> {"revision": str, "role": str}``
    from an isolated snapshot. Refs whose candidate is in the view pin to a
    ``ScopedDocument``; unknown refs stay unresolved (the router then
    clarifies — a typed outcome, never a guess). No lease, session, or
    production store is touched.
    """

    def __init__(self, document_view: Mapping[Any, Mapping[str, str]]) -> None:
        self._view = {
            (key if isinstance(key, UUID) else UUID(str(key))): dict(value)
            for key, value in dict(document_view).items()
        }

    async def resolve(
        self, document_refs: Any, capability_runtime: CapabilityRuntimeContext
    ) -> DocumentBindingSet:
        pinned: list[ScopedDocument] = []
        for reference in document_refs or ():
            document_id = getattr(reference, "resolved_document_id", None)
            if document_id is None:
                candidates = tuple(
                    getattr(reference, "candidate_document_ids", ()) or ()
                )
                document_id = candidates[0] if len(candidates) == 1 else None
            if document_id is None or document_id not in self._view:
                continue
            entry = self._view[document_id]
            pinned.append(
                ScopedDocument(
                    binding_id=binding_id_for_ref(reference.ref_id),
                    document_id=document_id,
                    document_revision=str(entry.get("revision", "r1")),
                    role=entry.get("role", "target"),  # type: ignore[arg-type]
                )
            )
        return DocumentBindingSet(bindings=tuple(pinned), revision_requirement_refs=())


class ShadowEvidenceBuilder:
    """Isolated ``EvidenceBuilder``: mints records + uses into shadow stores.

    Implements the ``EvidenceBuilder`` protocol (``persist_use``) so the
    REAL capabilities persist through it during a factual shadow run. Every
    write lands in :attr:`IsolatedShadowStores.evidence` /
    ``evidence_use`` — never in the production evidence tables.
    """

    def __init__(self, stores: IsolatedShadowStores) -> None:
        self._stores = stores

    async def persist_use(
        self,
        *,
        source: Any,
        content: str,
        provenance: Provenance,
        task_id: str,
        purpose: EvidencePurpose,
        target_id: str | None,
    ) -> EvidenceUseRef:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("evidence content must be non-blank")
        evidence_id = uuid4()
        use = EvidenceUse(
            use_id=uuid4(),
            evidence_id=evidence_id,
            task_id=task_id,
            purpose=purpose,
            target_id=target_id,
        )
        validate_evidence_use(use)
        classification = (
            "personal" if isinstance(source, PeopleSourceIdentity) else "normal"
        )
        self._stores.evidence[str(evidence_id)] = {
            "source": source,
            "content": content,
            "provenance": provenance,
            "classification": classification,
        }
        self._stores.evidence_use[str(use.use_id)] = {"use": use}
        return EvidenceUseRef(use_id=use.use_id)


class ShadowEvidenceHydrator:
    """Isolated evidence hydrator resolving ONLY shadow-store uses.

    Revalidates each use against the checkpointed plan (task membership;
    coverage uses need a planned target) and resolves content from the
    isolated evidence map. Unknown or plan-incompatible uses fail closed.
    """

    def __init__(self, stores: IsolatedShadowStores) -> None:
        self._stores = stores

    def _resolve(
        self, use_refs: tuple[Any, ...], *, plan: Any, bindings: Any
    ) -> tuple[HydratedEvidence, ...]:
        task_ids = {task.task_id for task in plan.tasks}
        target_ids = {unit.target_id for unit in plan.target_units}
        hydrated: list[HydratedEvidence] = []
        for reference in use_refs:
            use_id = getattr(reference, "use_id", None)
            entry = self._stores.evidence_use.get(str(use_id))
            if entry is None:
                raise ValueError(
                    f"shadow hydrator cannot resolve unknown use {use_id!r}; "
                    "refusing to hydrate outside the isolated stores"
                )
            use: EvidenceUse = entry["use"]
            if use.task_id not in task_ids:
                raise ValueError(
                    f"shadow use {use_id!r} names no planned task; refusing"
                )
            if use.purpose == "coverage" and use.target_id not in target_ids:
                raise ValueError(
                    f"shadow coverage use {use_id!r} names no planned target"
                )
            record = self._stores.evidence.get(str(use.evidence_id))
            if record is None:
                raise ValueError(
                    f"shadow use {use_id!r} resolves to no isolated record"
                )
            hydrated.append(
                HydratedEvidence(
                    use_id=use.use_id,
                    evidence_id=use.evidence_id,
                    task_id=use.task_id,
                    purpose=use.purpose,
                    target_id=use.target_id,
                    content=record["content"],
                    role=None,
                    source_label="shadow-isolated",
                    source_identity=record["source"],
                    classification=record["classification"],
                    locator=None,
                )
            )
        return tuple(hydrated)

    async def hydrate_for_evaluation(
        self, use_refs: tuple[Any, ...], *, runtime: Any, plan: Any, bindings: Any
    ) -> tuple[HydratedEvidence, ...]:
        return self._resolve(tuple(use_refs), plan=plan, bindings=bindings)

    async def hydrate_for_synthesis(
        self,
        use_refs: tuple[Any, ...],
        *,
        runtime: Any,
        plan: Any,
        bindings: Any,
        budget: Any,
    ) -> tuple[HydratedEvidence, ...]:
        return tuple(
            item
            for item in self._resolve(tuple(use_refs), plan=plan, bindings=bindings)
            if item.purpose != "discovery"
        )


class ShadowLeaseRepo:
    """Isolated retention-lease repository (scheduler/binding lease path).

    Records acquisitions into the isolated ``leases`` log so the shared
    scheduler's lease-before-checkpoint path executes for real without
    touching ``revision_retention_leases``.
    """

    def __init__(self, stores: IsolatedShadowStores) -> None:
        self._stores = stores
        self.session = self._Session(stores)

    class _Session:
        def __init__(self, stores: IsolatedShadowStores) -> None:
            self._stores = stores

        async def commit(self) -> None:
            self._stores.leases.append({"event": "commit"})

    async def acquire_or_refresh(
        self, run_id: str, revision_id: Any, evidence_use_id: Any = None, **kwargs: Any
    ) -> SimpleNamespace:
        self._stores.leases.append(
            {
                "run_id": run_id,
                "revision_id": revision_id,
                "evidence_use_id": evidence_use_id,
            }
        )
        return SimpleNamespace(
            run_id=run_id, revision_id=revision_id, evidence_use_id=evidence_use_id
        )


class ShadowChatMessages:
    """Read-only chat-messages stub: revision-tracked reads, no writes.

    The shadow never resumes a clarification (it starts a fresh isolated
    thread), so any read misses fail closed with ``LookupError`` — the
    same contract as the production service's not-found path.
    """

    async def get_user_message(self, message_id: UUID) -> Any:
        raise LookupError(f"shadow has no chat message {message_id!r}")


class ShadowAuthorization:
    """Read-only authorization stub over the isolated document view.

    Allows only documents present in the isolated view whose workspace is
    in the current runtime scope; everything else raises
    ``PermissionError``. No production ACL store is consulted.
    """

    def __init__(self, document_view: Mapping[Any, Any]) -> None:
        self._view = dict(document_view)

    async def require_document(
        self, document_id: UUID, capability_runtime: Any
    ) -> None:
        if isinstance(document_id, str):
            document_id = UUID(document_id)
        scope = {item for item in (capability_runtime.workspace_ids or ())}
        entry = self._view.get(document_id)
        workspace_id = (entry or {}).get("workspace_id")
        if entry is None or (scope and workspace_id not in scope):
            raise PermissionError(
                f"document {document_id!r} is not in the shadow runtime scope"
            )


@dataclass
class ShadowBundle:
    """One shadow run: isolated saver + stores + read-only adapters + graph."""

    raw_query: str
    thread_id: str
    checkpoint_bundle: ShadowCheckpointBundle
    stores: IsolatedShadowStores
    source_adapters: tuple[ReadOnlySourceAdapter, ...]
    graph: Any = field(repr=False)
    runtime_context: Any = field(repr=False)
    initial_state: SupervisorV2State = field(repr=False)
    metrics: ShadowMetrics | None = field(default=None)

    def isolated_write_count(self) -> int:
        """Entries written to isolated stores (never production)."""
        return self.stores.write_count()

    async def run(self) -> ShadowMetrics:
        """Run one shadow turn; return redacted metrics, discard the rest.

        Cancellation (primary-run follow) propagates as
        ``asyncio.CancelledError``: the run holds no commit path, so a
        cancelled shadow persists nothing — not even to its isolated
        stores. No outbound emitter is invoked on any path. A suspended
        (clarify) turn reports ``status="clarify"`` without resuming.
        """
        started = time.monotonic()
        config = {"configurable": {"thread_id": self.thread_id}}
        result = await self.graph.ainvoke(
            dict(self.initial_state), config, context=self.runtime_context
        )
        duration_ms = int((time.monotonic() - started) * 1000)
        # Typed dependency gap (R66.1): the graph converts a node raise
        # into a generic error marker, so the adapter records the people-
        # source cause on itself; a recorded gap means THIS failure — the
        # turn necessarily short-circuited at the semantic node — and the
        # run reports unavailable/dependency-gap, never generic zero-task.
        gap_reason = getattr(
            self.runtime_context.services.semantic_adapter,
            "last_gap_reason",
            None,
        )
        if gap_reason is not None:
            self.metrics = ShadowMetrics(
                status="unavailable",
                route=None,
                evaluation=None,
                task_count=0,
                isolated_writes=self.isolated_write_count(),
                duration_ms=duration_ms,
                reason=f"dependency-gap: {gap_reason}",
            )
            logger.warning(
                "shadow v2 turn dependency-gap: %s", gap_reason,
            )
            return self.metrics
        decision = result.get("route_decision")
        if isinstance(decision, dict):
            route = decision.get("route")
        else:
            route = getattr(decision, "route", None)
        execution = result.get("execution")
        if isinstance(execution, dict):
            plan = execution.get("plan")
            evaluation = execution.get("evidence_evaluation")
        else:
            plan = getattr(execution, "plan", None)
            evaluation = getattr(execution, "evidence_evaluation", None)
        if isinstance(evaluation, dict):
            evaluation_status = evaluation.get("status")
        else:
            evaluation_status = getattr(evaluation, "status", None)
        if "__interrupt__" in result:
            status = "clarify"
        else:
            final = result.get("final_response")
            if isinstance(final, dict):
                status = final.get("status", "error")
            else:
                status = getattr(final, "status", "error")
        if plan is None:
            task_count = 0
        elif isinstance(plan, dict):
            task_count = len(plan.get("tasks", ()))
        else:
            task_count = len(getattr(plan, "tasks", ()))
        # Record only counts in the isolated stores (never content).
        self.stores.checkpoint[self.thread_id] = {
            "status": str(status),
            "task_count": task_count,
        }
        self.metrics = ShadowMetrics(
            status=str(status),
            route=route,
            evaluation=evaluation_status,
            task_count=task_count,
            isolated_writes=self.isolated_write_count(),
            duration_ms=duration_ms,
        )
        logger.info(
            "shadow v2 turn complete: status=%s route=%s eval=%s tasks=%d "
            "isolated_writes=%d duration_ms=%d",
            self.metrics.status,
            self.metrics.route,
            self.metrics.evaluation,
            task_count,
            self.metrics.isolated_writes,
            duration_ms,
        )
        return self.metrics


def build_shadow_bundle(
    *,
    raw_query: str,
    thread_id: str,
    user_id: UUID | None = None,
    workspace_ids: tuple[UUID, ...] = (),
    can_read_people: bool,
    allowed_capabilities: frozenset[str],
    person_names: tuple[str, ...] = (),
    known_documents: tuple[UUID, ...] = (),
    document_view: Mapping[Any, Mapping[str, Any]] | None = None,
    people_directory: Mapping[str, Mapping[str, object]] | None = None,
    people_source: Any | None = None,
    session_factory: Any | None = None,
    intent_classifier: Any | None = None,
    adaptive_planner: Any | None = None,
    adaptive_replanner: Any | None = None,
    history: tuple[tuple[str, str], ...] = (),
    deadline_seconds: float = 120.0,
) -> ShadowBundle:
    """Assemble one isolated shadow bundle (R54 ownership chain + R60).

    Fresh isolated saver + isolated conversation/evidence/use/audit stores
    + read-only source adapters
    -> ``create_supervisor_v2_graph(checkpointer=isolated_saver)``.

    The SAME ``create_supervisor_v2_graph`` topology as production — no
    alternate graph/agent. ``RuntimeServices`` is wired with the REAL
    capability registry (built by ``build_v2_capability_registry`` around
    the isolated read-only people directory + isolated evidence builder),
    a non-empty ``allowed_capabilities`` consistent with
    ``can_read_people``, a real future deadline, the isolated hydrator,
    lease repo, read-only chat/authorization stubs, and a fresh
    ``AnswerDraftChannel`` — so factual queries reach the SHARED
    ``TaskScheduler`` while every write lands in isolated stores.
    Document/section reads stay gated out and fail closed to typed
    outcomes. ``person_names``/``known_documents``/``history`` preserve the
    primary turn's references read-only instead of stripping them.
    ``can_read_people`` and ``allowed_capabilities`` are REQUIRED caller
    inputs mirroring the primary turn's runtime authorization (R64) — the
    shadow never hardcodes them. ``people_source`` defaults to the
    production people lookup wrapped read-only; an isolated directory
    (bundle-level/test input) takes precedence when supplied.
    ``citation_resolver`` is wired exactly like production ingress — a
    read-only ``StoreCitationResolver`` over ``session_factory`` (default
    ``async_session_maker``) — so grounded shadow synthesis reaches the
    single ``CitationProjector`` owner instead of failing closed at
    ``_require_projector`` after consuming a real provider call.
    """
    saver = create_shadow_checkpointer()
    assert is_shadow_saver(saver)
    namespace = f"shadow-{uuid4().hex[:12]}"
    stores = IsolatedShadowStores()
    view = dict(document_view or {})
    if people_source is None and people_directory is None:
        people_source = ReadOnlyProductionPeopleSource()
    directory = ShadowPeopleDirectory(dict(people_directory or {}))
    semantic_adapter = ShadowSemanticAdapter(
        raw_query,
        person_names=tuple(person_names),
        known_documents=tuple(known_documents),
        document_view=view,
        people_source=(
            directory if people_directory is not None else people_source
        ),
        # Mirrored primary authorization (R66.1): a denied turn never
        # consults the People source, even read-only.
        can_read_people=bool(can_read_people),
    )
    binding_resolver = ShadowBindingResolver(view)
    evidence_builder = ShadowEvidenceBuilder(stores)
    hydrator = ShadowEvidenceHydrator(stores)
    leases = ShadowLeaseRepo(stores)
    adapters = (
        ReadOnlySourceAdapter("conversation"),
        ReadOnlySourceAdapter("evidence"),
        ReadOnlySourceAdapter("documents", reader=directory),
        ReadOnlySourceAdapter("memory"),
    )
    graph = create_supervisor_v2_graph(checkpointer=saver)
    capability_runtime = CapabilityRuntimeContext(
        request_id=f"shadow-req-{uuid4().hex[:12]}",
        run_id=f"shadow-run-{uuid4().hex[:12]}",
        user_id=user_id or UUID("00000000-0000-0000-0000-000000000000"),
        workspace_ids=tuple(workspace_ids),
        # Mirrored from the primary turn by the caller (R64): the shadow
        # never decides authorization itself.
        can_read_people=bool(can_read_people),
        allowed_capabilities=frozenset(allowed_capabilities),
        deadline_at=datetime.now(UTC) + timedelta(seconds=deadline_seconds),
    )
    # The capability backing mirrors the adapter's source choice so the
    # confirmed reference and the dispatched lookup read the same source.
    # Document/section reads stay registry-gated and fail closed to typed
    # DEPENDENCY_UNAVAILABLE outcomes.
    people_backing = (
        directory if people_directory is not None else people_source
    )
    # I1 (round-2 review): mirror of production ingress — one
    # request-scoped resolver shared by the document capabilities (via
    # the registry) AND ``RuntimeServices.pinned_target_resolver``, so the
    # shared scheduler's dispatch-time feed succeeds and targeted
    # document/section turns fail closed at the registry gate with typed
    # DEPENDENCY_UNAVAILABLE outcomes (route preserved), never a
    # SchedulerError boundary error. Document/section reads stay gated out
    # by ``available_services={"v1-people"}``.
    shadow_resolver = PlanBindingResolver()
    registry = build_v2_capability_registry(
        capability_runtime,
        bundle=V1ServiceBundle(
            people_lookup=people_backing,
            evidence=evidence_builder,
            resolver=shadow_resolver,
        ),
        evidence=evidence_builder,
        resolver=shadow_resolver,
        available_services=frozenset({"v1-people"}),
    )
    if session_factory is None:
        from app.core.database import async_session_maker

        session_factory = async_session_maker
    services = RuntimeServices(
        retention_leases=leases,
        semantic_adapter=semantic_adapter,
        binding_resolver=binding_resolver,
        capability_registry=registry,
        chat_messages=ShadowChatMessages(),
        authorization=ShadowAuthorization(view),
        evidence_hydrator=hydrator,
        answer_draft_channel=AnswerDraftChannel(),
        answer_draft_builder=StructuredLLMDraftBuilder(
            get_main_provider_for_synthesis
        ),
        pinned_target_resolver=shadow_resolver,
        # Exactly one citation resolver per shadow run (spec §11.1),
        # mirroring production ingress (runtime_selector.py): the
        # read-only resolver over the authoritative stores behind the
        # single CitationProjector owner. Reads only — no write surface.
        citation_resolver=StoreCitationResolver(session_factory),
        # Decision-parity services (same classes production ingress wires
        # in runtime_selector.py): all three are proposal-only — they own no
        # checkpointed state, dispatch no capabilities, and their outputs
        # still pass the frozen validate/lease/checkpoint boundary. Without
        # them the shadow silently exercises only the deterministic
        # fallbacks, so adaptive-path turns (uncovered work types, model
        # replans) would diverge from production semantics. Callers may
        # inject deterministic stand-ins (tests) — ``None`` means the real
        # request-scoped service.
        intent_classifier=intent_classifier or IntentClassifier(),
        adaptive_planner=adaptive_planner or AdaptivePlanner(),
        adaptive_replanner=adaptive_replanner or AdaptiveReplanner(),
    )
    runtime_context = GraphRuntimeContext(
        capability_runtime=capability_runtime,
        services=services,
    )
    known_resources = tuple(
        KnownDocumentResource(
            resource_id=str(document_id),
            document_id=document_id,
            source="api_explicit",
        )
        for document_id in tuple(known_documents)
    )
    recent_turns = tuple(
        ConversationTurn(role=role, content=content)  # type: ignore[arg-type]
        for role, content in tuple(history)
    )
    initial_state = SupervisorV2State(
        contract_version=CONTRACT_VERSION,
        request=RequestContext(
            contract_version=CONTRACT_VERSION,
            request_id=capability_runtime.request_id,
            thread_id=thread_id,
            original_query=raw_query,
            known_documents=known_resources,
        ),
        conversation=ConversationContext(
            summary="",
            active_entities=(),
            last_focus=None,
            recent_turns=recent_turns,
        ),
        semantic=SemanticContext(
            contextualized_query="",
            normalized_query="",
            abbreviations=(),
            coreferences=(),
            document_refs=(),
            person_refs=(),
            section_refs=(),
            blocking_ambiguities=(),
        ),
        bindings=DocumentBindingSet(bindings=(), revision_requirement_refs=()),
        query_analysis=None,
        route_decision=None,
        intent_analysis=None,
        execution=ExecutionState(plan=None, task_results=(), evidence_evaluation=None),
        clarification=None,
        synthesis=None,
        final_response=None,
        checkpoint_schema_revision=CHECKPOINT_SCHEMA_REVISION,
        discovery_need=None,
        discovery=None,
        document_selection_clarification=None,
        research_target_selection=None,
    )
    return ShadowBundle(
        raw_query=raw_query,
        thread_id=f"{namespace}:{thread_id}",
        checkpoint_bundle=ShadowCheckpointBundle(saver=saver, namespace=namespace),
        stores=stores,
        source_adapters=adapters,
        graph=graph,
        runtime_context=runtime_context,
        initial_state=initial_state,
    )


async def _stop_shadow_task(task: asyncio.Task, *, timeout: float = 5.0) -> bool:
    """Cancel and join a shadow task, bounded end-to-end (R65/R66.4).

    Both cancel-awaits go through ``asyncio.shield`` + ``wait_for``: the
    shield keeps the join bounded on EVERY Python (on 3.11 a bare
    ``wait_for(task)`` re-awaits a cancel-swallowing task in
    ``_cancel_and_wait`` without bound — the M1 hang). First cancel is
    delivered and awaited up to ``timeout``; on expiry the task is
    cancelled AGAIN and awaited up to ``timeout`` once more. A shadow
    that survives the double cancel yields ``stopped=False`` — reachable
    by design — instead of blocking primary cleanup indefinitely.
    Returns True only when the task is done. Every outcome is swallowed
    so callers can use this in ``finally`` paths; callers MUST still act
    on a False return (fail the shadow closed and log) before
    proceeding. Total wall-time is bounded by ~2*timeout.
    """
    if task.done():
        return True
    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except TimeoutError:
        pass
    except (asyncio.CancelledError, Exception):
        return task.done()
    if task.done():
        return True
    # The shadow resisted the first cancel: cancel again and re-await
    # BOUNDED — a persistently resistant shadow reports False instead of
    # hanging the caller.
    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except TimeoutError:
        return False
    except (asyncio.CancelledError, Exception):
        return task.done()
    return task.done()


def should_run_shadow(*, percent: float, sample: float | None = None) -> bool:
    """Sample one turn for shadowing: ``sample`` in [0, 100) vs ``percent``.

    ``percent=0`` (the safe default) never shadows; ``percent=100``
    always shadows. A caller-supplied ``sample`` keeps tests
    deterministic; live callers draw ``random.uniform(0, 100)``.
    """
    if percent <= 0:
        return False
    if percent >= 100:
        return True
    value = random.uniform(0, 100) if sample is None else sample
    return value < percent


def shadow_sampling_enabled() -> bool:
    """True only when shadow wiring is enabled with a nonzero percent."""
    from app.core.config import settings

    return bool(
        getattr(settings, "NEXUSRAG_AGENT_V2_SHADOW_ENABLED", False)
    ) and float(getattr(settings, "NEXUSRAG_AGENT_V2_SHADOW_PERCENT", 0)) > 0
