from app.models.knowledge_base import KnowledgeBase
from app.models.abbreviation import Abbreviation
from app.models.document_type import DocumentType
from app.models.document import Document, DocumentImage, DocumentTable
from app.models.chat_session import ChatSession
from app.models.chat_message import ChatMessage
from app.models.user import User
from app.models.tenant import Tenant, TenantUser
from app.models.invite_token import InviteToken
from app.models.exchange_summary import ExchangeSummary
from app.models.chat_file import ChatFile
from app.models.format_metadata import FormatMetadata
from app.models.integration import (
    ApiKey,
    TelegramBotConfig,
    TelegramLink,
    TelegramLinkCode,
)
from app.models.audit_log import AuditLog
from app.models.agent_trace import AgentTrace
from app.models.system_setting import SystemSetting
from app.models.document_alias import DocumentAlias

# ── Phase 1B — post-migration v2 ORM registration ─────────────────────────
# Importing ``v2_registry`` registers the 11 v2 ORM classes on
# ``Base.metadata`` and exposes the post-migration entrypoints used by
# ``app.main.lifespan`` (``LEGACY_STARTUP_TABLES``,
# ``assert_v2_readiness``). The lifespan function refuses to start
# the app unless the live database is at v2 schema version 1; this
# module performs the metadata registration only (no DB writes, no DDL).
#
# The model classes imported below are the 11 v2 tables from
# ``V2_SCHEMA_V1_TABLES`` (set in
# ``app.services.agents.v2.persistence.migrate``). ``v2_schema_version``
# is intentionally NOT mapped because it has no per-row semantics for
# application code — it is a migration-control surface.
from app.models.v2_registry import (  # noqa: E402,F401
    LEGACY_STARTUP_TABLES,
    assert_v2_readiness,
    register_v2_models,
)
from app.models.agent_rollout_control import AgentRolloutControl  # noqa: E402,F401
from app.models.agent_rollout_metric import AgentRolloutMetric  # noqa: E402,F401
from app.models.document_revision import DocumentRevision  # noqa: E402,F401
from app.models.document_revision_build import (  # noqa: E402,F401
    DocumentRevisionBuild,
)
from app.models.document_revision_chunk import (  # noqa: E402,F401
    DocumentRevisionChunk,
)
from app.models.document_ingestion_attempt import (  # noqa: E402,F401
    DocumentIngestionAttempt,
)
from app.models.source_arrival import SourceArrival  # noqa: E402,F401
from app.models.revision_retention_lease import (  # noqa: E402,F401
    RevisionRetentionLease,
)
from app.models.conversation_snapshot import (  # noqa: E402,F401
    ConversationSnapshot,
)
from app.models.semantic_snapshot import SemanticSnapshot  # noqa: E402,F401
from app.models.binding_audit import BindingAudit  # noqa: E402,F401
from app.models.evidence_record import EvidenceRecord  # noqa: E402,F401
from app.models.evidence_use import EvidenceUse  # noqa: E402,F401

__all__ = [
    "KnowledgeBase",
    "DocumentType",
    "Document",
    "DocumentImage",
    "DocumentTable",
    "ChatSession",
    "ChatMessage",
    "User",
    "Tenant",
    "TenantUser",
    "InviteToken",
    "Abbreviation",
    "ExchangeSummary",
    "ChatFile",
    "FormatMetadata",
    "ApiKey",
    "TelegramBotConfig",
    "TelegramLink",
    "TelegramLinkCode",
    "AuditLog",
    "AgentTrace",
    "SystemSetting",
    # Phase 3 Task 7B rollout tables (migration-owned; mapped, never created)
    "AgentRolloutControl",
    "AgentRolloutMetric",
    # Phase 1B v2 models
    "DocumentRevision",
    "DocumentRevisionBuild",
    "DocumentRevisionChunk",
    "DocumentIngestionAttempt",
    "SourceArrival",
    "RevisionRetentionLease",
    "ConversationSnapshot",
    "SemanticSnapshot",
    "BindingAudit",
    "EvidenceRecord",
    "EvidenceUse",
    "LEGACY_STARTUP_TABLES",
    "assert_v2_readiness",
    "register_v2_models",
]
