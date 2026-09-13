"""Deterministic dependency-materialization package (Phase 3, Task 4).

Only deterministic server-side dependency adapters live here: they hydrate
governed evidence, extract an approved scalar, and build a concrete
``CapabilityInput`` BEFORE the dependent task is appended and checkpointed.
They never call a connector, never read a raw record, and never dispatch work
(dispatch belongs to the shared ``TaskScheduler`` alone).
"""

from .people_document import (
    PERSON_IDENTIFIER_FIELD,
    MaterializationError,
    PeopleDocumentMaterialization,
    append_materialized_dependent,
    build_dependent_search_task,
    extract_person_identifier,
    materialize_person_dependency,
)

__all__ = [
    "PERSON_IDENTIFIER_FIELD",
    "MaterializationError",
    "PeopleDocumentMaterialization",
    "append_materialized_dependent",
    "build_dependent_search_task",
    "extract_person_identifier",
    "materialize_person_dependency",
]
