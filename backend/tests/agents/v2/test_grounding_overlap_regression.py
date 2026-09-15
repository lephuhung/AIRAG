"""RED regression: overlapping targetless retrieval must ground (Task 4).

Production defect: targetless document retrieval returned overlapping legal
chunks; build_extractive_draft made one claim per whole chunk, so grounding
split the draft into assertions and _mapping_report saw every repeated
marker / duplicated sentence as ambiguous (tiny '1.' contained in every
longer claim, duplicated penalty sentence in two claims) and raised
GroundingInsufficient despite sufficient evaluation + admitted hydration.

This test drives the real build_extractive_draft -> ground_answer path over
realistic overlapping evidence. It reproduces the pre-fix
GroundingInsufficient and pins the corrected behavior plus the fail-closed
properties that must remain intact.
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from app.services.agents.v2.contracts.evidence import DocumentSourceIdentity
from app.services.agents.v2.contracts.locators import DocumentLocator
from app.services.agents.v2.contracts.synthesis import (
    AnswerClaim,
    AnswerDraft,
    SynthesisEvidence,
)
from app.services.agents.v2.nodes.evaluate import HydratedEvidence
from app.services.agents.v2.nodes.grounding import (
    GroundingInsufficient,
    ground_answer,
)
from app.services.agents.v2.nodes.synthesize import build_extractive_draft

DOCUMENT_ID = uuid4()
REVISION = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

CHUNK_1 = "1. Phạm vi áp dụng. Điều 5 quy định mức phạt là 5 triệu đồng."
CHUNK_2 = "2. Đối tượng áp dụng. Điều 5 quy định mức phạt là 5 triệu đồng."
CHUNK_3 = "1. Phạm vi áp dụng. Mức phạt bổ sung là 2 triệu đồng."


def _hydrated(use_id, evidence_id, content):
    return HydratedEvidence(
        use_id=use_id,
        evidence_id=evidence_id,
        task_id="T1",
        purpose="supporting",
        target_id=None,
        content=content,
        role=None,
        source_label="doc",
        source_identity=DocumentSourceIdentity(
            kind="document",
            document_id=DOCUMENT_ID,
            document_revision=REVISION,
            locator=DocumentLocator(kind="document"),
        ),
        classification="normal",
        locator=DocumentLocator(kind="document"),
        document_revision=REVISION,
    )


def _overlapping_fixture():
    u1, u2, u3 = uuid4(), uuid4(), uuid4()
    e1, e2, e3 = uuid4(), uuid4(), uuid4()
    hydrated = (
        _hydrated(u1, e1, CHUNK_1),
        _hydrated(u2, e2, CHUNK_2),
        _hydrated(u3, e3, CHUNK_3),
    )
    projected = tuple(
        SynthesisEvidence(
            use_id=h.use_id,
            content=h.content,
            role=h.role,
            target_id=h.target_id,
            source_label=h.source_label,
        )
        for h in hydrated
    )
    return hydrated, projected, (u1, u2, u3)


@pytest.mark.asyncio
async def test_overlapping_chunks_ground_through_extractive_path() -> None:
    """Overlapping chunks (repeated markers + duplicated sentence) must ground."""
    hydrated, projected, (u1, u2, u3) = _overlapping_fixture()
    draft = build_extractive_draft(projected)
    grounded = await ground_answer(draft=draft, evidence=hydrated)
    assert grounded.draft == draft
    assert grounded.admitted_use_ids == frozenset({u1, u2, u3})
    # Claim granularity must not reformat the prior whole-chunk answer text.
    assert draft.content == f"{CHUNK_1}\n\n{CHUNK_2}\n\n{CHUNK_3}"
    # The duplicated penalty sentence merges both supporting uses, in order.
    penalty = next(
        c for c in draft.claims if "5 triệu" in c.text
    )
    assert penalty.evidence_use_ids == (u1, u2)
    # The repeated marker/scope merges first-seen order too.
    scope = next(
        c for c in draft.claims if "Phạm vi" in c.text
    )
    assert scope.evidence_use_ids == (u1, u3)
    # Deterministic first-seen assertion order and use order own the full shape.
    assert [(c.claim_id, c.text, c.evidence_use_ids) for c in draft.claims] == [
        ("claim-1", "1.", (u1, u3)),
        ("claim-2", "Phạm vi áp dụng.", (u1, u3)),
        ("claim-3", "Điều 5 quy định mức phạt là 5 triệu đồng.", (u1, u2)),
        ("claim-4", "2.", (u2,)),
        ("claim-5", "Đối tượng áp dụng.", (u2,)),
        ("claim-6", "Mức phạt bổ sung là 2 triệu đồng.", (u3,)),
    ]


@pytest.mark.asyncio
async def test_unmapped_draft_remains_fail_closed() -> None:
    """An assertion with no supporting claim still raises (no silent pass)."""
    hydrated, projected, _ = _overlapping_fixture()
    draft = build_extractive_draft(projected)
    bad = AnswerDraft(
        content=draft.content + " Câu này không có căn cứ.",
        claims=draft.claims,
    )
    with pytest.raises(GroundingInsufficient) as exc_info:
        await ground_answer(draft=bad, evidence=hydrated)
    assert len(exc_info.value.report.unmapped_assertions) == 1


@pytest.mark.asyncio
async def test_genuinely_ambiguous_containment_remains_fail_closed() -> None:
    """Containment in two longer claims with no exact match still raises."""
    u1, u2 = uuid4(), uuid4()
    draft = AnswerDraft(
        content="Mức phạt áp dụng.",
        claims=(
            AnswerClaim(
                claim_id="claim-1",
                text="Mức phạt áp dụng cho điều 5.",
                evidence_use_ids=(u1,),
            ),
            AnswerClaim(
                claim_id="claim-2",
                text="Mức phạt áp dụng cho điều 6.",
                evidence_use_ids=(u2,),
            ),
        ),
    )
    evidence = (
        _hydrated(u1, uuid4(), "Mức phạt áp dụng cho điều 5."),
        _hydrated(u2, uuid4(), "Mức phạt áp dụng cho điều 6."),
    )
    with pytest.raises(GroundingInsufficient) as exc_info:
        await ground_answer(draft=draft, evidence=evidence)
    assert len(exc_info.value.report.ambiguous_assertions) == 1
