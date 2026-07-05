"""
Golden-set IR metrics for the hybrid retriever (``HRAGService.query_deep``).

Runs every case in ``datasets/golden_retrieval.yaml`` against the LIVE stack
(PostgreSQL + ChromaDB + BM25 + reranker + KG), records the rank of the
expected document and expected "Điều" in the top-k, then aggregates:

    doc_recall@4 / doc_recall@8 / doc_MRR      (over positive cases)
    art_recall@4 / art_recall@8 / art_MRR      (over cases with expect_article)

A JSON report is written to ``tests/retrieval/reports/recall_<ts>.json`` so
two runs can be diffed (same idea as tests/prompts reports). The aggregate
test soft-gates against ``baseline_recall.json`` (committed): the run fails
only if doc/art recall@8 drops more than EPSILON below the baseline — so the
suite is a regression net, not a fixed bar.

Needs the live stack — run inside the backend container:

    docker exec -e RETRIEVAL_EVAL=1 -w /app/backend hrag-backend \
        pytest tests/retrieval/test_recall.py -q

Per-case tests never fail on retrieval quality (only on infrastructure
errors); quality is judged in the aggregate test at the end of the module.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from sqlalchemy import text

pytestmark = pytest.mark.skipif(
    os.getenv("RETRIEVAL_EVAL") != "1",
    reason="set RETRIEVAL_EVAL=1 to run live retrieval golden-set evaluation",
)

_HERE = Path(__file__).parent
_DATASET = _HERE / "datasets" / "golden_retrieval.yaml"
_BASELINE = _HERE / "baseline_recall.json"
_REPORT_DIR = _HERE / "reports"

# Tolerated drop vs baseline before the aggregate test fails.
EPSILON = 0.05

_spec = yaml.safe_load(_DATASET.read_text(encoding="utf-8"))
DEFAULTS = _spec.get("defaults") or {}
CASES = _spec["cases"]
TOP_K = int(DEFAULTS.get("top_k", 8))
MODE = DEFAULTS.get("mode", "hybrid")

# Filled by the per-case tests, consumed by test_zz_aggregate_and_report.
RESULTS: list[dict] = []


async def _resolve_doc(pattern: str) -> tuple[str, str] | None:
    """document_number ILIKE pattern → (document_id, workspace_id), embed done."""
    from app.core.database import async_session_maker

    async with async_session_maker() as db:
        row = (await db.execute(
            text(
                "SELECT id, workspace_id FROM documents "
                "WHERE document_number ILIKE :pat AND embed_done = true LIMIT 1"
            ),
            {"pat": pattern},
        )).fetchone()
    return (str(row[0]), str(row[1])) if row else None


async def _resolve_in_workspace(pattern: str, workspace_id: str) -> str | None:
    from app.core.database import async_session_maker

    async with async_session_maker() as db:
        row = (await db.execute(
            text(
                "SELECT id FROM documents "
                "WHERE document_number ILIKE :pat AND workspace_id = :ws "
                "AND embed_done = true LIMIT 1"
            ),
            {"pat": pattern, "ws": workspace_id},
        )).fetchone()
    return str(row[0]) if row else None


def _article_matches(chunk, article_nos: list[int]) -> bool:
    """Chunk covers "Điều N"? Checked on heading_path (backfilled corpus-wide)
    then on a markdown article header in the content (OCR path keeps it)."""
    hp = " > ".join(str(c) for c in (chunk.heading_path or []))
    content = chunk.content or ""
    for n in article_nos:
        if re.search(rf"\bĐiều\s+{n}\b(?!\d)", hp):
            return True
        if re.search(rf"(?im)^#{{0,6}}\s*Điều\s+{n}\s*[.:]", content):
            return True
    return False


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
async def test_case_retrieval(case):
    from app.core.database import async_session_maker
    from app.services.retrieval.hrag_service import HRAGService

    primary = await _resolve_doc(case["expect_document"])
    if not primary:
        pytest.skip(f"no embedded document matching {case['expect_document']!r}")
    primary_id, ws_id = primary

    accepted_ids = {primary_id}
    for pat in case.get("accept_documents") or []:
        extra = await _resolve_in_workspace(pat, ws_id)
        if extra:
            accepted_ids.add(extra)

    t0 = time.monotonic()
    async with async_session_maker() as db:
        service = HRAGService(db, uuid.UUID(ws_id))
        # include_images=False: images/tables don't affect chunk ranking and
        # would only add DB round-trips per case.
        result = await service.query_deep(
            case["query"], top_k=TOP_K, mode=MODE, include_images=False
        )
    latency_ms = int((time.monotonic() - t0) * 1000)

    chunks = result.chunks or []
    row: dict = {
        "id": case["id"],
        "query": case["query"],
        "tags": case.get("tags") or [],
        "negative": bool(case.get("negative")),
        "n_chunks": len(chunks),
        "top_score": round(max((c.score for c in chunks), default=0.0), 4),
        "latency_ms": latency_ms,
    }

    if not case.get("negative"):
        doc_rank = next(
            (i for i, c in enumerate(chunks, 1) if str(c.document_id) in accepted_ids),
            None,
        )
        row["doc_rank"] = doc_rank
        articles = case.get("expect_article")
        if articles:
            row["art_rank"] = next(
                (
                    i
                    for i, c in enumerate(chunks, 1)
                    if str(c.document_id) == primary_id
                    and _article_matches(c, articles)
                ),
                None,
            )
    RESULTS.append(row)


def _aggregate(rows: list[dict]) -> dict:
    def recall_at(key: str, k: int, pool: list[dict]) -> float | None:
        if not pool:
            return None
        return round(
            sum(1 for r in pool if r.get(key) is not None and r[key] <= k) / len(pool), 4
        )

    def mrr(key: str, pool: list[dict]) -> float | None:
        if not pool:
            return None
        return round(
            sum(1 / r[key] for r in pool if r.get(key) is not None) / len(pool), 4
        )

    positives = [r for r in rows if not r["negative"]]
    with_art = [r for r in positives if "art_rank" in r]
    negatives = [r for r in rows if r["negative"]]
    return {
        "n_cases": len(rows),
        "n_positive": len(positives),
        "n_with_article": len(with_art),
        "doc_recall_at_4": recall_at("doc_rank", 4, positives),
        "doc_recall_at_8": recall_at("doc_rank", 8, positives),
        "doc_mrr": mrr("doc_rank", positives),
        "art_recall_at_4": recall_at("art_rank", 4, with_art),
        "art_recall_at_8": recall_at("art_rank", 8, with_art),
        "art_mrr": mrr("art_rank", with_art),
        "neg_avg_top_score": round(
            sum(r["top_score"] for r in negatives) / len(negatives), 4
        ) if negatives else None,
    }


def _config_snapshot() -> dict:
    from app.core.config import settings

    keys = [
        "HRAG_ENABLE_BM25",
        "HRAG_BM25_WORD_SEGMENT",
        "HRAG_RRF_K",
        "HRAG_MIN_RELEVANCE_SCORE",
        "HRAG_RERANKER_TOP_K",
        "HRAG_RECENTNESS_BOOST",
        "HRAG_ENABLE_CONTEXTUAL_EMBEDDINGS",
        "HRAG_KG_MODE",
    ]
    return {k: getattr(settings, k, None) for k in keys}


async def test_zz_aggregate_and_report():
    """Runs LAST (definition order): aggregate, persist report, gate vs baseline."""
    if not RESULTS:
        pytest.skip("no per-case results collected (all cases skipped?)")

    agg = _aggregate(RESULTS)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": _DATASET.name,
        "top_k": TOP_K,
        "mode": MODE,
        "config": _config_snapshot(),
        "aggregate": agg,
        "cases": RESULTS,
    }
    _REPORT_DIR.mkdir(exist_ok=True)
    out = _REPORT_DIR / f"recall_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["", f"golden retrieval — {agg['n_positive']} positive cases (top_k={TOP_K})"]
    for k in ("doc_recall_at_4", "doc_recall_at_8", "doc_mrr",
              "art_recall_at_4", "art_recall_at_8", "art_mrr", "neg_avg_top_score"):
        lines.append(f"  {k:<20} {agg[k]}")
    lines.append(f"  report → {out}")
    print("\n".join(lines))

    misses = [r["id"] for r in RESULTS
              if not r["negative"] and (r.get("doc_rank") is None or r.get("art_rank", 1) is None)]
    if misses:
        print(f"  misses (doc or art not in top-{TOP_K}): {', '.join(misses)}")

    if _BASELINE.exists():
        base = json.loads(_BASELINE.read_text(encoding="utf-8"))["aggregate"]
        for key in ("doc_recall_at_8", "art_recall_at_8"):
            if base.get(key) is None or agg.get(key) is None:
                continue
            assert agg[key] >= base[key] - EPSILON, (
                f"{key} regressed: {agg[key]} < baseline {base[key]} - {EPSILON} "
                f"(baseline: {_BASELINE.name})"
            )
