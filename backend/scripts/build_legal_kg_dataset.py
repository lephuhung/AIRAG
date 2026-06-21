"""
Build a clean LegalKG fine-tuning dataset from the raw extraction logs.
=======================================================================

The LegalKG ingestion pipeline (when HRAG_KG_LOG_EXTRACTION=true) writes one
JSONL file per document to MinIO under:

    datasets/legal_kg_extraction/{YYYY-MM-DD}/ws_{workspace}_doc_{document}.jsonl

Each line is one LLM call in OpenAI chat format:
    {"messages": [{"role": "system"|"user"|"assistant", "content": ...}],
     "metadata": {"stage": "article_extract"|"preamble"|"resolve", "doc_type": ...}}

Raw logs contain failures (broken JSON, empty extractions, duplicates) that hurt
fine-tuning. This script downloads the raw logs, applies quality filters, and
emits a single de-duplicated train.jsonl ready for SFT.

Usage (run from backend/):
    python -m scripts.build_legal_kg_dataset --out data/legal_kg_train.jsonl
    python -m scripts.build_legal_kg_dataset --date 2026-06-21 --stage article_extract
    python -m scripts.build_legal_kg_dataset --min-entities 1 --upload
    python -m scripts.build_legal_kg_dataset --dry-run        # stats only, no write
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter

from app.services.storage_service import get_storage_service

PREFIX = "datasets/legal_kg_extraction/"

# Stages that are expected to produce {"entities": [...], "relations": [...]}.
# `preamble` produces {"can_cu_list": [...]}; `resolve` produces a mapping.
EXTRACTION_STAGES = {"article_extract"}


def _parse_assistant_json(raw: str) -> dict | None:
    """Mirror legal_kg_service._parse_llm_json: tolerate markdown code fences.

    Returns the parsed object, or None if it cannot be parsed at all.
    """
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                return None
    return None


async def _list_keys(storage, date: str | None) -> list[str]:
    prefix = PREFIX + (f"{date}/" if date else "")
    keys: list[str] = []
    async with storage._client() as s3:
        paginator = s3.get_paginator("list_objects_v2")
        async for page in paginator.paginate(
            Bucket=storage._bucket_uploads, Prefix=prefix
        ):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".jsonl"):
                    keys.append(obj["Key"])
    return keys


async def build(args) -> None:
    storage = get_storage_service()
    keys = await _list_keys(storage, args.date)
    if not keys:
        print(f"No log files found under {PREFIX}{args.date or ''}")
        return
    print(f"Found {len(keys)} log file(s).")

    stats = Counter()
    by_stage = Counter()
    seen: set[tuple] = set()  # dedup on (system, user, assistant)
    kept: list[dict] = []

    for key in keys:
        try:
            body = await storage.download_file(key)
        except Exception as e:
            print(f"  ! skip {key}: {e}")
            continue
        for line in body.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                stats["dropped_bad_line"] += 1
                continue

            messages = entry.get("messages", [])
            meta = entry.get("metadata", {})
            stage = meta.get("stage", "unknown")

            roles = {m.get("role") for m in messages}
            if not {"system", "user", "assistant"} <= roles:
                stats["dropped_incomplete_messages"] += 1
                continue

            if args.stage and stage != args.stage:
                stats["dropped_stage_filter"] += 1
                continue

            assistant = next(
                (m["content"] for m in messages if m.get("role") == "assistant"), ""
            )
            if not assistant.strip():
                stats["dropped_empty_response"] += 1
                continue

            # Only validate JSON for stages that must emit structured output.
            if stage in EXTRACTION_STAGES:
                parsed = _parse_assistant_json(assistant)
                if parsed is None:
                    stats["dropped_bad_json"] += 1
                    continue
                n_ent = len(parsed.get("entities", []) or [])
                n_rel = len(parsed.get("relations", []) or [])
                if (n_ent + n_rel) < args.min_entities:
                    stats["dropped_empty_extraction"] += 1
                    continue

            user = next(
                (m["content"] for m in messages if m.get("role") == "user"), ""
            )
            system = next(
                (m["content"] for m in messages if m.get("role") == "system"), ""
            )
            dedup_key = (system, user, assistant)
            if dedup_key in seen:
                stats["dropped_duplicate"] += 1
                continue
            seen.add(dedup_key)

            kept.append(entry)
            by_stage[stage] += 1
            stats["kept"] += 1

    print("\n=== Stats ===")
    for k, v in sorted(stats.items()):
        print(f"  {k:28s}: {v}")
    print("  --- kept by stage ---")
    for k, v in sorted(by_stage.items()):
        print(f"  {k:28s}: {v}")

    if args.dry_run:
        print("\n(dry-run) no file written.")
        return

    out_bytes = (
        "\n".join(json.dumps(e, ensure_ascii=False) for e in kept) + "\n"
    ).encode("utf-8")

    with open(args.out, "wb") as f:
        f.write(out_bytes)
    print(f"\nWrote {len(kept)} examples → {args.out}")

    if args.upload:
        up_key = f"datasets/legal_kg_clean/{args.out.split('/')[-1]}"
        await storage.ensure_uploads_bucket()
        await storage.upload_file(
            key=up_key, data=out_bytes, content_type="application/x-ndjson"
        )
        print(f"Uploaded clean dataset → MinIO {up_key}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build clean LegalKG SFT dataset")
    p.add_argument("--out", default="data/legal_kg_train.jsonl",
                   help="local output path for the cleaned JSONL")
    p.add_argument("--date", default=None,
                   help="only process logs from this date (YYYY-MM-DD)")
    p.add_argument("--stage", default=None,
                   help="keep only this stage (article_extract|preamble|resolve)")
    p.add_argument("--min-entities", type=int, default=1,
                   help="min entities+relations to keep an extraction example")
    p.add_argument("--upload", action="store_true",
                   help="upload the cleaned dataset back to MinIO")
    p.add_argument("--dry-run", action="store_true",
                   help="print stats only, do not write any file")
    args = p.parse_args()
    asyncio.run(build(args))


if __name__ == "__main__":
    main()
