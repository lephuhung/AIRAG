"""
LLM Logger Service
==================
Logs LLM requests and responses into JSONL format and uploads directly to MinIO.
Useful for collecting fine-tuning datasets from relation extraction.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from app.services.storage_service import get_storage_service

logger = logging.getLogger(__name__)


class MinIOLoggerService:
    def __init__(self):
        self._logs: list[dict] = []

    def log_llm_call(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        response: str,
        model: str,
        metadata_extra: Optional[dict] = None
    ) -> None:
        """Buffer a single LLM request/response interaction in OpenAI Chat format."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": response})

        meta = {
            "model_used": model,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        if metadata_extra:
            meta.update(metadata_extra)

        log_entry = {
            "messages": messages,
            "metadata": meta,
        }
        self._logs.append(log_entry)

    async def flush_to_minio(self, workspace_id: int, document_id: int) -> None:
        """Upload all buffered logs to MinIO under datasets/relation_extraction."""
        if not self._logs:
            return

        jsonl_content = "\n".join(
            json.dumps(log, ensure_ascii=False) for log in self._logs
        )
        jsonl_bytes = jsonl_content.encode("utf-8")

        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        key = f"datasets/relation_extraction/{date_str}/ws_{workspace_id}_doc_{document_id}.jsonl"

        storage = get_storage_service()
        try:
            # Reusing the existing uploads bucket, which allows us to save arbitrary files
            await storage.ensure_uploads_bucket()
            await storage.upload_file(
                key=key,
                data=jsonl_bytes,
                content_type="application/x-ndjson"
            )
            logger.info(f"Flushed {len(self._logs)} LLM logs to MinIO key: {key}")
        except Exception as e:
            logger.error(f"Failed to flush LLM logs to MinIO for doc {document_id}: {e}")
        finally:
            self._logs.clear()
