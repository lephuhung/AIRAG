"""Async HTTP client for the PII extraction microservice."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


async def extract_pii(text: str) -> dict[str, list[str]] | None:
    """Call the PII extraction service to extract entities from text.

    Returns a dict like:
        {
            "phone_number": ["0912345678"],
            "id_number": ["012345678901"],
            "human_name": ["Nguyễn Văn A"],
            ...
        }
    Returns None if the service is not configured or the call fails.
    """
    from app.core.config import settings

    url = settings.PII_SERVICE_URL
    if not url:
        logger.debug("[pii_client] PII_SERVICE_URL not set, skipping extraction")
        return None

    import httpx

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{url}/extract",
                json={"text": text},
                timeout=settings.PII_SERVICE_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            entities = data.get("entities", {})
            logger.info("[pii_client] extracted entities: %s", entities)
            return entities
    except Exception as e:
        logger.warning("[pii_client] PII extraction failed: %s", e)
        return None
