"""
Storage Service — MinIO (S3-compatible) object storage for markdown content
and raw uploaded files.

Markdown bucket: hrag-markdown  (key: kb_{workspace_id}/doc_{document_id}.md)
Uploads bucket:  hrag-uploads   (key: kb_{workspace_id}/doc_{document_id}.{ext})
"""
from __future__ import annotations

import logging
from io import BytesIO

import aioboto3
from botocore.exceptions import ClientError

from app.core.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """Async MinIO client using aioboto3."""

    def __init__(self) -> None:
        self._session = aioboto3.Session()
        self._endpoint_url = settings.MINIO_ENDPOINT
        self._access_key = settings.MINIO_ACCESS_KEY
        self._secret_key = settings.MINIO_SECRET_KEY
        self._bucket = settings.MINIO_BUCKET_MARKDOWN
        self._bucket_uploads = settings.MINIO_BUCKET_UPLOADS

    def _client(self):
        """Return a context-managed S3 client."""
        return self._session.client(
            "s3",
            endpoint_url=self._endpoint_url,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            use_ssl=settings.MINIO_SECURE,
        )

    @staticmethod
    def _make_key(workspace_id: int, document_id: int) -> str:
        return f"kb_{workspace_id}/doc_{document_id}.md"

    @staticmethod
    def _make_upload_key(workspace_id: int, document_id: int, ext: str) -> str:
        """Return the upload key for a raw file. ext must include leading dot."""
        # Ensure ext has a leading dot
        if ext and not ext.startswith("."):
            ext = f".{ext}"
        return f"kb_{workspace_id}/doc_{document_id}{ext}"

    async def ensure_bucket(self) -> None:
        """Create the markdown bucket if it doesn't exist (idempotent)."""
        async with self._client() as s3:
            try:
                await s3.head_bucket(Bucket=self._bucket)
                logger.debug(f"[storage] bucket '{self._bucket}' already exists")
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code in ("404", "NoSuchBucket"):
                    await s3.create_bucket(Bucket=self._bucket)
                    logger.info(f"[storage] created bucket '{self._bucket}'")
                else:
                    logger.error(f"[storage] bucket check failed: {e}")
                    raise

    async def ensure_uploads_bucket(self) -> None:
        """Create the uploads bucket if it doesn't exist (idempotent)."""
        async with self._client() as s3:
            try:
                await s3.head_bucket(Bucket=self._bucket_uploads)
                logger.debug(f"[storage] bucket '{self._bucket_uploads}' already exists")
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code in ("404", "NoSuchBucket"):
                    await s3.create_bucket(Bucket=self._bucket_uploads)
                    logger.info(f"[storage] created bucket '{self._bucket_uploads}'")
                else:
                    logger.error(f"[storage] uploads bucket check failed: {e}")
                    raise

    async def upload_markdown(
        self,
        workspace_id: int,
        document_id: int,
        content: str,
    ) -> str:
        """Upload markdown text to MinIO. Returns the object key."""
        key = self._make_key(workspace_id, document_id)
        body = content.encode("utf-8")
        async with self._client() as s3:
            await s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=BytesIO(body),
                ContentType="text/markdown; charset=utf-8",
                ContentLength=len(body),
            )
        logger.debug(f"[storage] uploaded {key} ({len(body)} bytes)")
        return key

    async def download_markdown(self, key: str) -> str:
        """Download and return markdown text from MinIO."""
        async with self._client() as s3:
            response = await s3.get_object(Bucket=self._bucket, Key=key)
            body = await response["Body"].read()
        return body.decode("utf-8")

    async def delete_markdown(self, key: str) -> None:
        """Delete a markdown object from MinIO (no-op if not found)."""
        async with self._client() as s3:
            try:
                await s3.delete_object(Bucket=self._bucket, Key=key)
                logger.debug(f"[storage] deleted {key}")
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code not in ("404", "NoSuchKey"):
                    raise

    # ------------------------------------------------------------------
    # Raw file methods (hrag-uploads bucket)
    # ------------------------------------------------------------------

    async def upload_file(self, key: str, data: bytes, content_type: str) -> str:
        """Upload raw file bytes to the uploads bucket. Returns the key."""
        async with self._client() as s3:
            await s3.put_object(
                Bucket=self._bucket_uploads,
                Key=key,
                Body=BytesIO(data),
                ContentType=content_type,
                ContentLength=len(data),
            )
        logger.debug(f"[storage] uploaded raw file {key} ({len(data)} bytes)")
        return key

    async def download_file(self, key: str) -> bytes:
        """Download raw file bytes from the uploads bucket."""
        async with self._client() as s3:
            response = await s3.get_object(Bucket=self._bucket_uploads, Key=key)
            body = await response["Body"].read()
        logger.debug(f"[storage] downloaded raw file {key} ({len(body)} bytes)")
        return body

    async def delete_file(self, key: str) -> None:
        """Delete a raw file from the uploads bucket (no-op if not found)."""
        async with self._client() as s3:
            try:
                await s3.delete_object(Bucket=self._bucket_uploads, Key=key)
                logger.debug(f"[storage] deleted raw file {key}")
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code not in ("404", "NoSuchKey"):
                    raise

    async def generate_presigned_upload_url(
        self,
        key: str,
        content_type: str,
        expires_in: int = 900,
    ) -> str:
        """Generate a presigned PUT URL so the client can upload directly to MinIO.

        The URL is generated against ``MINIO_ENDPOINT`` (internal) then the
        hostname is rewritten to ``MINIO_PUBLIC_ENDPOINT`` so the browser can
        reach MinIO directly.  If ``MINIO_PUBLIC_ENDPOINT`` is empty it falls
        back to ``MINIO_ENDPOINT``.

        Args:
            key: Object key in the uploads bucket.
            content_type: MIME type of the file (set as Content-Type on PUT).
            expires_in: URL expiry in seconds (default 15 min).

        Returns:
            Presigned URL string reachable from the browser.
        """
        async with self._client() as s3:
            url: str = await s3.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": self._bucket_uploads,
                    "Key": key,
                    "ContentType": content_type,
                },
                ExpiresIn=expires_in,
            )

        # Rewrite internal hostname → public hostname so the browser can PUT
        public = settings.MINIO_PUBLIC_ENDPOINT.rstrip("/")
        internal = settings.MINIO_ENDPOINT.rstrip("/")
        if public and public != internal and url.startswith(internal):
            url = public + url[len(internal):]

        logger.debug(f"[storage] generated presigned PUT URL for {key} (expires_in={expires_in}s)")
        return url

    async def object_exists(self, key: str) -> bool:
        """Return True if the given key exists in the uploads bucket."""
        async with self._client() as s3:
            try:
                await s3.head_object(Bucket=self._bucket_uploads, Key=key)
                return True
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code in ("404", "NoSuchKey"):
                    return False
                raise

    # ------------------------------------------------------------------
    # Avatar methods (hrag-uploads bucket, avatars/ prefix)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_avatar_key(user_id: int, ext: str) -> str:
        """Key pattern: avatars/user_{id}.{ext}"""
        if ext and not ext.startswith("."):
            ext = f".{ext}"
        return f"avatars/user_{user_id}{ext}"

    async def upload_avatar(self, user_id: int, data: bytes, content_type: str, ext: str) -> str:
        """Upload avatar image to MinIO. Returns a public URL (presigned GET, 1 year).

        Stores under ``hrag-uploads`` at key ``avatars/user_{id}.{ext}``.
        Returns a public-style URL by generating a long-lived presigned GET URL.
        """
        key = self._make_avatar_key(user_id, ext)
        async with self._client() as s3:
            await s3.put_object(
                Bucket=self._bucket_uploads,
                Key=key,
                Body=BytesIO(data),
                ContentType=content_type,
                ContentLength=len(data),
            )
            # Generate a long-lived presigned GET URL (1 year)
            url: str = await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket_uploads, "Key": key},
                ExpiresIn=365 * 24 * 3600,
            )

        # Rewrite internal hostname → public hostname
        public = settings.MINIO_PUBLIC_ENDPOINT.rstrip("/")
        internal = settings.MINIO_ENDPOINT.rstrip("/")
        if public and public != internal and url.startswith(internal):
            url = public + url[len(internal):]

        logger.debug(f"[storage] uploaded avatar for user {user_id} at key={key}")
        return url

    async def delete_avatar(self, user_id: int, ext: str) -> None:
        """Delete an existing avatar from MinIO (no-op if not found)."""
        key = self._make_avatar_key(user_id, ext)
        async with self._client() as s3:
            try:
                await s3.delete_object(Bucket=self._bucket_uploads, Key=key)
                logger.debug(f"[storage] deleted avatar for user {user_id}")
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code not in ("404", "NoSuchKey"):
                    raise


# Module-level singleton
_storage_service: StorageService | None = None


def get_storage_service() -> StorageService:
    """Return the module-level StorageService singleton."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
