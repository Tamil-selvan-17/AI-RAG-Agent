"""
Qdrant service.

Manages the vector collection lifecycle and read/write operations:
- dynamic collection creation
- upserting document chunk vectors with full metadata payload
- similarity search (top-k retrieval)
- document listing / deletion by document_id
"""

import uuid

from fastapi import HTTPException, status
from qdrant_client import AsyncQdrantClient, models

from app.core.config import get_settings
from app.core.logging import logger
from app.models.document import DocumentChunk


class QdrantService:
    """Async wrapper around the Qdrant client for the document collection."""

    def __init__(self) -> None:
        settings = get_settings()
        self.collection_name = settings.qdrant_collection_name
        self.vector_size = settings.effective_vector_size

        if settings.qdrant_url:
            # Qdrant Cloud (or any hosted instance reachable by URL + API key).
            self._client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
        else:
            # Local Qdrant running in Docker on this machine.
            self._client = AsyncQdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                grpc_port=settings.qdrant_grpc_port,
                prefer_grpc=False,
            )

    async def ensure_collection(self) -> None:
        """Create the collection dynamically if it does not already exist."""
        try:
            exists = await self._client.collection_exists(self.collection_name)
            if not exists:
                await self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
                await self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                logger.info(f"Created Qdrant collection '{self.collection_name}'")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to ensure Qdrant collection: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Cannot reach Qdrant vector database: {exc}",
            ) from exc

    async def upsert_chunks(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        """Store chunk vectors and metadata payloads in Qdrant."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        if not chunks:
            return

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_OID, chunk.chunk_id)),
                vector=vector,
                payload={
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "chunk_id": chunk.chunk_id,
                    "chunk_text": chunk.chunk_text,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "metadata": chunk.metadata,
                    "created_date": chunk.created_date.isoformat(),
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        try:
            await self._client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Upserted {len(points)} chunk vectors into Qdrant")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to upsert vectors into Qdrant: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to store vectors in Qdrant: {exc}",
            ) from exc

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
        document_id: str | None = None,
    ) -> list[dict]:
        """Perform a similarity search and return matching chunk payloads with scores."""
        query_filter = None
        if document_id:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
            )

        try:
            results = await self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Qdrant search failed: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Vector search failed: {exc}",
            ) from exc

        return [
            {
                "score": point.score,
                **point.payload,
            }
            for point in results.points
        ]

    async def delete_document(self, document_id: str) -> None:
        """Delete all vectors belonging to a document."""
        try:
            await self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id", match=models.MatchValue(value=document_id)
                            )
                        ]
                    )
                ),
            )
            logger.info(f"Deleted vectors for document_id={document_id}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to delete document vectors: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to delete document vectors: {exc}",
            ) from exc

    async def wipe_collection(self) -> None:
        """Drop and recreate the entire collection, deleting every vector unconditionally.

        Unlike delete_document (which needs to know a specific document_id), this
        clears everything regardless of what the app's memory/registry currently
        tracks. This matters when using MEMORY_BACKEND=memory alongside a
        persistent Qdrant instance: the registry forgets documents on every
        restart, but Qdrant doesn't -- so repeated uploads across restarts can
        silently accumulate orphaned duplicate vectors that a registry-driven
        "delete all" would never find. This is the reliable full reset.
        """
        try:
            exists = await self._client.collection_exists(self.collection_name)
            if exists:
                await self._client.delete_collection(self.collection_name)
                logger.info(f"Dropped Qdrant collection '{self.collection_name}'")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to drop Qdrant collection: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to clear vector store: {exc}",
            ) from exc

        await self.ensure_collection()

    async def count_points(self) -> int:
        try:
            result = await self._client.count(collection_name=self.collection_name, exact=True)
            return result.count
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to count Qdrant points: {exc}")
            return 0

    async def check_health(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False
