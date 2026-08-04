import logging
from dataclasses import dataclass

from app.rag.embedding_client import EmbeddingClientError, OllamaEmbeddingClient
from app.rag.vector_store import ChromaVectorStore, VectorStoreError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A document chunk returned from a similarity search."""

    chunk_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int
    distance: float
    relevance_score: float


class RetrievalServiceError(Exception):
    """Raised when a retrieval operation fails."""


class RetrievalService:
    """Converts a user question into relevant document chunks.

    Embeds the query, searches the vector store, and returns
    ranked results with normalised relevance scores.

    Args:
        embedding_client: Client used to embed the query.
        vector_store: The ChromaDB-backed vector store.
    """

    def __init__(
        self,
        embedding_client: OllamaEmbeddingClient,
        vector_store: ChromaVectorStore,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_store = vector_store

    # ── public API ──────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source: str | None = None,
        minimum_relevance: float | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a user question.

        Args:
            query: The user's question or search text.
            top_k: Maximum number of results to return.
            source: Optional exact-source filter.
            minimum_relevance: Optional minimum relevance score
                (0.0–1.0); results below this threshold are excluded.

        Returns:
            A list of :class:`RetrievedChunk` objects, ordered from
            most to least relevant.

        Raises:
            RetrievalServiceError: If the query is empty, the
                embedding fails, or the vector-store query fails.
        """
        if not query or not query.strip():
            raise RetrievalServiceError("Query must be non-empty")

        if top_k <= 0:
            raise RetrievalServiceError(
                f"top_k must be positive, got {top_k}"
            )

        if minimum_relevance is not None and not (
            0.0 <= minimum_relevance <= 1.0
        ):
            raise RetrievalServiceError(
                f"minimum_relevance must be between 0.0 and 1.0, "
                f"got {minimum_relevance}"
            )

        logger.info(
            "RAG retrieval started: query_len=%d top_k=%d source=%s minimum_relevance=%s",
            len(query),
            top_k,
            source or "(all)",
            minimum_relevance if minimum_relevance is not None else "(none)",
        )

        # 1. embed the query
        try:
            query_embedding = await self._embedding_client.embed_text(query)
        except EmbeddingClientError as exc:
            raise RetrievalServiceError(
                f"Failed to embed query: {exc}"
            ) from exc

        # 2. query the vector store
        try:
            raw = self._vector_store.query(
                query_embedding=query_embedding,
                top_k=top_k,
                source=source,
            )
        except VectorStoreError as exc:
            raise RetrievalServiceError(
                f"Failed to query vector store: {exc}"
            ) from exc

        # 3. parse the nested ChromaDB response
        chunks = self._parse_query_result(raw)

        # 4. filter by minimum relevance
        if minimum_relevance is not None:
            before = len(chunks)
            chunks = [c for c in chunks if c.relevance_score >= minimum_relevance]
            filtered = before - len(chunks)
            if filtered:
                logger.info(
                    "Filtered %d result(s) below minimum_relevance=%.2f",
                    filtered,
                    minimum_relevance,
                )

        if chunks:
            top = chunks[0]
            sources_found = sorted(set(c.source for c in chunks))
            logger.info(
                "RAG retrieval completed: chunks=%d top_source=%s top_page=%s "
                "top_relevance=%.4f sources=%s",
                len(chunks),
                top.source,
                top.page_number,
                top.relevance_score,
                sources_found,
            )
        else:
            logger.info("RAG retrieval completed: chunks=0")

        return chunks

    @staticmethod
    def format_context(
        chunks: list[RetrievedChunk],
        max_characters: int = 6000,
    ) -> str:
        """Format retrieved chunks into a citation-ready context string.

        Each chunk is prefixed with a ``[Source: …, Page: …]`` header.

        Args:
            chunks: Retrieved chunks in ranked order.
            max_characters: Maximum total characters for the output.

        Returns:
            A formatted string, or an empty string if *chunks* is empty.

        Raises:
            RetrievalServiceError: If *max_characters* is not positive.
        """
        if max_characters <= 0:
            raise RetrievalServiceError(
                f"max_characters must be positive, got {max_characters}"
            )

        if not chunks:
            return ""

        parts: list[str] = []
        remaining = max_characters

        for chunk in chunks:
            header = f"[Source: {chunk.source}, Page: {chunk.page_number}]\n"
            block = header + chunk.text + "\n\n"

            if remaining <= 0:
                break

            if len(block) <= remaining:
                parts.append(block)
                remaining -= len(block)
            elif not parts:
                # First chunk exceeds the limit — include header + truncated text
                # Reserve 2 chars for the trailing newlines
                allowed = remaining - len(header) - 2
                if allowed > 0:
                    parts.append(header + chunk.text[:allowed] + "\n\n")
                break
            else:
                break

        return "".join(parts)

    # ── private helpers ─────────────────────────────────────────────

    @staticmethod
    def _parse_query_result(raw: dict) -> list[RetrievedChunk]:
        """Parse and validate the nested ChromaDB query result.

        Args:
            raw: The raw response from ``ChromaVectorStore.query()``.

        Returns:
            A list of :class:`RetrievedChunk` objects, ordered by
            relevance.

        Raises:
            RetrievalServiceError: If the response structure is
                invalid, metadata is missing, or distances are
                malformed.
        """
        try:
            outer_ids = raw["ids"]
            outer_docs = raw["documents"]
            outer_meta = raw["metadatas"]
            outer_dist = raw["distances"]
        except KeyError as exc:
            raise RetrievalServiceError(
                f"ChromaDB response missing key: {exc}"
            ) from exc

        if not (
            isinstance(outer_ids, list)
            and isinstance(outer_docs, list)
            and isinstance(outer_meta, list)
            and isinstance(outer_dist, list)
        ):
            raise RetrievalServiceError(
                "ChromaDB response fields must be lists"
            )

        if not outer_ids:
            return []

        inner_ids = outer_ids[0]
        inner_docs = outer_docs[0]
        inner_meta = outer_meta[0]
        inner_dist = outer_dist[0]

        if not (
            isinstance(inner_ids, list)
            and isinstance(inner_docs, list)
            and isinstance(inner_meta, list)
            and isinstance(inner_dist, list)
        ):
            raise RetrievalServiceError(
                "ChromaDB response inner fields must be lists"
            )

        n = len(inner_ids)
        if not (len(inner_docs) == len(inner_meta) == len(inner_dist) == n):
            raise RetrievalServiceError(
                f"Mismatched result lengths: ids={n} "
                f"documents={len(inner_docs)} "
                f"metadatas={len(inner_meta)} "
                f"distances={len(inner_dist)}"
            )

        seen: set[str] = set()
        chunks: list[RetrievedChunk] = []

        for i in range(n):
            chunk_id = inner_ids[i]
            text = inner_docs[i]
            metadata = inner_meta[i]
            distance = inner_dist[i]

            # duplicate handling
            if chunk_id in seen:
                logger.warning("Duplicate chunk_id in results: %s", chunk_id)
                continue
            seen.add(chunk_id)

            # validate metadata
            if not isinstance(metadata, dict):
                raise RetrievalServiceError(
                    f"Chunk '{chunk_id}' has non-dict metadata: "
                    f"{type(metadata).__name__}"
                )

            source = metadata.get("source")
            page_number = metadata.get("page_number")

            if not source:
                raise RetrievalServiceError(
                    f"Chunk '{chunk_id}' is missing 'source' in metadata"
                )
            if page_number is None:
                raise RetrievalServiceError(
                    f"Chunk '{chunk_id}' is missing 'page_number' in metadata"
                )

            # validate distance
            if not isinstance(distance, (int, float)):
                raise RetrievalServiceError(
                    f"Chunk '{chunk_id}' has non-numeric distance: {distance!r}"
                )

            chunk_index = metadata.get("chunk_index", 0)

            relevance = max(0.0, min(1.0, 1.0 - distance))

            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source=str(source),
                    page_number=int(page_number),
                    chunk_index=int(chunk_index),
                    distance=float(distance),
                    relevance_score=relevance,
                )
            )

        return chunks


if __name__ == "__main__":
    import asyncio
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    question = sys.argv[1] if len(sys.argv) > 1 else "What is an H-bridge?"

    from app.core.config import settings

    settings.validate()

    embed_client = OllamaEmbeddingClient()
    store = ChromaVectorStore()
    service = RetrievalService(
        embedding_client=embed_client,
        vector_store=store,
    )

    async def run() -> None:
        try:
            results = await service.retrieve(
                query=question,
                top_k=3,
            )
        except RetrievalServiceError as exc:
            print(f"Error: {exc}")
            return

        if not results:
            print("No results found.")
            return

        print(f"\nRetrieved {len(results)} chunk(s):\n")
        for rank, c in enumerate(results, 1):
            preview = c.text[:200].replace("\n", " ¶ ")
            print(
                f"  #{rank}  [{c.chunk_id}]  src={c.source}  p={c.page_number}\n"
                f"        distance={c.distance:.4f}  "
                f"relevance={c.relevance_score:.4f}\n"
                f"        {preview}...\n"
            )

        print("─" * 50)
        print("Formatted context:")
        print(service.format_context(results, max_characters=1500))

    asyncio.run(run())
