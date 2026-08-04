import logging
import math
from pathlib import Path

import chromadb

from app.core.config import settings
from app.rag.embedding_client import EmbeddedChunk

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


class VectorStoreError(Exception):
    """Raised when a vector-store operation fails."""


class ChromaVectorStore:
    """Persists embedded document chunks in a local ChromaDB collection.

    Uses ``upsert`` so that re-ingesting the same ``chunk_id`` updates
    the existing record rather than creating a duplicate.

    Args:
        persist_directory: Override the persistence directory
            (default: ``settings.chroma_persist_directory``).
        collection_name: Override the collection name
            (default: ``settings.chroma_collection_name``).
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._persist_directory = persist_directory or settings.chroma_persist_directory
        self._collection_name = collection_name or settings.chroma_collection_name

        persist_path = Path(self._persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ChromaDB store — directory=%s collection=%s",
            str(persist_path.resolve()),
            self._collection_name,
        )

        try:
            self._client = chromadb.PersistentClient(path=str(persist_path))
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to initialize ChromaDB client at "
                f"'{persist_path}': {exc}"
            ) from exc

        try:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to create or retrieve collection "
                f"'{self._collection_name}': {exc}"
            ) from exc

        logger.info(
            "ChromaDB collection ready — name=%s count=%d",
            self._collection_name,
            self._collection.count(),
        )

    # ── public API ──────────────────────────────────────────────────

    def add_chunks(self, chunks: list[EmbeddedChunk]) -> int:
        """Store a list of embedded chunks (idempotent via upsert).

        Args:
            chunks: The embedded chunks to store.

        Returns:
            The number of chunks successfully stored.

        Raises:
            VectorStoreError: If any chunk is invalid or insertion
                fails.
        """
        if not chunks:
            return 0

        self._validate_chunks(chunks)

        batches = self._batch_chunks(chunks)
        total = 0

        for batch_num, batch in enumerate(batches):
            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            embeddings = [c.embedding for c in batch]
            metadatas = [
                {
                    "source": c.source,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                }
                for c in batch
            ]

            logger.debug(
                "Upserting batch %d/%d — %d chunk(s)",
                batch_num + 1,
                len(batches),
                len(batch),
            )

            try:
                self._collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to upsert batch {batch_num + 1} "
                    f"({len(batch)} chunk(s)): {exc}"
                ) from exc

            total += len(batch)

        logger.info(
            "Stored %d chunk(s) in '%s' (collection count: %d)",
            total,
            self._collection_name,
            self._collection.count(),
        )

        return total

    def count(self) -> int:
        """Return the number of records currently in the collection.

        Returns:
            The record count.
        """
        try:
            return self._collection.count()
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to get collection count: {exc}"
            ) from exc

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks whose metadata ``source`` matches exactly.

        Args:
            source: The source filename to delete.

        Returns:
            The number of deleted chunks.

        Raises:
            VectorStoreError: If the source is empty or deletion fails.
        """
        if not source or not source.strip():
            raise VectorStoreError("Cannot delete by empty source")

        try:
            result = self._collection.get(where={"source": source})
            ids_to_delete = result.get("ids", [])

            if not ids_to_delete:
                logger.info("No chunks found for source='%s'", source)
                return 0

            self._collection.delete(ids=ids_to_delete)
            logger.info(
                "Deleted %d chunk(s) for source='%s'",
                len(ids_to_delete),
                source,
            )
            return len(ids_to_delete)

        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete chunks for source='{source}': {exc}"
            ) from exc

    def contains_chunk(self, chunk_id: str) -> bool:
        """Check whether a chunk ID exists in the collection.

        Args:
            chunk_id: The chunk ID to look up.

        Returns:
            ``True`` if the ID exists, ``False`` otherwise.

        Raises:
            VectorStoreError: If the chunk ID is empty or the lookup
                fails.
        """
        if not chunk_id or not chunk_id.strip():
            raise VectorStoreError("chunk_id must be non-empty")

        try:
            result = self._collection.get(ids=[chunk_id])
            ids = result.get("ids", [])
            return len(ids) > 0
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to check chunk '{chunk_id}': {exc}"
            ) from exc

    def get_sources(self) -> list[str]:
        """Return unique, sorted source filenames in the collection.

        Returns:
            A sorted list of distinct source filenames, or an empty
            list if the collection is empty.
        """
        try:
            result = self._collection.get(include=["metadatas"])
            metadatas = result.get("metadatas", [])
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to retrieve sources from collection: {exc}"
            ) from exc

        if not metadatas:
            return []

        sources: set[str] = set()
        for md in metadatas:
            src = md.get("source") if isinstance(md, dict) else None
            if src:
                sources.add(str(src))

        return sorted(sources)

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        source: str | None = None,
    ) -> dict:
        """Query the collection for the nearest neighbours.

        Args:
            query_embedding: The embedding vector to search with.
            top_k: Number of results to return.
            source: Optional exact-source filter.

        Returns:
            The raw ChromaDB ``query`` result dict with keys
            ``ids``, ``documents``, ``metadatas``, ``distances``.

        Raises:
            VectorStoreError: If the embedding is empty, contains
                non-finite values, ``top_k`` is not positive, or the
                query itself fails.
        """
        if not query_embedding:
            raise VectorStoreError("Query embedding must be non-empty")

        for idx, val in enumerate(query_embedding):
            if not isinstance(val, (int, float)):
                raise VectorStoreError(
                    f"Query embedding index {idx} is non-numeric: {val!r}"
                )
            if not math.isfinite(val):
                raise VectorStoreError(
                    f"Query embedding index {idx} is non-finite: {val!r}"
                )

        if top_k <= 0:
            raise VectorStoreError(
                f"top_k must be positive, got {top_k}"
            )

        where_filter: dict | None = None
        if source is not None and source.strip():
            where_filter = {"source": source.strip()}

        try:
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to query collection '{self._collection_name}': {exc}"
            ) from exc

        return result

    # ── private helpers ─────────────────────────────────────────────

    @staticmethod
    def _validate_chunks(chunks: list[EmbeddedChunk]) -> None:
        """Validate all chunks before insertion.

        Checks for:
        * non-empty chunk_id
        * non-empty text
        * non-empty embedding
        * finite numeric embedding values
        * consistent embedding dimensions across the batch

        Args:
            chunks: The chunks to validate.

        Raises:
            VectorStoreError: If any validation check fails.
        """
        if not chunks:
            return

        dim: int | None = None

        for i, c in enumerate(chunks):
            if not c.chunk_id or not c.chunk_id.strip():
                raise VectorStoreError(
                    f"Chunk at index {i} has an empty chunk_id"
                )
            if not c.text or not c.text.strip():
                raise VectorStoreError(
                    f"Chunk '{c.chunk_id}' has empty text"
                )
            if not c.embedding:
                raise VectorStoreError(
                    f"Chunk '{c.chunk_id}' has an empty embedding"
                )

            for j, v in enumerate(c.embedding):
                if not isinstance(v, (int, float)):
                    raise VectorStoreError(
                        f"Chunk '{c.chunk_id}' embedding index {j} "
                        f"is non-numeric: {v!r}"
                    )
                if not math.isfinite(v):
                    raise VectorStoreError(
                        f"Chunk '{c.chunk_id}' embedding index {j} "
                        f"is non-finite: {v!r}"
                    )

            if dim is None:
                dim = len(c.embedding)
            elif len(c.embedding) != dim:
                raise VectorStoreError(
                    f"Chunk '{c.chunk_id}' has embedding dimension "
                    f"{len(c.embedding)}, expected {dim}"
                )

    @staticmethod
    def _batch_chunks(
        chunks: list[EmbeddedChunk],
    ) -> list[list[EmbeddedChunk]]:
        """Split a list of chunks into batches of *BATCH_SIZE*.

        Args:
            chunks: The full list of chunks.

        Returns:
            A list of batches.
        """
        return [chunks[i : i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"

    from app.rag.document_loader import DocumentLoader
    from app.rag.text_chunker import TextChunker

    loader = DocumentLoader()
    try:
        pages = loader.load_pdf(pdf_path)
    except Exception as exc:
        print(f"Error loading PDF: {exc}")
        sys.exit(1)

    chunker = TextChunker()
    chunks = chunker.chunk_pages(pages)

    print(f"Loaded {len(pages)} page(s) -> {len(chunks)} chunk(s)")

    import asyncio
    from app.rag.embedding_client import OllamaEmbeddingClient

    n = min(3, len(chunks))
    embed_client = OllamaEmbeddingClient()

    async def embed() -> list:
        embedded = []
        for i in range(n):
            ec = await embed_client.embed_chunk(chunks[i])
            embedded.append(ec)
        return embedded

    embedded_chunks = asyncio.run(embed())
    print(f"Embedded {len(embedded_chunks)} chunk(s)")

    store = ChromaVectorStore(
        persist_directory="./data/test_manual_chroma",
        collection_name="manual_test",
    )

    added = store.add_chunks(embedded_chunks)
    print(f"\nAdded: {added}")
    print(f"Count: {store.count()}")
    print(f"Sources: {store.get_sources()}")
    if embedded_chunks:
        print(f"Contains '{embedded_chunks[0].chunk_id}': {store.contains_chunk(embedded_chunks[0].chunk_id)}")
