import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.rag.document_loader import DocumentLoader, DocumentLoaderError
from app.rag.embedding_client import EmbeddingClientError, OllamaEmbeddingClient
from app.rag.text_chunker import TextChunker, TextChunkerError
from app.rag.vector_store import ChromaVectorStore, VectorStoreError

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None] | None


@dataclass(frozen=True)
class IngestionResult:
    """Outcome of a single PDF ingestion pipeline run."""

    source: str
    pages_extracted: int
    chunks_created: int
    chunks_embedded: int
    chunks_stored: int
    previous_chunks_deleted: int
    collection_count: int


class IngestionServiceError(Exception):
    """Raised when the ingestion pipeline fails."""


class DocumentIngestionService:
    """Orchestrates the full PDF ingestion pipeline.

    Pipeline stages:

        1. Load PDF via :class:`DocumentLoader`
        2. Chunk pages via :class:`TextChunker`
        3. Embed chunks via :class:`OllamaEmbeddingClient`
        4. (Optionally) delete existing source records
        5. Store via :class:`ChromaVectorStore`

    Args:
        document_loader: Component that extracts text from PDFs.
        text_chunker: Component that splits pages into chunks.
        embedding_client: Component that produces embedding vectors.
        vector_store: Component that persists embedded chunks.
    """

    def __init__(
        self,
        document_loader: DocumentLoader,
        text_chunker: TextChunker,
        embedding_client: OllamaEmbeddingClient,
        vector_store: ChromaVectorStore,
    ) -> None:
        self._loader = document_loader
        self._chunker = text_chunker
        self._embedder = embedding_client
        self._store = vector_store

    @property
    def store(self) -> ChromaVectorStore:
        """Read-only access to the underlying vector store for queries."""
        return self._store

    # ── public API ──────────────────────────────────────────────────

    async def ingest_pdf(
        self,
        pdf_path: str,
        replace_existing: bool = True,
        source_name: str | None = None,
        progress_callback: ProgressCallback = None,
    ) -> IngestionResult:
        """Run the full ingestion pipeline on a PDF file.

        Args:
            pdf_path: Path to the PDF file to ingest.
            replace_existing: If ``True``, delete any existing chunks
                for the same source before storing new ones.
            source_name: Optional override for the source name. When
                provided this is used as the canonical source identifier
                instead of ``Path(pdf_path).name``.
            progress_callback: Optional ``(stage, completed, total)``
                callback for progress reporting.

        Returns:
            An :class:`IngestionResult` summarising the operation.

        Raises:
            IngestionServiceError: If any pipeline stage fails or
                consistency checks are violated.
        """
        path = Path(pdf_path)
        source = source_name or path.name

        logger.info(
            "Ingestion started — source=%s replace_existing=%s",
            source,
            replace_existing,
        )

        # ── 1. Load PDF ──────────────────────────────────────────
        self._report(progress_callback, "load", 0, 1)
        try:
            pages = self._loader.load_pdf(pdf_path, source_name=source_name)
        except (DocumentLoaderError, FileNotFoundError) as exc:
            raise IngestionServiceError(
                f"Failed to load PDF '{source}': {exc}"
            ) from exc

        if not pages:
            raise IngestionServiceError(
                f"PDF '{source}' contains no extractable text pages"
            )

        self._validate_pages(pages, source)

        pages_count = len(pages)
        self._report(progress_callback, "load", 1, 1)
        logger.info("Pages extracted: %d", pages_count)

        # ── 2. Chunk pages ───────────────────────────────────────
        self._report(progress_callback, "chunk", 0, 1)
        try:
            chunks = self._chunker.chunk_pages(pages)
        except TextChunkerError as exc:
            raise IngestionServiceError(
                f"Failed to chunk pages for '{source}': {exc}"
            ) from exc

        if not chunks:
            raise IngestionServiceError(
                f"No chunks produced from '{source}'"
            )

        self._validate_chunks(chunks, source, pages_count)

        chunks_count = len(chunks)
        self._report(progress_callback, "chunk", 1, 1)
        logger.info("Chunks created: %d", chunks_count)

        # ── 3. Embed chunks ──────────────────────────────────────
        self._report(progress_callback, "embed", 0, chunks_count)
        try:
            embedded = await self._embedder.embed_chunks(chunks)
        except EmbeddingClientError as exc:
            raise IngestionServiceError(
                f"Failed to embed chunks for '{source}': {exc}"
            ) from exc

        if len(embedded) != chunks_count:
            raise IngestionServiceError(
                f"Embedded count mismatch for '{source}': "
                f"expected {chunks_count}, got {len(embedded)}"
            )

        self._validate_embedded(embedded, source, chunks_count)

        self._report(progress_callback, "embed", chunks_count, chunks_count)
        logger.info("Chunks embedded: %d", len(embedded))

        # ── 4. (Optional) delete existing source ─────────────────
        deleted = 0
        if replace_existing:
            self._report(progress_callback, "delete", 0, 1)
            try:
                deleted = self._store.delete_by_source(source)
            except VectorStoreError as exc:
                raise IngestionServiceError(
                    f"Failed to delete existing source '{source}': {exc}"
                ) from exc
            self._report(progress_callback, "delete", 1, 1)

            if deleted:
                logger.info(
                    "Deleted %d existing chunk(s) for source='%s'",
                    deleted,
                    source,
                )

        # ── 5. Store chunks ──────────────────────────────────────
        self._report(progress_callback, "store", 0, 1)
        try:
            stored = self._store.add_chunks(embedded)
        except VectorStoreError as exc:
            msg = (
                f"Failed to store embedded chunks for '{source}'."
            )
            if replace_existing and deleted > 0:
                msg += (
                    f" Previous records ({deleted}) for this source "
                    "may already have been removed."
                )
            raise IngestionServiceError(msg) from exc

        if stored != len(embedded):
            raise IngestionServiceError(
                f"Stored count mismatch for '{source}': "
                f"expected {len(embedded)}, got {stored}"
            )

        self._report(progress_callback, "store", 1, 1)

        # ── 6. Final count ───────────────────────────────────────
        try:
            final_count = self._store.count()
        except VectorStoreError as exc:
            raise IngestionServiceError(
                f"Failed to get collection count after storing '{source}': {exc}"
            ) from exc

        logger.info(
            "Ingestion complete — source=%s pages=%d chunks=%d "
            "embedded=%d stored=%d deleted=%d collection=%d",
            source,
            pages_count,
            chunks_count,
            len(embedded),
            stored,
            deleted,
            final_count,
        )

        return IngestionResult(
            source=source,
            pages_extracted=pages_count,
            chunks_created=chunks_count,
            chunks_embedded=len(embedded),
            chunks_stored=stored,
            previous_chunks_deleted=deleted,
            collection_count=final_count,
        )

    # ── private helpers ─────────────────────────────────────────────

    @staticmethod
    def _report(
        callback: ProgressCallback,
        stage: str,
        completed: int,
        total: int,
    ) -> None:
        """Invoke the progress callback if one is set."""
        if callback is not None:
            callback(stage, completed, total)

    @staticmethod
    def _validate_pages(
        pages: list,
        expected_source: str,
    ) -> None:
        """Validate page-level consistency.

        Args:
            pages: List of ``DocumentPage`` objects.
            expected_source: The expected source filename.

        Raises:
            IngestionServiceError: If any page has a mismatched source,
                a non-positive page number, or an empty chunk ID.
        """
        for page in pages:
            src = getattr(page, "source", None)
            if src != expected_source:
                raise IngestionServiceError(
                    f"Inconsistent source in page {getattr(page, 'page_number', '?')}: "
                    f"expected '{expected_source}', got '{src}'"
                )

            pn = getattr(page, "page_number", None)
            if pn is not None and pn <= 0:
                raise IngestionServiceError(
                    f"Invalid page number {pn} in '{expected_source}'"
                )

    @staticmethod
    def _validate_chunks(
        chunks: list,
        expected_source: str,
        max_expected_pages: int,
    ) -> None:
        """Validate chunk-level consistency.

        Args:
            chunks: List of ``DocumentChunk`` objects.
            expected_source: The expected source filename.
            max_expected_pages: Upper bound for valid page numbers.

        Raises:
            IngestionServiceError: If any chunk has a mismatched source,
                an invalid page number, or an empty chunk ID.
        """
        for chunk in chunks:
            src = getattr(chunk, "source", None)
            if src != expected_source:
                raise IngestionServiceError(
                    f"Inconsistent source in chunk '{getattr(chunk, 'chunk_id', '?')}': "
                    f"expected '{expected_source}', got '{src}'"
                )

            pn = getattr(chunk, "page_number", None)
            if pn is not None and pn <= 0:
                raise IngestionServiceError(
                    f"Invalid page number {pn} in chunk "
                    f"'{getattr(chunk, 'chunk_id', '?')}'"
                )

            cid = getattr(chunk, "chunk_id", None)
            if not cid or (isinstance(cid, str) and not cid.strip()):
                raise IngestionServiceError(
                    "Encountered a chunk with an empty chunk_id"
                )

    @staticmethod
    def _validate_embedded(
        embedded: list,
        expected_source: str,
        expected_count: int,
    ) -> None:
        """Validate embedded-chunk consistency.

        Args:
            embedded: List of ``EmbeddedChunk`` objects.
            expected_source: The expected source filename.
            expected_count: Expected number of chunks.

        Raises:
            IngestionServiceError: If any embedded chunk has a
                mismatched source or if the count is wrong.
        """
        if len(embedded) != expected_count:
            raise IngestionServiceError(
                f"Embedded count mismatch: expected {expected_count}, "
                f"got {len(embedded)}"
            )

        for ec in embedded:
            src = getattr(ec, "source", None)
            if src != expected_source:
                raise IngestionServiceError(
                    f"Inconsistent source in embedded chunk "
                    f"'{getattr(ec, 'chunk_id', '?')}': "
                    f"expected '{expected_source}', got '{src}'"
                )


if __name__ == "__main__":
    import asyncio
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"

    from app.core.config import settings

    settings.validate()

    loader = DocumentLoader()
    chunker = TextChunker()
    embedder = OllamaEmbeddingClient()
    store = ChromaVectorStore()

    service = DocumentIngestionService(
        document_loader=loader,
        text_chunker=chunker,
        embedding_client=embedder,
        vector_store=store,
    )

    async def run() -> None:
        try:
            result = await service.ingest_pdf(pdf_path, replace_existing=True)
        except IngestionServiceError as exc:
            print(f"Error: {exc}")
            return

        print(f"\nIngestion result:")
        print(f"  source:                   {result.source}")
        print(f"  pages_extracted:          {result.pages_extracted}")
        print(f"  chunks_created:           {result.chunks_created}")
        print(f"  chunks_embedded:          {result.chunks_embedded}")
        print(f"  chunks_stored:            {result.chunks_stored}")
        print(f"  previous_chunks_deleted:  {result.previous_chunks_deleted}")
        print(f"  collection_count:         {result.collection_count}")

    asyncio.run(run())
