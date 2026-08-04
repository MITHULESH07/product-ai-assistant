from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.rag.document_loader import DocumentLoader, DocumentLoaderError
from app.rag.embedding_client import EmbeddingClientError, EmbeddedChunk, OllamaEmbeddingClient
from app.rag.ingestion_service import DocumentIngestionService, IngestionResult, IngestionServiceError
from app.rag.text_chunker import DocumentChunk, TextChunker, TextChunkerError
from app.rag.vector_store import ChromaVectorStore, VectorStoreError

pytestmark = pytest.mark.asyncio

# ── helpers ─────────────────────────────────────────────────────────


def _make_page(
    page_number: int = 1,
    text: str = "Page text content",
    source: str = "doc.pdf",
) -> object:
    """Create a page-like object matching DocumentPage shape."""
    return type("FakePage", (), {
        "page_number": page_number,
        "text": text,
        "source": source,
    })()


def _make_chunk(
    chunk_id: str = "doc.pdf:p1:c0",
    text: str = "Chunk text",
    source: str = "doc.pdf",
    page_number: int = 1,
    chunk_index: int = 0,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        page_number=page_number,
        chunk_index=chunk_index,
    )


def _make_embedded(
    chunk_id: str = "doc.pdf:p1:c0",
    text: str = "Chunk text",
    source: str = "doc.pdf",
    page_number: int = 1,
    chunk_index: int = 0,
    embedding: list[float] | None = None,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        page_number=page_number,
        chunk_index=chunk_index,
        embedding=embedding or [0.1, 0.2, 0.3],
    )


def _make_service(
    loader: MagicMock | None = None,
    chunker: MagicMock | None = None,
    embedder: AsyncMock | None = None,
    store: MagicMock | None = None,
) -> DocumentIngestionService:
    return DocumentIngestionService(
        document_loader=loader or MagicMock(spec=DocumentLoader),
        text_chunker=chunker or MagicMock(spec=TextChunker),
        embedding_client=embedder or AsyncMock(spec=OllamaEmbeddingClient),
        vector_store=store or MagicMock(spec=ChromaVectorStore),
    )


# ── tests ───────────────────────────────────────────────────────────


class TestIngestPdf:
    async def test_success(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [
            _make_page(page_number=1),
            _make_page(page_number=2),
        ]

        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.return_value = [
            _make_chunk(chunk_id="doc.pdf:p1:c0", page_number=1),
            _make_chunk(chunk_id="doc.pdf:p2:c0", page_number=2),
        ]

        embedder = AsyncMock(spec=OllamaEmbeddingClient)
        embedder.embed_chunks.return_value = [
            _make_embedded(chunk_id="doc.pdf:p1:c0", page_number=1),
            _make_embedded(chunk_id="doc.pdf:p2:c0", page_number=2),
        ]

        store = MagicMock(spec=ChromaVectorStore)
        store.delete_by_source.return_value = 0
        store.add_chunks.return_value = 2
        store.count.return_value = 5

        service = _make_service(loader, chunker, embedder, store)
        result = await service.ingest_pdf("doc.pdf", replace_existing=True)

        assert isinstance(result, IngestionResult)
        assert result.source == "doc.pdf"
        assert result.pages_extracted == 2
        assert result.chunks_created == 2
        assert result.chunks_embedded == 2
        assert result.chunks_stored == 2
        assert result.previous_chunks_deleted == 0
        assert result.collection_count == 5

    async def test_replace_existing_deletes_before_store(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [_make_page()]
        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.return_value = [_make_chunk()]
        embedder = AsyncMock(spec=OllamaEmbeddingClient)
        embedder.embed_chunks.return_value = [_make_embedded()]
        store = MagicMock(spec=ChromaVectorStore)
        store.delete_by_source.return_value = 7
        store.add_chunks.return_value = 1
        store.count.return_value = 10

        service = _make_service(loader, chunker, embedder, store)
        result = await service.ingest_pdf("doc.pdf", replace_existing=True)

        store.delete_by_source.assert_called_once_with("doc.pdf")
        # delete must happen before add
        store.add_chunks.assert_called_once()

        assert result.previous_chunks_deleted == 7
        assert result.chunks_stored == 1
        assert result.collection_count == 10

    async def test_replace_existing_false_no_deletion(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [_make_page()]
        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.return_value = [_make_chunk()]
        embedder = AsyncMock(spec=OllamaEmbeddingClient)
        embedder.embed_chunks.return_value = [_make_embedded()]
        store = MagicMock(spec=ChromaVectorStore)
        store.add_chunks.return_value = 1
        store.count.return_value = 3

        service = _make_service(loader, chunker, embedder, store)
        result = await service.ingest_pdf("doc.pdf", replace_existing=False)

        store.delete_by_source.assert_not_called()
        assert result.previous_chunks_deleted == 0

    async def test_empty_pages_raises(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = []
        service = _make_service(loader=loader)

        with pytest.raises(IngestionServiceError, match="no extractable text"):
            await service.ingest_pdf("doc.pdf")

    async def test_empty_chunks_raises(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [_make_page()]
        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.return_value = []
        service = _make_service(loader=loader, chunker=chunker)

        with pytest.raises(IngestionServiceError, match="No chunks produced"):
            await service.ingest_pdf("doc.pdf")

    async def test_embedding_failure_raises(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [_make_page()]
        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.return_value = [_make_chunk()]
        embedder = AsyncMock(spec=OllamaEmbeddingClient)
        embedder.embed_chunks.side_effect = EmbeddingClientError("Ollama down")

        service = _make_service(loader=loader, chunker=chunker, embedder=embedder)

        with pytest.raises(IngestionServiceError, match="Failed to embed"):
            await service.ingest_pdf("doc.pdf")

    async def test_embedded_count_mismatch_raises(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [_make_page(), _make_page()]
        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.return_value = [
            _make_chunk(chunk_id="doc.pdf:p1:c0"),
            _make_chunk(chunk_id="doc.pdf:p2:c0"),
        ]
        embedder = AsyncMock(spec=OllamaEmbeddingClient)
        embedder.embed_chunks.return_value = [_make_embedded()]  # only 1

        service = _make_service(loader=loader, chunker=chunker, embedder=embedder)

        with pytest.raises(IngestionServiceError, match="count mismatch"):
            await service.ingest_pdf("doc.pdf")

    async def test_storage_failure_raises(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [_make_page()]
        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.return_value = [_make_chunk()]
        embedder = AsyncMock(spec=OllamaEmbeddingClient)
        embedder.embed_chunks.return_value = [_make_embedded()]
        store = MagicMock(spec=ChromaVectorStore)
        store.delete_by_source.return_value = 0
        store.add_chunks.side_effect = VectorStoreError("Chroma error")

        service = _make_service(loader, chunker, embedder, store)

        with pytest.raises(IngestionServiceError, match="Failed to store"):
            await service.ingest_pdf("doc.pdf", replace_existing=True)

    async def test_stored_count_mismatch_raises(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [_make_page()]
        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.return_value = [_make_chunk()]
        embedder = AsyncMock(spec=OllamaEmbeddingClient)
        embedder.embed_chunks.return_value = [_make_embedded()]
        store = MagicMock(spec=ChromaVectorStore)
        store.add_chunks.return_value = 0  # stored 0, expected 1

        service = _make_service(loader, chunker, embedder, store)

        with pytest.raises(IngestionServiceError, match="Stored count mismatch"):
            await service.ingest_pdf("doc.pdf")

    async def test_delete_failure_raises(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [_make_page()]
        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.return_value = [_make_chunk()]
        embedder = AsyncMock(spec=OllamaEmbeddingClient)
        embedder.embed_chunks.return_value = [_make_embedded()]
        store = MagicMock(spec=ChromaVectorStore)
        store.delete_by_source.side_effect = VectorStoreError("Delete failed")

        service = _make_service(loader, chunker, embedder, store)

        with pytest.raises(IngestionServiceError, match="Failed to delete"):
            await service.ingest_pdf("doc.pdf", replace_existing=True)

    async def test_pdf_loader_error_chains(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.side_effect = DocumentLoaderError("Bad PDF")
        service = _make_service(loader=loader)

        with pytest.raises(IngestionServiceError, match="Failed to load") as exc:
            await service.ingest_pdf("doc.pdf")
        assert isinstance(exc.value.__cause__, DocumentLoaderError)

    async def test_chunker_error_chains(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [_make_page()]
        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.side_effect = TextChunkerError("Bad chunk")
        service = _make_service(loader=loader, chunker=chunker)

        with pytest.raises(IngestionServiceError, match="Failed to chunk") as exc:
            await service.ingest_pdf("doc.pdf")
        assert isinstance(exc.value.__cause__, TextChunkerError)

    async def test_embedder_error_chains(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [_make_page()]
        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.return_value = [_make_chunk()]
        embedder = AsyncMock(spec=OllamaEmbeddingClient)
        embedder.embed_chunks.side_effect = EmbeddingClientError("No server")
        service = _make_service(loader=loader, chunker=chunker, embedder=embedder)

        with pytest.raises(IngestionServiceError, match="Failed to embed") as exc:
            await service.ingest_pdf("doc.pdf")
        assert isinstance(exc.value.__cause__, EmbeddingClientError)

    async def test_storage_error_chains(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [_make_page()]
        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.return_value = [_make_chunk()]
        embedder = AsyncMock(spec=OllamaEmbeddingClient)
        embedder.embed_chunks.return_value = [_make_embedded()]
        store = MagicMock(spec=ChromaVectorStore)
        store.delete_by_source.return_value = 0
        store.add_chunks.side_effect = VectorStoreError("Insert failed")
        service = _make_service(loader, chunker, embedder, store)

        with pytest.raises(IngestionServiceError, match="Failed to store") as exc:
            await service.ingest_pdf("doc.pdf")
        assert isinstance(exc.value.__cause__, VectorStoreError)

    async def test_inconsistent_page_source_raises(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [
            _make_page(source="doc.pdf"),
            _make_page(source="other.pdf"),
        ]
        service = _make_service(loader=loader)

        with pytest.raises(IngestionServiceError, match="Inconsistent source"):
            await service.ingest_pdf("doc.pdf")

    async def test_invalid_page_number_raises(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [
            _make_page(page_number=0),
        ]
        service = _make_service(loader=loader)

        with pytest.raises(IngestionServiceError, match="Invalid page number"):
            await service.ingest_pdf("doc.pdf")

    async def test_inconsistent_chunk_source_raises(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [_make_page()]
        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.return_value = [
            _make_chunk(source="wrong.pdf"),
        ]
        service = _make_service(loader=loader, chunker=chunker)

        with pytest.raises(IngestionServiceError, match="Inconsistent source"):
            await service.ingest_pdf("doc.pdf")

    async def test_empty_chunk_id_raises(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [_make_page()]
        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.return_value = [
            _make_chunk(chunk_id=""),
        ]
        service = _make_service(loader=loader, chunker=chunker)

        with pytest.raises(IngestionServiceError, match="empty chunk_id"):
            await service.ingest_pdf("doc.pdf")

    async def test_inconsistent_embedded_source_raises(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [_make_page()]
        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.return_value = [_make_chunk()]
        embedder = AsyncMock(spec=OllamaEmbeddingClient)
        embedder.embed_chunks.return_value = [
            _make_embedded(source="wrong.pdf"),
        ]
        service = _make_service(loader=loader, chunker=chunker, embedder=embedder)

        with pytest.raises(IngestionServiceError, match="Inconsistent source"):
            await service.ingest_pdf("doc.pdf")

    async def test_deletion_happens_after_embedding(self):
        """Verify that delete_by_source is called only after embedding
        succeeds (never before)."""
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [_make_page()]
        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.return_value = [_make_chunk()]

        embedder = AsyncMock(spec=OllamaEmbeddingClient)
        embedder.embed_chunks.return_value = [_make_embedded()]

        store = MagicMock(spec=ChromaVectorStore)
        store.delete_by_source.return_value = 2
        store.add_chunks.return_value = 1
        store.count.return_value = 5

        # Track call order
        call_log: list[str] = []

        def log_delete(*args, **kwargs):
            call_log.append("delete")
            return 2

        def log_add(*args, **kwargs):
            call_log.append("add")
            return 1

        store.delete_by_source.side_effect = log_delete
        store.add_chunks.side_effect = log_add

        service = _make_service(loader, chunker, embedder, store)
        await service.ingest_pdf("doc.pdf", replace_existing=True)

        assert call_log == ["delete", "add"], f"Unexpected order: {call_log}"

    async def test_progress_callback_invoked(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [_make_page()]
        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.return_value = [_make_chunk()]
        embedder = AsyncMock(spec=OllamaEmbeddingClient)
        embedder.embed_chunks.return_value = [_make_embedded()]
        store = MagicMock(spec=ChromaVectorStore)
        store.delete_by_source.return_value = 0
        store.add_chunks.return_value = 1
        store.count.return_value = 3

        calls: list[tuple[str, int, int]] = []

        def cb(stage: str, completed: int, total: int) -> None:
            calls.append((stage, completed, total))

        service = _make_service(loader, chunker, embedder, store)
        await service.ingest_pdf("doc.pdf", replace_existing=True, progress_callback=cb)

        # Should have been called for at least load, chunk, embed, delete, store
        stages = {c[0] for c in calls}
        assert "load" in stages
        assert "chunk" in stages
        assert "embed" in stages
        assert "delete" in stages
        assert "store" in stages

    async def test_ingestion_result_values(self):
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [
            _make_page(page_number=1),
            _make_page(page_number=2),
            _make_page(page_number=3),
        ]
        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.return_value = [
            _make_chunk(chunk_id="d:p1:c0", page_number=1),
            _make_chunk(chunk_id="d:p1:c1", page_number=1),
            _make_chunk(chunk_id="d:p2:c0", page_number=2),
            _make_chunk(chunk_id="d:p3:c0", page_number=3),
        ]
        embedder = AsyncMock(spec=OllamaEmbeddingClient)
        embedder.embed_chunks.return_value = [
            _make_embedded(chunk_id="d:p1:c0", page_number=1),
            _make_embedded(chunk_id="d:p1:c1", page_number=1),
            _make_embedded(chunk_id="d:p2:c0", page_number=2),
            _make_embedded(chunk_id="d:p3:c0", page_number=3),
        ]
        store = MagicMock(spec=ChromaVectorStore)
        store.delete_by_source.return_value = 5
        store.add_chunks.return_value = 4
        store.count.return_value = 12

        service = _make_service(loader, chunker, embedder, store)
        result = await service.ingest_pdf("doc.pdf", replace_existing=True)

        assert result.source == "doc.pdf"
        assert result.pages_extracted == 3
        assert result.chunks_created == 4
        assert result.chunks_embedded == 4
        assert result.chunks_stored == 4
        assert result.previous_chunks_deleted == 5
        assert result.collection_count == 12

    async def test_delete_failure_with_previous_deleted_message(self):
        """When delete succeeds but storage fails, the error message
        must warn about possible partial deletion."""
        loader = MagicMock(spec=DocumentLoader)
        loader.load_pdf.return_value = [_make_page()]
        chunker = MagicMock(spec=TextChunker)
        chunker.chunk_pages.return_value = [_make_chunk()]
        embedder = AsyncMock(spec=OllamaEmbeddingClient)
        embedder.embed_chunks.return_value = [_make_embedded()]
        store = MagicMock(spec=ChromaVectorStore)
        store.delete_by_source.return_value = 3
        store.add_chunks.side_effect = VectorStoreError("Insert failed")

        service = _make_service(loader, chunker, embedder, store)

        with pytest.raises(IngestionServiceError, match="may already have been removed"):
            await service.ingest_pdf("doc.pdf", replace_existing=True)
