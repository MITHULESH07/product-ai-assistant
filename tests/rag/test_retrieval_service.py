from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.rag.embedding_client import EmbeddingClientError, OllamaEmbeddingClient
from app.rag.retrieval_service import (
    RetrievalService,
    RetrievalServiceError,
    RetrievedChunk,
)
from app.rag.vector_store import ChromaVectorStore, VectorStoreError

# ── helpers ─────────────────────────────────────────────────────────

PATCH_EMBED = "app.rag.retrieval_service.OllamaEmbeddingClient"
PATCH_STORE = "app.rag.retrieval_service.ChromaVectorStore"

SAMPLE_QUERY_RESULT = {
    "ids": [["a.pdf:p1:c0", "a.pdf:p1:c1"]],
    "documents": [["Motor driver text", "H-bridge explanation"]],
    "metadatas": [
        [
            {"source": "a.pdf", "page_number": 1, "chunk_index": 0},
            {"source": "a.pdf", "page_number": 1, "chunk_index": 1},
        ],
    ],
    "distances": [[0.1, 0.3]],
}


def _make_service(
    embed_mock: AsyncMock | None = None,
    store_mock: MagicMock | None = None,
) -> RetrievalService:
    if embed_mock is None:
        embed_mock = AsyncMock(spec=OllamaEmbeddingClient)
    if store_mock is None:
        store_mock = MagicMock(spec=ChromaVectorStore)
    return RetrievalService(
        embedding_client=embed_mock,
        vector_store=store_mock,
    )


def _make_chunk(
    chunk_id: str = "a.pdf:p1:c0",
    text: str = "Some text",
    source: str = "a.pdf",
    page_number: int = 1,
    chunk_index: int = 0,
    distance: float = 0.1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        page_number=page_number,
        chunk_index=chunk_index,
        distance=distance,
        relevance_score=max(0.0, min(1.0, 1.0 - distance)),
    )


# ── retrieve tests ──────────────────────────────────────────────────


class TestRetrieve:
    pytestmark = pytest.mark.asyncio
    async def test_success(self):
        embed = AsyncMock(spec=OllamaEmbeddingClient)
        embed.embed_text.return_value = [1.0, 0.0, 0.0]
        store = MagicMock(spec=ChromaVectorStore)
        store.query.return_value = SAMPLE_QUERY_RESULT
        service = _make_service(embed, store)

        results = await service.retrieve(query="H-bridge", top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "a.pdf:p1:c0"
        assert results[1].chunk_id == "a.pdf:p1:c1"
        assert results[0].relevance_score == 0.9
        assert results[1].relevance_score == 0.7

    async def test_relevance_score_conversion(self):
        embed = AsyncMock(spec=OllamaEmbeddingClient)
        embed.embed_text.return_value = [1.0, 0.0, 0.0]
        store = MagicMock(spec=ChromaVectorStore)
        store.query.return_value = {
            "ids": [["c0"]],
            "documents": [["text"]],
            "metadatas": [[{"source": "x.pdf", "page_number": 1, "chunk_index": 0}]],
            "distances": [[0.75]],
        }
        service = _make_service(embed, store)
        results = await service.retrieve(query="test")
        assert len(results) == 1
        assert results[0].distance == 0.75
        assert results[0].relevance_score == 0.25

    async def test_empty_query_rejected(self):
        service = _make_service()
        with pytest.raises(RetrievalServiceError, match="empty"):
            await service.retrieve(query="")

    async def test_whitespace_query_rejected(self):
        service = _make_service()
        with pytest.raises(RetrievalServiceError, match="empty"):
            await service.retrieve(query="   \n\t   ")

    async def test_invalid_top_k_zero(self):
        service = _make_service()
        with pytest.raises(RetrievalServiceError, match="positive"):
            await service.retrieve(query="test", top_k=0)

    async def test_invalid_top_k_negative(self):
        service = _make_service()
        with pytest.raises(RetrievalServiceError, match="positive"):
            await service.retrieve(query="test", top_k=-1)

    async def test_minimum_relevance_too_low(self):
        service = _make_service()
        with pytest.raises(RetrievalServiceError, match="between"):
            await service.retrieve(query="test", minimum_relevance=-0.1)

    async def test_minimum_relevance_too_high(self):
        service = _make_service()
        with pytest.raises(RetrievalServiceError, match="between"):
            await service.retrieve(query="test", minimum_relevance=1.1)

    async def test_source_filter_passthrough(self):
        embed = AsyncMock(spec=OllamaEmbeddingClient)
        embed.embed_text.return_value = [1.0, 0.0, 0.0]
        store = MagicMock(spec=ChromaVectorStore)
        store.query.return_value = {
            "ids": [["b.pdf:p1:c0"]],
            "documents": [["text"]],
            "metadatas": [[{"source": "b.pdf", "page_number": 1, "chunk_index": 0}]],
            "distances": [[0.2]],
        }
        service = _make_service(embed, store)
        results = await service.retrieve(query="test", source="b.pdf")
        assert len(results) == 1
        assert results[0].source == "b.pdf"
        store.query.assert_called_with(
            query_embedding=[1.0, 0.0, 0.0], top_k=5, source="b.pdf"
        )

    async def test_embedding_failure(self):
        embed = AsyncMock(spec=OllamaEmbeddingClient)
        embed.embed_text.side_effect = EmbeddingClientError("No server")
        service = _make_service(embed, MagicMock(spec=ChromaVectorStore))
        with pytest.raises(RetrievalServiceError, match="Failed to embed"):
            await service.retrieve(query="test")

    async def test_vector_store_failure(self):
        embed = AsyncMock(spec=OllamaEmbeddingClient)
        embed.embed_text.return_value = [1.0, 0.0, 0.0]
        store = MagicMock(spec=ChromaVectorStore)
        store.query.side_effect = VectorStoreError("Query failed")
        service = _make_service(embed, store)
        with pytest.raises(RetrievalServiceError, match="Failed to query"):
            await service.retrieve(query="test")

    async def test_no_results(self):
        embed = AsyncMock(spec=OllamaEmbeddingClient)
        embed.embed_text.return_value = [1.0, 0.0, 0.0]
        store = MagicMock(spec=ChromaVectorStore)
        store.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        service = _make_service(embed, store)
        results = await service.retrieve(query="test")
        assert results == []

    async def test_malformed_nested_response(self):
        embed = AsyncMock(spec=OllamaEmbeddingClient)
        embed.embed_text.return_value = [1.0, 0.0, 0.0]
        store = MagicMock(spec=ChromaVectorStore)
        # Missing inner nesting
        store.query.return_value = {
            "ids": "not_a_list",
            "documents": [],
            "metadatas": [],
            "distances": [],
        }
        service = _make_service(embed, store)
        with pytest.raises(RetrievalServiceError, match="must be lists"):
            await service.retrieve(query="test")

    async def test_mismatched_lengths(self):
        embed = AsyncMock(spec=OllamaEmbeddingClient)
        embed.embed_text.return_value = [1.0, 0.0, 0.0]
        store = MagicMock(spec=ChromaVectorStore)
        store.query.return_value = {
            "ids": [["c0", "c1"]],
            "documents": [["text"]],
            "metadatas": [[{}]],
            "distances": [[0.1]],
        }
        service = _make_service(embed, store)
        with pytest.raises(RetrievalServiceError, match="Mismatched"):
            await service.retrieve(query="test")

    async def test_missing_source_metadata(self):
        embed = AsyncMock(spec=OllamaEmbeddingClient)
        embed.embed_text.return_value = [1.0, 0.0, 0.0]
        store = MagicMock(spec=ChromaVectorStore)
        store.query.return_value = {
            "ids": [["c0"]],
            "documents": [["text"]],
            "metadatas": [[{"page_number": 1}]],
            "distances": [[0.1]],
        }
        service = _make_service(embed, store)
        with pytest.raises(RetrievalServiceError, match="missing.*source"):
            await service.retrieve(query="test")

    async def test_missing_page_metadata(self):
        embed = AsyncMock(spec=OllamaEmbeddingClient)
        embed.embed_text.return_value = [1.0, 0.0, 0.0]
        store = MagicMock(spec=ChromaVectorStore)
        store.query.return_value = {
            "ids": [["c0"]],
            "documents": [["text"]],
            "metadatas": [[{"source": "x.pdf"}]],
            "distances": [[0.1]],
        }
        service = _make_service(embed, store)
        with pytest.raises(RetrievalServiceError, match="missing.*page_number"):
            await service.retrieve(query="test")

    async def test_invalid_distance(self):
        embed = AsyncMock(spec=OllamaEmbeddingClient)
        embed.embed_text.return_value = [1.0, 0.0, 0.0]
        store = MagicMock(spec=ChromaVectorStore)
        store.query.return_value = {
            "ids": [["c0"]],
            "documents": [["text"]],
            "metadatas": [[{"source": "x.pdf", "page_number": 1}]],
            "distances": [["not_a_number"]],
        }
        service = _make_service(embed, store)
        with pytest.raises(RetrievalServiceError, match="non-numeric distance"):
            await service.retrieve(query="test")

    async def test_minimum_relevance_filtering(self):
        embed = AsyncMock(spec=OllamaEmbeddingClient)
        embed.embed_text.return_value = [1.0, 0.0, 0.0]
        store = MagicMock(spec=ChromaVectorStore)
        store.query.return_value = {
            "ids": [["c0", "c1"]],
            "documents": [["a", "b"]],
            "metadatas": [
                [
                    {"source": "x.pdf", "page_number": 1, "chunk_index": 0},
                    {"source": "x.pdf", "page_number": 1, "chunk_index": 1},
                ],
            ],
            "distances": [[0.1, 0.9]],
        }
        service = _make_service(embed, store)
        results = await service.retrieve(query="test", minimum_relevance=0.5)
        assert len(results) == 1
        assert results[0].chunk_id == "c0"

    async def test_duplicate_chunk_ids(self):
        embed = AsyncMock(spec=OllamaEmbeddingClient)
        embed.embed_text.return_value = [1.0, 0.0, 0.0]
        store = MagicMock(spec=ChromaVectorStore)
        store.query.return_value = {
            "ids": [["c0", "c0"]],
            "documents": [["first", "second"]],
            "metadatas": [
                [
                    {"source": "x.pdf", "page_number": 1, "chunk_index": 0},
                    {"source": "x.pdf", "page_number": 1, "chunk_index": 0},
                ],
            ],
            "distances": [[0.1, 0.2]],
        }
        service = _make_service(embed, store)
        results = await service.retrieve(query="test")
        assert len(results) == 1
        assert results[0].text == "first"

    async def test_order_preserved(self):
        embed = AsyncMock(spec=OllamaEmbeddingClient)
        embed.embed_text.return_value = [1.0, 0.0, 0.0]
        store = MagicMock(spec=ChromaVectorStore)
        store.query.return_value = {
            "ids": [["c0", "c1", "c2"]],
            "documents": [["a", "b", "c"]],
            "metadatas": [
                [
                    {"source": "x.pdf", "page_number": 1, "chunk_index": 0},
                    {"source": "x.pdf", "page_number": 1, "chunk_index": 0},
                    {"source": "x.pdf", "page_number": 1, "chunk_index": 0},
                ],
            ],
            "distances": [[0.1, 0.5, 0.9]],
        }
        service = _make_service(embed, store)
        results = await service.retrieve(query="test")
        assert [r.chunk_id for r in results] == ["c0", "c1", "c2"]
        assert [r.relevance_score for r in results] == pytest.approx([0.9, 0.5, 0.1])


# ── format_context tests ────────────────────────────────────────────


class TestFormatContext:
    def test_empty_list(self):
        assert RetrievalService.format_context([]) == ""

    def test_rejects_non_positive_max(self):
        with pytest.raises(RetrievalServiceError, match="positive"):
            RetrievalService.format_context([], max_characters=0)

    def test_single_chunk(self):
        chunks = [_make_chunk(text="Hello world")]
        result = RetrievalService.format_context(chunks)
        assert "[Source: a.pdf, Page: 1]" in result
        assert "Hello world" in result

    def test_multiple_chunks_separated(self):
        chunks = [
            _make_chunk(chunk_id="a:p1:c0", text="First chunk", source="a.pdf"),
            _make_chunk(chunk_id="a:p1:c1", text="Second chunk", source="a.pdf"),
        ]
        result = RetrievalService.format_context(chunks)
        assert result.count("[Source:") == 2

    def test_character_limit(self):
        chunks = [
            _make_chunk(chunk_id="a:p1:c0", text="Short text", source="x.pdf"),
            _make_chunk(chunk_id="a:p1:c1", text="Another text", source="x.pdf"),
        ]
        # Set limit small enough to only fit the first chunk + header
        result = RetrievalService.format_context(chunks, max_characters=50)
        # Should include the first chunk's header and part of its text
        assert "[Source: x.pdf, Page: 1]" in result
        # Second chunk should NOT appear
        assert "Another text" not in result

    def test_first_chunk_truncated(self):
        chunks = [_make_chunk(text="A" * 2000, source="big.pdf")]
        result = RetrievalService.format_context(chunks, max_characters=100)
        assert "[Source: big.pdf, Page: 1]" in result
        # Should be truncated: header (24) + some text (max 76 chars)
        assert len(result) <= 100
        # Not the full 2000 chars
        assert "A" * 2000 not in result

    def test_header_not_cut(self):
        """If the first chunk exceeds the limit, the header must be
        fully included before any text is truncated."""
        chunks = [_make_chunk(text="B" * 5000, source="big.pdf")]
        # Very small limit — may need to truncate text, but header is intact
        result = RetrievalService.format_context(chunks, max_characters=40)
        assert "[Source: big.pdf, Page: 1]\n" in result
        assert len(result) <= 40

    def test_no_mutation(self):
        chunks = [_make_chunk(text="Test")]
        original_ids = [c.chunk_id for c in chunks]
        RetrievalService.format_context(chunks)
        assert [c.chunk_id for c in chunks] == original_ids
