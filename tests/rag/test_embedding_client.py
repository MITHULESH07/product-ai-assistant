import os
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

# Ensure settings are validated before importing the client
os.environ.setdefault("OLLAMA_BASE_URL", "http://test:11434")
os.environ.setdefault("OLLAMA_MODEL", "test-model")
os.environ.setdefault("OLLAMA_EMBEDDING_MODEL", "test-embed-model")

from app.core.config import settings  # noqa: E402

try:
    settings.validate()
except RuntimeError:
    pass

from app.rag.embedding_client import (  # noqa: E402
    EmbeddedChunk,
    EmbeddingClientError,
    OllamaEmbeddingClient,
)
from app.rag.text_chunker import DocumentChunk  # noqa: E402

pytestmark = pytest.mark.asyncio

# ── helpers ─────────────────────────────────────────────────────────

PATCH_TARGET = "app.rag.embedding_client.httpx.AsyncClient"

SAMPLE_EMBEDDING = [0.1, 0.2, 0.3, 0.4, 0.5]


def _mock_embed_response(embedding: list[float] | None = None) -> dict:
    return {
        "model": "test-embed-model",
        "embeddings": [embedding or SAMPLE_EMBEDDING],
    }


def _make_mock_http_client(
    response_data: dict | None = None,
    status_code: int = 200,
    side_effect: Exception | None = None,
) -> AsyncMock:
    mock_response = Mock()
    mock_response.status_code = status_code

    if side_effect is not None:
        mock_response.raise_for_status.side_effect = side_effect
    else:
        mock_response.raise_for_status.return_value = None

    mock_response.json.return_value = response_data or _mock_embed_response()
    mock_response.text = str(response_data or _mock_embed_response())

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)

    if side_effect is not None and isinstance(
        side_effect, (httpx.ConnectError, httpx.TimeoutException)
    ):
        mock_http.post.side_effect = side_effect

    return mock_http


def _make_chunk(
    text: str = "Some text to embed",
    chunk_id: str = "doc.pdf:p1:c0",
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


@pytest.fixture
def client() -> OllamaEmbeddingClient:
    return OllamaEmbeddingClient(max_concurrency=2, max_retries=0)


@pytest.fixture
def mock_http() -> AsyncMock:
    return _make_mock_http_client()


def _patch_client(mock_http: AsyncMock):
    patcher = patch(PATCH_TARGET)
    mock_cls = patcher.start()
    mock_cls.return_value.__aenter__.return_value = mock_http
    return patcher


# ── embed_text tests ────────────────────────────────────────────────


class TestEmbedText:
    async def test_success(self, client, mock_http):
        patcher = _patch_client(mock_http)
        try:
            result = await client.embed_text("Hello world")
            assert result == SAMPLE_EMBEDDING
            assert all(isinstance(v, float) for v in result)
        finally:
            patcher.stop()

    async def test_empty_text_rejected(self, client):
        with pytest.raises(EmbeddingClientError, match="empty or whitespace-only"):
            await client.embed_text("")

    async def test_whitespace_text_rejected(self, client):
        with pytest.raises(EmbeddingClientError, match="empty or whitespace-only"):
            await client.embed_text("   \t\n  ")

    async def test_connection_failure(self, client):
        mock_http = AsyncMock()
        mock_http.post.side_effect = httpx.ConnectError("Connection refused")
        patcher = _patch_client(mock_http)
        try:
            with pytest.raises(EmbeddingClientError, match="Cannot connect"):
                await client.embed_text("Hello")
        finally:
            patcher.stop()

    async def test_timeout(self, client):
        mock_http = AsyncMock()
        mock_http.post.side_effect = httpx.TimeoutException("Timed out")
        patcher = _patch_client(mock_http)
        try:
            with pytest.raises(EmbeddingClientError, match="did not respond"):
                await client.embed_text("Hello")
        finally:
            patcher.stop()

    async def test_model_not_found(self, client):
        error_body = {"error": "model 'x' not found"}
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=Mock(), response=mock_response
        )
        mock_response.json.return_value = error_body
        mock_response.text = str(error_body)
        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        patcher = _patch_client(mock_http)
        try:
            with pytest.raises(EmbeddingClientError, match="not available"):
                await client.embed_text("Hello")
        finally:
            patcher.stop()

    async def test_http_500(self, client):
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=Mock(), response=mock_response
        )
        mock_response.text = "Internal error"
        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        patcher = _patch_client(mock_http)
        try:
            with pytest.raises(EmbeddingClientError, match="500"):
                await client.embed_text("Hello")
        finally:
            patcher.stop()

    async def test_malformed_json(self, client):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("Not JSON")
        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        patcher = _patch_client(mock_http)
        try:
            with pytest.raises(EmbeddingClientError):
                await client.embed_text("Hello")
        finally:
            patcher.stop()

    async def test_missing_embeddings_field(self, client, mock_http):
        mock_http.post.return_value.json.return_value = {"model": "x"}
        patcher = _patch_client(mock_http)
        try:
            with pytest.raises(EmbeddingClientError, match="missing.*embeddings"):
                await client.embed_text("Hello")
        finally:
            patcher.stop()

    async def test_empty_embeddings_list(self, client, mock_http):
        mock_http.post.return_value.json.return_value = {
            "model": "x",
            "embeddings": [],
        }
        patcher = _patch_client(mock_http)
        try:
            with pytest.raises(EmbeddingClientError, match="empty"):
                await client.embed_text("Hello")
        finally:
            patcher.stop()

    async def test_empty_vector(self, client, mock_http):
        mock_http.post.return_value.json.return_value = {
            "model": "x",
            "embeddings": [[]],
        }
        patcher = _patch_client(mock_http)
        try:
            with pytest.raises(EmbeddingClientError, match="empty embedding vector"):
                await client.embed_text("Hello")
        finally:
            patcher.stop()

    async def test_non_numeric_value(self, client, mock_http):
        mock_http.post.return_value.json.return_value = {
            "model": "x",
            "embeddings": [[0.1, "NaN"]],
        }
        patcher = _patch_client(mock_http)
        try:
            with pytest.raises(EmbeddingClientError, match="Non-numeric"):
                await client.embed_text("Hello")
        finally:
            patcher.stop()

    async def test_non_finite_value(self, client, mock_http):
        import math

        mock_http.post.return_value.json.return_value = {
            "model": "x",
            "embeddings": [[0.1, math.inf]],
        }
        patcher = _patch_client(mock_http)
        try:
            with pytest.raises(EmbeddingClientError, match="Non-finite"):
                await client.embed_text("Hello")
        finally:
            patcher.stop()


# ── embed_chunk tests ───────────────────────────────────────────────


class TestEmbedChunk:
    async def test_success(self, client, mock_http):
        chunk = _make_chunk()
        patcher = _patch_client(mock_http)
        try:
            result = await client.embed_chunk(chunk)
            assert isinstance(result, EmbeddedChunk)
            assert result.chunk_id == "doc.pdf:p1:c0"
            assert result.text == chunk.text
            assert result.source == "doc.pdf"
            assert result.page_number == 1
            assert result.chunk_index == 0
            assert result.embedding == SAMPLE_EMBEDDING
        finally:
            patcher.stop()

    async def test_metadata_preserved(self, client, mock_http):
        chunk = _make_chunk(
            text="Custom text",
            chunk_id="manual.pdf:p5:c2",
            source="manual.pdf",
            page_number=5,
            chunk_index=2,
        )
        patcher = _patch_client(mock_http)
        try:
            result = await client.embed_chunk(chunk)
            assert result.chunk_id == "manual.pdf:p5:c2"
            assert result.text == "Custom text"
            assert result.source == "manual.pdf"
            assert result.page_number == 5
            assert result.chunk_index == 2
        finally:
            patcher.stop()


# ── embed_chunks tests ──────────────────────────────────────────────


class TestEmbedChunks:
    async def test_success_batch(self, client, mock_http):
        chunks = [
            _make_chunk(text="First", chunk_id="doc.pdf:p1:c0"),
            _make_chunk(text="Second", chunk_id="doc.pdf:p1:c1"),
        ]
        patcher = _patch_client(mock_http)
        try:
            results = await client.embed_chunks(chunks)
            assert len(results) == 2
            assert results[0].chunk_id == "doc.pdf:p1:c0"
            assert results[1].chunk_id == "doc.pdf:p1:c1"
        finally:
            patcher.stop()

    async def test_empty_list(self, client):
        results = await client.embed_chunks([])
        assert results == []

    async def test_ordering_preserved(self, client, mock_http):
        chunks = [
            _make_chunk(text="A", chunk_id="doc.pdf:p1:c0"),
            _make_chunk(text="B", chunk_id="doc.pdf:p1:c1"),
            _make_chunk(text="C", chunk_id="doc.pdf:p1:c2"),
        ]
        patcher = _patch_client(mock_http)
        try:
            results = await client.embed_chunks(chunks)
            ids = [r.chunk_id for r in results]
            assert ids == ["doc.pdf:p1:c0", "doc.pdf:p1:c1", "doc.pdf:p1:c2"]
        finally:
            patcher.stop()

    async def test_failure_reports_chunk_id(self, client, mock_http):
        chunks = [
            _make_chunk(text="Good", chunk_id="doc.pdf:p1:c0"),
            _make_chunk(text="", chunk_id="doc.pdf:p1:c1"),
            _make_chunk(text="Also good", chunk_id="doc.pdf:p1:c2"),
        ]
        patcher = _patch_client(mock_http)
        try:
            with pytest.raises(EmbeddingClientError, match="doc.pdf:p1:c1"):
                await client.embed_chunks(chunks)
        finally:
            patcher.stop()

    async def test_empty_chunk_propagates_error(self, client):
        chunks = [
            _make_chunk(text="", chunk_id="doc.pdf:p1:c0"),
        ]
        with pytest.raises(EmbeddingClientError):
            await client.embed_chunks(chunks)
