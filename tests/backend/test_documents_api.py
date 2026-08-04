import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes.documents import get_document_ingestion_service
from app.rag.document_loader import DocumentLoader
from app.rag.embedding_client import OllamaEmbeddingClient
from app.rag.ingestion_service import DocumentIngestionService, IngestionResult, IngestionServiceError
from app.rag.text_chunker import TextChunker
from app.rag.vector_store import ChromaVectorStore, VectorStoreError

os.environ.setdefault("OLLAMA_BASE_URL", "http://test:11434")
os.environ.setdefault("OLLAMA_MODEL", "test-model")

from app.core.config import settings  # noqa: E402

try:
    settings.validate()
except RuntimeError:
    pass

client = TestClient(app)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def mock_ingestion_result():
    return IngestionResult(
        source="manual.pdf",
        pages_extracted=5,
        chunks_created=12,
        chunks_embedded=12,
        chunks_stored=12,
        previous_chunks_deleted=3,
        collection_count=25,
    )


@pytest.fixture
def mock_ingestion_service(mock_ingestion_result):
    svc = AsyncMock(spec=DocumentIngestionService)
    svc.ingest_pdf.return_value = mock_ingestion_result

    store = MagicMock(spec=ChromaVectorStore)
    store.get_sources.return_value = ["manual.pdf", "datasheet.pdf"]
    store.count.return_value = 42
    store.delete_by_source.return_value = 5
    svc.store = store

    return svc


@pytest.fixture
def override_service(mock_ingestion_service):
    """Override the ingestion service dependency for all tests.

    This approach uses dependency override which is cleaner than
    monkey-patching module-level objects.
    """

    def _override():
        return mock_ingestion_service

    app.dependency_overrides[get_document_ingestion_service] = _override
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def pdf_bytes():
    return b"%PDF-1.4 minimal fake pdf content for testing\n"


# ── POST /api/documents/ingest ──────────────────────────────────


class TestIngestDocument:
    def test_success(self, override_service, mock_ingestion_service, mock_ingestion_result, pdf_bytes):
        response = client.post(
            "/api/documents/ingest",
            files={"file": ("manual.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "manual.pdf"
        assert data["pages_extracted"] == 5
        assert data["chunks_created"] == 12
        assert data["chunks_embedded"] == 12
        assert data["chunks_stored"] == 12
        assert data["previous_chunks_deleted"] == 3
        assert data["collection_count"] == 25
        assert "Successfully ingested" in data["message"]

        call_kwargs = mock_ingestion_service.ingest_pdf.call_args.kwargs
        assert call_kwargs["source_name"] == "manual.pdf"
        assert call_kwargs["replace_existing"] is True
        assert call_kwargs["pdf_path"] is not None
        assert call_kwargs["pdf_path"].endswith(".pdf")

    def test_replace_existing_true(self, override_service, mock_ingestion_service, pdf_bytes):
        client.post(
            "/api/documents/ingest",
            files={"file": ("doc.pdf", pdf_bytes, "application/pdf")},
            data={"replace_existing": "true"},
        )
        kwargs = mock_ingestion_service.ingest_pdf.call_args.kwargs
        assert kwargs["replace_existing"] is True

    def test_replace_existing_false(self, override_service, mock_ingestion_service, pdf_bytes):
        client.post(
            "/api/documents/ingest",
            files={"file": ("doc.pdf", pdf_bytes, "application/pdf")},
            data={"replace_existing": "false"},
        )
        kwargs = mock_ingestion_service.ingest_pdf.call_args.kwargs
        assert kwargs["replace_existing"] is False

    def test_uppercase_pdf_extension(self, override_service, pdf_bytes):
        response = client.post(
            "/api/documents/ingest",
            files={"file": ("MANUAL.PDF", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 200

    def test_missing_filename(self, override_service, pdf_bytes):
        response = client.post(
            "/api/documents/ingest",
            files={"file": ("", pdf_bytes, "application/pdf")},
        )
        # FastAPI rejects empty filename via form validation (422) or route
        # validation (400). Accept either.
        assert response.status_code in (400, 422)
        detail = response.json().get("detail", "")
        if response.status_code == 400:
            assert "Filename is required" in detail

    def test_non_pdf_extension(self, override_service, pdf_bytes):
        response = client.post(
            "/api/documents/ingest",
            files={"file": ("readme.txt", pdf_bytes, "text/plain")},
        )
        assert response.status_code == 415
        assert "Unsupported file extension" in response.json()["detail"]

    def test_empty_upload(self, override_service, pdf_bytes):
        response = client.post(
            "/api/documents/ingest",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_upload_exceeds_size_limit(self, override_service):
        large_content = b"x" * (settings.max_pdf_upload_size_mb * 1024 * 1024 + 1)
        response = client.post(
            "/api/documents/ingest",
            files={"file": ("large.pdf", large_content, "application/pdf")},
        )
        assert response.status_code == 413
        assert "exceeds maximum size" in response.json()["detail"].lower()

    @patch.dict(os.environ, {"MAX_PDF_UPLOAD_SIZE_MB": "1"})
    def test_smaller_size_limit(self, override_service):
        # Re-validate settings to pick up env change
        settings._validated = False
        try:
            settings.validate()
        except RuntimeError:
            pass

        content = b"x" * (2 * 1024 * 1024)  # 2 MB
        response = client.post(
            "/api/documents/ingest",
            files={"file": ("big.pdf", content, "application/pdf")},
        )
        assert response.status_code == 413
        assert "exceeds maximum size" in response.json()["detail"].lower()

        # Restore setting
        os.environ.pop("MAX_PDF_UPLOAD_SIZE_MB", None)
        settings._validated = False
        try:
            settings.validate()
        except RuntimeError:
            pass

    def test_ingestion_validation_failure_load(self, override_service, mock_ingestion_service, pdf_bytes):
        mock_ingestion_service.ingest_pdf.side_effect = IngestionServiceError(
            "Failed to load PDF 'bad.pdf': Cannot read PDF file"
        )
        response = client.post(
            "/api/documents/ingest",
            files={"file": ("bad.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 422
        assert "Failed to load" in response.json()["detail"]

    def test_ingestion_validation_failure_encrypted(self, override_service, mock_ingestion_service, pdf_bytes):
        mock_ingestion_service.ingest_pdf.side_effect = IngestionServiceError(
            "PDF file 'secret.pdf' is encrypted"
        )
        response = client.post(
            "/api/documents/ingest",
            files={"file": ("secret.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 422
        assert "encrypted" in response.json()["detail"]

    def test_ingestion_validation_failure_no_text(self, override_service, mock_ingestion_service, pdf_bytes):
        mock_ingestion_service.ingest_pdf.side_effect = IngestionServiceError(
            "PDF 'blank.pdf' contains no extractable text pages"
        )
        response = client.post(
            "/api/documents/ingest",
            files={"file": ("blank.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 422
        assert "no extractable" in response.json()["detail"]

    def test_ingestion_validation_failure_no_chunks(self, override_service, mock_ingestion_service, pdf_bytes):
        mock_ingestion_service.ingest_pdf.side_effect = IngestionServiceError(
            "No chunks produced from 'bad.pdf'"
        )
        response = client.post(
            "/api/documents/ingest",
            files={"file": ("bad.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 422

    def test_embedding_failure(self, override_service, mock_ingestion_service, pdf_bytes):
        mock_ingestion_service.ingest_pdf.side_effect = IngestionServiceError(
            "Failed to embed chunks for 'doc.pdf': Connection refused"
        )
        response = client.post(
            "/api/documents/ingest",
            files={"file": ("doc.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 503
        assert "Embedding service unavailable" in response.json()["detail"]

    def test_vector_store_failure(self, override_service, mock_ingestion_service, pdf_bytes):
        mock_ingestion_service.ingest_pdf.side_effect = IngestionServiceError(
            "Failed to store embedded chunks for 'doc.pdf'."
        )
        response = client.post(
            "/api/documents/ingest",
            files={"file": ("doc.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 500
        assert "Vector store error" in response.json()["detail"]

    def test_delete_failure(self, override_service, mock_ingestion_service, pdf_bytes):
        mock_ingestion_service.ingest_pdf.side_effect = IngestionServiceError(
            "Failed to delete existing source 'doc.pdf': Chroma error"
        )
        response = client.post(
            "/api/documents/ingest",
            files={"file": ("doc.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 500

    def test_unexpected_failure(self, override_service, mock_ingestion_service, pdf_bytes):
        mock_ingestion_service.ingest_pdf.side_effect = RuntimeError("Something weird happened")
        response = client.post(
            "/api/documents/ingest",
            files={"file": ("doc.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 500
        assert "unexpected error" in response.json()["detail"].lower()

    def test_not_a_pdf_extension(self, override_service, mock_ingestion_service, pdf_bytes):
        mock_ingestion_service.ingest_pdf.side_effect = IngestionServiceError(
            "Failed to load PDF 'bad.pdf': is not a PDF (got extension '.txt')"
        )
        response = client.post(
            "/api/documents/ingest",
            files={"file": ("bad.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 422

    def test_response_field_mapping(self, override_service, mock_ingestion_service, mock_ingestion_result, pdf_bytes):
        response = client.post(
            "/api/documents/ingest",
            files={"file": ("manual.pdf", pdf_bytes, "application/pdf")},
        )
        data = response.json()
        assert set(data.keys()) == {
            "source",
            "pages_extracted",
            "chunks_created",
            "chunks_embedded",
            "chunks_stored",
            "previous_chunks_deleted",
            "collection_count",
            "message",
        }

    def test_temp_file_cleaned_after_success(self, override_service, pdf_bytes):
        import tempfile
        original_named = tempfile.NamedTemporaryFile

        created_paths = []

        def tracking_named(*args, **kwargs):
            tmp = original_named(*args, **kwargs)
            created_paths.append(tmp.name)
            return tmp

        with patch("app.api.routes.documents.tempfile.NamedTemporaryFile", tracking_named):
            response = client.post(
                "/api/documents/ingest",
                files={"file": ("manual.pdf", pdf_bytes, "application/pdf")},
            )

        assert response.status_code == 200
        for path in created_paths:
            assert not os.path.exists(path), f"Temp file not cleaned: {path}"

    def test_temp_file_cleaned_after_failure(self, override_service, mock_ingestion_service, pdf_bytes):
        mock_ingestion_service.ingest_pdf.side_effect = RuntimeError("fail")

        import tempfile
        original_named = tempfile.NamedTemporaryFile

        created_paths = []

        def tracking_named(*args, **kwargs):
            tmp = original_named(*args, **kwargs)
            created_paths.append(tmp.name)
            return tmp

        with patch("app.api.routes.documents.tempfile.NamedTemporaryFile", tracking_named):
            response = client.post(
                "/api/documents/ingest",
                files={"file": ("manual.pdf", pdf_bytes, "application/pdf")},
            )

        assert response.status_code == 500
        for path in created_paths:
            assert not os.path.exists(path), f"Temp file not cleaned: {path}"


# ── GET /api/documents ──────────────────────────────────────────


class TestListDocuments:
    def test_list_documents(self, override_service, mock_ingestion_service):
        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["sources"] == ["manual.pdf", "datasheet.pdf"]
        assert data["count"] == 2
        assert data["stored_chunks"] == 42

    def test_list_documents_empty(self, override_service, mock_ingestion_service):
        mock_ingestion_service.store.get_sources.return_value = []
        mock_ingestion_service.store.count.return_value = 0

        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["sources"] == []
        assert data["count"] == 0
        assert data["stored_chunks"] == 0

    def test_list_documents_store_failure(self, override_service, mock_ingestion_service):
        mock_ingestion_service.store.get_sources.side_effect = VectorStoreError("DB down")

        response = client.get("/api/documents")
        assert response.status_code == 500
        assert "Failed to retrieve" in response.json()["detail"]


# ── DELETE /api/documents/{source} ──────────────────────────────


class TestDeleteDocument:
    def test_delete_existing(self, override_service, mock_ingestion_service):
        response = client.delete("/api/documents/manual.pdf")
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "manual.pdf"
        assert data["deleted_chunks"] == 5

        mock_ingestion_service.store.delete_by_source.assert_called_once_with("manual.pdf")

    def test_delete_non_existent(self, override_service, mock_ingestion_service):
        mock_ingestion_service.store.delete_by_source.return_value = 0
        mock_ingestion_service.store.count.side_effect = [10, 10]  # count_before == count_after

        response = client.delete("/api/documents/nonexistent.pdf")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_delete_path_traversal_sanitized(self):
        from app.api.routes.documents import _sanitize_filename
        assert _sanitize_filename("../../secret.pdf") == "secret.pdf"
        assert _sanitize_filename("foo/bar/doc.pdf") == "doc.pdf"
        assert _sanitize_filename("normal.pdf") == "normal.pdf"

    def test_delete_store_failure(self, override_service, mock_ingestion_service):
        mock_ingestion_service.store.delete_by_source.side_effect = VectorStoreError("DB error")

        response = client.delete("/api/documents/manual.pdf")
        assert response.status_code == 500

    def test_delete_empty_source(self, override_service, mock_ingestion_service):
        # Trailing slash matches the GET route (empty source), not DELETE
        response = client.delete("/api/documents/")
        assert response.status_code == 405
