import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.rag.document_loader import DocumentLoader
from app.rag.embedding_client import OllamaEmbeddingClient
from app.rag.ingestion_service import DocumentIngestionService, IngestionServiceError
from app.rag.text_chunker import TextChunker
from app.rag.vector_store import ChromaVectorStore, VectorStoreError
from app.schemas.documents import DocumentDeleteResponse, DocumentIngestionResponse, DocumentListResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"],
)

_ingestion_service: DocumentIngestionService | None = None


def get_document_ingestion_service() -> DocumentIngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = DocumentIngestionService(
            document_loader=DocumentLoader(),
            text_chunker=TextChunker(),
            embedding_client=OllamaEmbeddingClient(),
            vector_store=ChromaVectorStore(),
        )
    return _ingestion_service


_ALLOWED_EXTENSIONS = frozenset({".pdf"})


def _sanitize_filename(filename: str) -> str:
    return Path(filename).name


def _validate_pdf_upload(filename: str, content_type: str | None) -> str:
    if not filename or not filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    sanitized = _sanitize_filename(filename)
    ext = Path(sanitized).suffix.lower()

    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file extension '{ext}'. Only PDF files are accepted.",
        )

    if content_type and content_type not in (
        "application/pdf",
        "application/octet-stream",
        "",
        None,
    ):
        logger.warning("Unexpected content type for PDF upload: '%s'", content_type)

    return sanitized


@router.post(
    "/ingest",
    response_model=DocumentIngestionResponse,
    summary="Upload and ingest a product-manual PDF",
    description=(
        "Accepts a PDF file via multipart form-data, runs the full ingestion "
        "pipeline (load, chunk, embed, store), and returns indexing results. "
        "The uploaded file is saved to a temporary location and removed after "
        "processing."
    ),
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF file to ingest"),
    replace_existing: bool = Form(True, description="Replace existing chunks for this source"),
    ingestion_service: DocumentIngestionService = Depends(get_document_ingestion_service),
) -> DocumentIngestionResponse:
    sanitized_filename = _validate_pdf_upload(
        filename=file.filename or "",
        content_type=file.content_type,
    )

    max_bytes = settings.max_pdf_upload_size_mb * 1024 * 1024
    temp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            temp_path = tmp.name
            bytes_written = 0
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Upload exceeds maximum size of {settings.max_pdf_upload_size_mb} MB.",
                    )
                tmp.write(chunk)

        logger.info(
            "Upload saved — original=%s temp=%s size=%d replace_existing=%s",
            sanitized_filename,
            temp_path,
            bytes_written,
            replace_existing,
        )

        if bytes_written == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        result = await ingestion_service.ingest_pdf(
            pdf_path=temp_path,
            replace_existing=replace_existing,
            source_name=sanitized_filename,
        )

        logger.info(
            "Ingestion complete — source=%s pages=%d chunks=%d stored=%d",
            result.source,
            result.pages_extracted,
            result.chunks_created,
            result.chunks_stored,
        )

        return DocumentIngestionResponse(
            source=result.source,
            pages_extracted=result.pages_extracted,
            chunks_created=result.chunks_created,
            chunks_embedded=result.chunks_embedded,
            chunks_stored=result.chunks_stored,
            previous_chunks_deleted=result.previous_chunks_deleted,
            collection_count=result.collection_count,
            message=f"Successfully ingested '{result.source}'.",
        )

    except HTTPException:
        raise

    except IngestionServiceError as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        detail = str(exc)
        if any(kw in detail for kw in ("Failed to load", "encrypted", "not a PDF", "no extractable", "No chunks")):
            raise HTTPException(status_code=422, detail=detail) from exc
        if "Failed to embed" in detail:
            raise HTTPException(
                status_code=503,
                detail="Embedding service unavailable. Check Ollama connection.",
            ) from exc
        if "Failed to store" in detail or "Failed to delete" in detail:
            raise HTTPException(status_code=500, detail="Vector store error. Please try again.") from exc
        raise HTTPException(status_code=422, detail=detail) from exc

    except Exception as exc:
        logger.error("Unexpected error during ingestion: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from exc

    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
                logger.debug("Temporary file removed: %s", temp_path)
            except OSError as exc:
                logger.warning("Failed to remove temp file '%s': %s", temp_path, exc)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List indexed document sources",
    description="Returns all unique source filenames and the total chunk count in the vector store.",
)
async def list_documents(
    ingestion_service: DocumentIngestionService = Depends(get_document_ingestion_service),
) -> DocumentListResponse:
    try:
        sources = ingestion_service.store.get_sources()
        stored_chunks = ingestion_service.store.count()
    except VectorStoreError as exc:
        logger.error("Failed to list documents: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve document list.") from exc

    return DocumentListResponse(
        sources=sources,
        count=len(sources),
        stored_chunks=stored_chunks,
    )


@router.delete(
    "/{source}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document source by name",
    description="Deletes all indexed chunks for the given source filename.",
)
async def delete_document(
    source: str,
    ingestion_service: DocumentIngestionService = Depends(get_document_ingestion_service),
) -> DocumentDeleteResponse:
    sanitized = _sanitize_filename(source)

    if not sanitized or not sanitized.strip():
        raise HTTPException(status_code=400, detail="Source name is required.")

    try:
        count_before = ingestion_service.store.count()
        deleted = ingestion_service.store.delete_by_source(sanitized)
        count_after = ingestion_service.store.count()
    except VectorStoreError as exc:
        logger.error("Failed to delete source '%s': %s", sanitized, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete document.") from exc

    if deleted == 0 and count_before == count_after:
        raise HTTPException(
            status_code=404,
            detail=f"Source '{sanitized}' not found.",
        )

    logger.info("Deleted source='%s' — removed %d chunk(s)", sanitized, deleted)

    return DocumentDeleteResponse(
        source=sanitized,
        deleted_chunks=deleted,
    )
