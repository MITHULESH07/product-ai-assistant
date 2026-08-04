from pydantic import BaseModel, Field


class DocumentIngestionResponse(BaseModel):
    source: str
    pages_extracted: int
    chunks_created: int
    chunks_embedded: int
    chunks_stored: int
    previous_chunks_deleted: int
    collection_count: int
    message: str


class DocumentListResponse(BaseModel):
    sources: list[str]
    count: int
    stored_chunks: int


class DocumentDeleteResponse(BaseModel):
    source: str
    deleted_chunks: int
