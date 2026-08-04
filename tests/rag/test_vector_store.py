import math
import os
import tempfile

import pytest

from app.rag.embedding_client import EmbeddedChunk
from app.rag.vector_store import ChromaVectorStore, VectorStoreError

# ── helpers ─────────────────────────────────────────────────────────


_DEFAULT_EMBEDDING = [0.1, 0.2, 0.3]


def _make_chunk(
    chunk_id: str = "doc.pdf:p1:c0",
    text: str = "Some text content",
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
        embedding=_DEFAULT_EMBEDDING if embedding is None else embedding,
    )


@pytest.fixture
def temp_dir() -> str:
    td = tempfile.mkdtemp()
    yield td
    import shutil

    try:
        shutil.rmtree(td)
    except PermissionError:
        pass


@pytest.fixture
def store(temp_dir: str) -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_directory=temp_dir,
        collection_name="test_col",
    )


# ── initialization ──────────────────────────────────────────────────


class TestInit:
    def test_creates_directory(self):
        td = tempfile.mkdtemp()
        persist = os.path.join(td, "nested", "chroma")
        store = ChromaVectorStore(
            persist_directory=persist,
            collection_name="init_test",
        )
        assert os.path.isdir(persist)
        assert store.count() == 0
        import shutil

        shutil.rmtree(td, ignore_errors=True)

    def test_reuses_existing_collection(self, store):
        c1 = store.count()
        store.add_chunks([_make_chunk()])
        # Create a second store pointing to the same directory
        store2 = ChromaVectorStore(
            persist_directory=store._persist_directory,
            collection_name="test_col",
        )
        assert store2.count() == c1 + 1


# ── add_chunks ──────────────────────────────────────────────────────


class TestAddChunks:
    def test_empty_list(self, store):
        assert store.add_chunks([]) == 0

    def test_successful_insertion(self, store):
        chunks = [_make_chunk(chunk_id="a.pdf:p1:c0")]
        result = store.add_chunks(chunks)
        assert result == 1
        assert store.count() == 1

    def test_metadata_preserved(self, store):
        chunks = [
            _make_chunk(
                chunk_id="manual.pdf:p5:c2",
                text="Custom text",
                source="manual.pdf",
                page_number=5,
                chunk_index=2,
            )
        ]
        store.add_chunks(chunks)
        # Verify by checking sources and counts
        assert store.get_sources() == ["manual.pdf"]
        assert store.contains_chunk("manual.pdf:p5:c2")

    def test_multiple_chunks(self, store):
        chunks = [
            _make_chunk(chunk_id=f"doc.pdf:p1:c{i}", embedding=[0.1, 0.2, 0.3])
            for i in range(5)
        ]
        assert store.add_chunks(chunks) == 5
        assert store.count() == 5

    def test_upsert_updates_existing(self, store):
        c1 = _make_chunk(
            chunk_id="doc.pdf:p1:c0",
            text="Original text",
            embedding=[0.1, 0.2, 0.3],
        )
        store.add_chunks([c1])
        assert store.count() == 1

        # Re-ingest with same ID but different text
        c2 = _make_chunk(
            chunk_id="doc.pdf:p1:c0",
            text="Updated text",
            embedding=[0.4, 0.5, 0.6],
        )
        store.add_chunks([c2])
        # Count should still be 1 (upsert, not duplicate)
        assert store.count() == 1

    def test_empty_chunk_id_rejected(self, store):
        with pytest.raises(VectorStoreError, match="empty chunk_id"):
            store.add_chunks([_make_chunk(chunk_id="")])

    def test_empty_text_rejected(self, store):
        with pytest.raises(VectorStoreError, match="empty text"):
            store.add_chunks([_make_chunk(text="")])

    def test_whitespace_text_rejected(self, store):
        with pytest.raises(VectorStoreError, match="empty text"):
            store.add_chunks([_make_chunk(text="   ")])

    def test_empty_embedding_rejected(self, store):
        with pytest.raises(VectorStoreError, match="empty embedding"):
            store.add_chunks([_make_chunk(embedding=[])])

    def test_mismatched_dimensions(self, store):
        chunks = [
            _make_chunk(chunk_id="a.pdf:p1:c0", embedding=[0.1, 0.2, 0.3]),
            _make_chunk(chunk_id="a.pdf:p1:c1", embedding=[0.1, 0.2]),
        ]
        with pytest.raises(VectorStoreError, match="dimension"):
            store.add_chunks(chunks)

    def test_non_numeric_embedding(self, store):
        with pytest.raises(VectorStoreError, match="non-numeric"):
            store.add_chunks([_make_chunk(embedding=[0.1, "nan"])])

    def test_non_finite_embedding(self, store):
        with pytest.raises(VectorStoreError, match="non-finite"):
            store.add_chunks([_make_chunk(embedding=[0.1, math.inf])])

    def test_negative_infinity_rejected(self, store):
        with pytest.raises(VectorStoreError, match="non-finite"):
            store.add_chunks([_make_chunk(embedding=[0.1, -math.inf])])

    def test_nan_rejected(self, store):
        with pytest.raises(VectorStoreError, match="non-finite"):
            store.add_chunks([_make_chunk(embedding=[0.1, math.nan])])

    def test_batching_no_error(self, store):
        # Insert more than BATCH_SIZE chunks (BATCH_SIZE=100)
        chunks = [
            _make_chunk(
                chunk_id=f"big.pdf:p1:c{i}",
                embedding=[float(j) for j in range(10)],
            )
            for i in range(105)
        ]
        result = store.add_chunks(chunks)
        assert result == 105
        assert store.count() == 105


# ── count ───────────────────────────────────────────────────────────


class TestCount:
    def test_empty_collection(self, store):
        assert store.count() == 0

    def test_after_insertion(self, store):
        store.add_chunks([_make_chunk(), _make_chunk(chunk_id="doc.pdf:p1:c1")])
        assert store.count() == 2


# ── contains_chunk ──────────────────────────────────────────────────


class TestContainsChunk:
    def test_exists(self, store):
        store.add_chunks([_make_chunk(chunk_id="doc.pdf:p1:c0")])
        assert store.contains_chunk("doc.pdf:p1:c0") is True

    def test_missing(self, store):
        assert store.contains_chunk("nonexistent") is False

    def test_empty_id_rejected(self, store):
        with pytest.raises(VectorStoreError, match="non-empty"):
            store.contains_chunk("")

    def test_whitespace_id_rejected(self, store):
        with pytest.raises(VectorStoreError, match="non-empty"):
            store.contains_chunk("   ")


# ── get_sources ─────────────────────────────────────────────────────


class TestGetSources:
    def test_empty_collection(self, store):
        assert store.get_sources() == []

    def test_single_source(self, store):
        store.add_chunks([
            _make_chunk(chunk_id="a.pdf:p1:c0", source="a.pdf"),
            _make_chunk(chunk_id="a.pdf:p1:c1", source="a.pdf"),
        ])
        assert store.get_sources() == ["a.pdf"]

    def test_multiple_sources_sorted(self, store):
        store.add_chunks([
            _make_chunk(chunk_id="z.pdf:p1:c0", source="z.pdf"),
            _make_chunk(chunk_id="a.pdf:p1:c0", source="a.pdf"),
        ])
        assert store.get_sources() == ["a.pdf", "z.pdf"]

    def test_no_duplicate_sources(self, store):
        store.add_chunks([
            _make_chunk(chunk_id="x.pdf:p1:c0", source="x.pdf"),
            _make_chunk(chunk_id="x.pdf:p1:c1", source="x.pdf"),
        ])
        assert store.get_sources() == ["x.pdf"]


# ── delete_by_source ────────────────────────────────────────────────


class TestDeleteBySource:
    def test_delete_existing(self, store):
        store.add_chunks([
            _make_chunk(chunk_id="a.pdf:p1:c0", source="a.pdf"),
            _make_chunk(chunk_id="a.pdf:p1:c1", source="a.pdf"),
            _make_chunk(chunk_id="b.pdf:p1:c0", source="b.pdf"),
        ])
        deleted = store.delete_by_source("a.pdf")
        assert deleted == 2
        assert store.count() == 1
        assert store.contains_chunk("b.pdf:p1:c0") is True

    def test_delete_missing_source(self, store):
        store.add_chunks([_make_chunk(source="a.pdf")])
        deleted = store.delete_by_source("nonexistent.pdf")
        assert deleted == 0
        assert store.count() == 1

    def test_delete_other_source_unaffected(self, store):
        store.add_chunks([
            _make_chunk(chunk_id="a.pdf:p1:c0", source="a.pdf"),
            _make_chunk(chunk_id="b.pdf:p1:c0", source="b.pdf"),
        ])
        store.delete_by_source("a.pdf")
        assert store.contains_chunk("b.pdf:p1:c0") is True
        assert store.contains_chunk("a.pdf:p1:c0") is False

    def test_empty_source_rejected(self, store):
        with pytest.raises(VectorStoreError, match="empty source"):
            store.delete_by_source("")

    def test_whitespace_source_rejected(self, store):
        with pytest.raises(VectorStoreError, match="empty source"):
            store.delete_by_source("   ")

    def test_delete_empty_collection(self, store):
        deleted = store.delete_by_source("any.pdf")
        assert deleted == 0


# ── persistence ─────────────────────────────────────────────────────


class TestPersistence:
    def test_survives_store_reinit(self, temp_dir):
        s1 = ChromaVectorStore(persist_directory=temp_dir, collection_name="persist_test")
        s1.add_chunks([
            _make_chunk(chunk_id="persist.pdf:p1:c0", source="persist.pdf"),
        ])
        c1 = s1.count()

        s2 = ChromaVectorStore(persist_directory=temp_dir, collection_name="persist_test")
        c2 = s2.count()
        assert c2 == c1
        assert s2.contains_chunk("persist.pdf:p1:c0") is True
        assert s2.get_sources() == ["persist.pdf"]


# ── query ───────────────────────────────────────────────────────────


class TestQuery:
    def test_basic_query(self, store):
        store.add_chunks([
            _make_chunk(chunk_id="a.pdf:p1:c0", embedding=[1.0, 0.0, 0.0]),
            _make_chunk(chunk_id="b.pdf:p1:c0", embedding=[0.0, 1.0, 0.0]),
        ])
        result = store.query(query_embedding=[1.0, 0.0, 0.0], top_k=2)
        ids = result["ids"][0]
        assert ids[0] == "a.pdf:p1:c0"

    def test_source_filter(self, store):
        store.add_chunks([
            _make_chunk(chunk_id="a.pdf:p1:c0", source="a.pdf", embedding=[1.0, 0.0, 0.0]),
            _make_chunk(chunk_id="b.pdf:p1:c0", source="b.pdf", embedding=[0.0, 0.0, 1.0]),
        ])
        result = store.query(query_embedding=[1.0, 0.0, 0.0], top_k=2, source="a.pdf")
        ids = result["ids"][0]
        assert ids == ["a.pdf:p1:c0"]

    def test_empty_embedding_rejected(self, store):
        with pytest.raises(VectorStoreError, match="non-empty"):
            store.query(query_embedding=[], top_k=5)

    def test_non_numeric_embedding_rejected(self, store):
        with pytest.raises(VectorStoreError, match="non-numeric"):
            store.query(query_embedding=["bad"], top_k=5)

    def test_non_finite_embedding_rejected(self, store):
        with pytest.raises(VectorStoreError, match="non-finite"):
            store.query(query_embedding=[float("inf")], top_k=5)

    def test_top_k_zero_rejected(self, store):
        with pytest.raises(VectorStoreError, match="positive"):
            store.query(query_embedding=[0.1], top_k=0)

    def test_top_k_negative_rejected(self, store):
        with pytest.raises(VectorStoreError, match="positive"):
            store.query(query_embedding=[0.1], top_k=-1)

    def test_no_results(self, store):
        result = store.query(query_embedding=[1.0, 0.0, 0.0], top_k=5)
        assert result["ids"] == [[]]
        assert result["distances"] == [[]]
        assert result["documents"] == [[]]
        assert result["metadatas"] == [[]]

    def test_includes_required_keys(self, store):
        store.add_chunks([
            _make_chunk(chunk_id="x.pdf:p1:c0", embedding=[1.0, 0.0, 0.0]),
        ])
        result = store.query(query_embedding=[1.0, 0.0, 0.0], top_k=1)
        assert set(result.keys()) >= {"ids", "documents", "metadatas", "distances"}
