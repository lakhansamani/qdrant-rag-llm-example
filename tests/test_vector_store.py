"""
tests/test_vector_store.py
--------------------------
Unit tests for VectorStore and Retriever.

These tests use in-memory Qdrant (:memory:) and mock embeddings so they run
instantly without needing to download any model or start any server.

Run with:
    pytest tests/test_vector_store.py -v
"""

import numpy as np
import pytest

from src.vector_store import VectorStore
from src.retriever import Retriever, DocumentChunk, RetrievedChunk


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def store():
    """Provide a fresh in-memory VectorStore for each test."""
    s = VectorStore(collection="test_col", vector_size=4, path=":memory:")
    yield s
    # Teardown: drop the collection after each test
    s.delete_collection()


class MockEmbedder:
    """
    A minimal fake embedder that returns fixed vectors for known inputs.

    Using mock embedders in tests:
    - Eliminates network dependency for model downloads
    - Makes tests deterministic and fast
    - Isolates the code under test from external dependencies
    """

    vector_size = 4

    # Deterministic mapping from text snippets to 4-dim vectors
    _VECTORS = {
        "security": np.array([1.0, 0.0, 0.0, 0.0]),
        "onboarding": np.array([0.0, 1.0, 0.0, 0.0]),
        "tech": np.array([0.0, 0.0, 1.0, 0.0]),
        "other": np.array([0.0, 0.0, 0.0, 1.0]),
    }

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Return a fixed vector based on keywords in the text."""
        results = []
        for text in texts:
            text_lower = text.lower()
            # Find which category this text belongs to
            vec = self._VECTORS["other"]
            for key, v in self._VECTORS.items():
                if key in text_lower:
                    vec = v
                    break
            results.append(vec.copy())
        return results

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

    def embed_iter(self, texts, batch_size=64):
        yield from self.embed(texts)


# ── VectorStore Tests ─────────────────────────────────────────────────────────

class TestVectorStore:
    """Tests for VectorStore: upsert, search, count, filters."""

    def test_store_starts_empty(self, store):
        """A freshly created collection should contain zero points."""
        assert store.count() == 0

    def test_upsert_increases_count(self, store):
        """After upserting N vectors, count() should return N."""
        vectors = [np.array([1.0, 0.0, 0.0, 0.0]) for _ in range(5)]
        payloads = [{"text": f"doc {i}", "source": "test.txt", "chunk_index": i} for i in range(5)]
        store.upsert(vectors=vectors, payloads=payloads)
        assert store.count() == 5

    def test_upsert_empty_list_no_error(self, store):
        """Upserting an empty list should be a no-op."""
        store.upsert(vectors=[], payloads=[])
        assert store.count() == 0

    def test_search_returns_correct_top_k(self, store):
        """search() should return at most top_k results."""
        # Insert 10 vectors
        vectors = [np.random.rand(4).astype(np.float32) for _ in range(10)]
        payloads = [{"text": f"doc {i}", "source": "x.txt", "chunk_index": i} for i in range(10)]
        store.upsert(vectors=vectors, payloads=payloads)

        results = store.search(query_vector=vectors[0], top_k=3)
        assert len(results) <= 3

    def test_search_returns_scored_results(self, store):
        """Search results should have a .score attribute between 0 and 1."""
        vec = np.array([1.0, 0.0, 0.0, 0.0])
        store.upsert(vectors=[vec], payloads=[{"text": "test", "source": "a.txt", "chunk_index": 0}])
        results = store.search(query_vector=vec, top_k=1)
        assert len(results) == 1
        assert 0.0 <= results[0].score <= 1.0 + 1e-6  # Allow small float rounding

    def test_exact_match_has_highest_score(self, store):
        """Searching with the exact same vector should give the highest score."""
        v1 = np.array([1.0, 0.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0, 0.0])

        store.upsert(
            vectors=[v1, v2, v3],
            payloads=[
                {"text": "security", "source": "a.txt", "chunk_index": 0},
                {"text": "onboarding", "source": "b.txt", "chunk_index": 0},
                {"text": "tech", "source": "c.txt", "chunk_index": 0},
            ]
        )

        # Querying with v1 should rank the security doc highest
        results = store.search(query_vector=v1, top_k=3)
        assert results[0].payload["text"] == "security"

    def test_payload_filter(self, store):
        """Filtering by payload should restrict results to matching documents."""
        v1 = np.array([1.0, 0.0, 0.0, 0.0])
        v2 = np.array([0.9, 0.1, 0.0, 0.0])  # Very similar to v1

        store.upsert(
            vectors=[v1, v2],
            payloads=[
                {"text": "security doc", "source": "security.txt", "chunk_index": 0},
                {"text": "onboarding doc", "source": "onboarding.txt", "chunk_index": 0},
            ]
        )

        # Filter to only security.txt — should not return the onboarding doc
        results = store.search(
            query_vector=v1,
            top_k=5,
            filter_by={"source": "security.txt"},
        )
        assert all(r.payload["source"] == "security.txt" for r in results)

    def test_score_threshold_filters_low_scores(self, store):
        """Vectors with low similarity should be filtered out by score_threshold."""
        v1 = np.array([1.0, 0.0, 0.0, 0.0])
        v_opposite = np.array([0.0, 0.0, 0.0, 1.0])  # orthogonal → score ~0

        store.upsert(
            vectors=[v_opposite],
            payloads=[{"text": "unrelated", "source": "x.txt", "chunk_index": 0}]
        )

        results = store.search(query_vector=v1, top_k=5, score_threshold=0.9)
        assert len(results) == 0  # Orthogonal vector has ~0 cosine similarity

    def test_upsert_custom_ids(self, store):
        """When custom IDs are provided (as valid UUIDs), they should be used."""
        vec = np.array([1.0, 0.0, 0.0, 0.0])
        # Qdrant requires IDs to be valid UUIDs or unsigned integers
        custom_id = "12345678-1234-5678-1234-567812345678"
        store.upsert(
            vectors=[vec],
            payloads=[{"text": "custom id test", "source": "x.txt", "chunk_index": 0}],
            ids=[custom_id],
        )
        results = store.search(query_vector=vec, top_k=1)
        assert str(results[0].id) == custom_id


# ── Retriever Tests ───────────────────────────────────────────────────────────

class TestRetriever:
    """Tests for Retriever: chunking, ingestion, and retrieval."""

    @pytest.fixture
    def retriever(self, store):
        return Retriever(
            store=store,
            embedder=MockEmbedder(),
            chunk_size=100,
            chunk_overlap=20,
        )

    def test_ingest_text_adds_chunks(self, retriever, store):
        """Ingesting a document should add chunks to the store."""
        text = "This is a test document about security. " * 10  # ~400 chars
        count = retriever.ingest_text(text, source="test.txt")
        assert count > 0
        assert store.count() > 0

    def test_ingest_empty_text_returns_zero(self, retriever, store):
        """Ingesting empty text should produce zero chunks."""
        count = retriever.ingest_text("", source="empty.txt")
        assert count == 0
        assert store.count() == 0

    def test_chunk_count_grows_with_document_length(self, retriever):
        """Longer documents should produce more chunks."""
        short_text = "This is about security. " * 3
        long_text = "This is about security. " * 30

        # Use a fresh store for each to compare independently
        short_store = VectorStore("short_test", vector_size=4, path=":memory:")
        long_store = VectorStore("long_test", vector_size=4, path=":memory:")

        r_short = Retriever(short_store, MockEmbedder(), chunk_size=100, chunk_overlap=20)
        r_long = Retriever(long_store, MockEmbedder(), chunk_size=100, chunk_overlap=20)

        count_short = r_short.ingest_text(short_text, "short.txt")
        count_long = r_long.ingest_text(long_text, "long.txt")

        assert count_long > count_short

    def test_retrieve_returns_chunks(self, retriever, store):
        """After ingestion, retrieve() should return relevant chunks."""
        retriever.ingest_text("This is about security protocols.", "security.txt")
        results = retriever.retrieve("security", top_k=5, score_threshold=0.0)
        assert len(results) > 0
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_retrieve_respects_top_k(self, retriever, store):
        """retrieve() should return at most top_k results."""
        # Ingest enough text to create multiple chunks
        text = "Security policy document. " * 50
        retriever.ingest_text(text, "security.txt")
        results = retriever.retrieve("security", top_k=2, score_threshold=0.0)
        assert len(results) <= 2

    def test_format_context_empty(self, retriever):
        """format_context with no chunks should return a 'not found' message."""
        context = retriever.format_context([])
        assert "No relevant documents" in context

    def test_format_context_includes_source(self, retriever, store):
        """format_context should include the source filename in the output."""
        retriever.ingest_text("Security incident reporting procedure.", "security_policy.txt")
        chunks = retriever.retrieve("security", top_k=3, score_threshold=0.0)
        context = retriever.format_context(chunks)
        assert "security_policy.txt" in context

    def test_ingest_file(self, retriever, store, tmp_path):
        """ingest_file() should read and ingest a text file."""
        doc_file = tmp_path / "test_doc.txt"
        doc_file.write_text("Security policy: all employees must use MFA. " * 10)

        count = retriever.ingest_file(doc_file)
        assert count > 0
        assert store.count() == count
