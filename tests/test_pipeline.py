"""
tests/test_pipeline.py
-----------------------
Integration tests for the full RAG pipeline.

These tests use mock embeddings and a mock LLM to avoid external dependencies.
They verify the full flow: ingest → retrieve → generate → response.

Run with:
    pytest tests/test_pipeline.py -v
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.pipeline import RAGPipeline, RAGResponse
from src.retriever import RetrievedChunk


# ── Mock components ───────────────────────────────────────────────────────────

class MockEmbedder:
    """Fast deterministic embedder for pipeline tests."""
    vector_size = 8

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        # Hash the text to produce a deterministic vector
        vecs = []
        for text in texts:
            seed = abs(hash(text)) % (2**31)
            rng = np.random.RandomState(seed)
            v = rng.rand(8).astype(np.float32)
            # Normalise to unit vector (required for cosine similarity)
            v /= np.linalg.norm(v)
            vecs.append(v)
        return vecs

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

    def embed_iter(self, texts, batch_size=64):
        yield from self.embed(texts)


class MockLLM:
    """Mock LLM that returns a canned response without calling Ollama."""
    model = "mock-llama"

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, system: str = "", stream: bool = False) -> str:
        # Return a response that mentions the context was used
        if "No relevant documents" in prompt:
            return "I don't have enough information in the knowledge base to answer this."
        return "Based on the provided documents, the answer is: this is a test response."

    def build_rag_prompt(self, context: str, question: str) -> str:
        return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def pipeline():
    """
    Create a RAGPipeline with mocked embedder and LLM.

    We use patch.object to replace the real Embedder and OllamaClient with
    our fast mock implementations, then rebuild the pipeline's internal state.
    """
    p = RAGPipeline.__new__(RAGPipeline)
    p.top_k = 4
    p.score_threshold = 0.0  # Accept all results in tests

    mock_embedder = MockEmbedder()

    from src.vector_store import VectorStore
    from src.retriever import Retriever

    p.embedder = mock_embedder
    p.store = VectorStore(collection="test_kb", vector_size=8, path=":memory:")
    p.retriever = Retriever(
        store=p.store,
        embedder=mock_embedder,
        chunk_size=200,
        chunk_overlap=40,
    )
    p.llm = MockLLM()
    return p


# ── Pipeline Tests ────────────────────────────────────────────────────────────

class TestRAGPipeline:
    """Integration tests for the full RAG pipeline."""

    def test_ingest_text_increases_document_count(self, pipeline):
        """After ingesting a document, document_count should be > 0."""
        pipeline.ingest_text(
            "Security policy: all employees must use multi-factor authentication.",
            "security.txt",
        )
        assert pipeline.document_count > 0

    def test_ask_returns_rag_response(self, pipeline):
        """ask() should return a RAGResponse object."""
        pipeline.ingest_text("Employees get 25 days annual leave.", "hr.txt")
        response = pipeline.ask("How many days of annual leave do we get?")
        assert isinstance(response, RAGResponse)

    def test_response_has_all_fields(self, pipeline):
        """RAGResponse should have all expected fields populated."""
        pipeline.ingest_text("MFA is required for all systems.", "policy.txt")
        response = pipeline.ask("Is MFA required?")

        assert response.question == "Is MFA required?"
        assert isinstance(response.answer, str)
        assert len(response.answer) > 0
        assert isinstance(response.sources, list)
        assert isinstance(response.context_used, str)
        assert response.model_used == "mock-llama"

    def test_sources_are_retrieved_chunks(self, pipeline):
        """Sources in the response should be RetrievedChunk objects."""
        pipeline.ingest_text("The on-call rotation starts after 3 months.", "eng.txt")
        response = pipeline.ask("When does on-call start?")
        for source in response.sources:
            assert isinstance(source, RetrievedChunk)
            assert isinstance(source.text, str)
            assert isinstance(source.source, str)
            assert isinstance(source.score, float)

    def test_multiple_documents_ingested(self, pipeline):
        """After ingesting multiple documents, all should be searchable."""
        pipeline.ingest_text("Security: Use strong passwords.", "security.txt")
        pipeline.ingest_text("Onboarding: Meet your buddy on day one.", "onboarding.txt")
        pipeline.ingest_text("Tech: We use FastAPI and PostgreSQL.", "tech.txt")

        # Should have chunks from all three docs
        assert pipeline.document_count > 2

    def test_ingest_directory(self, pipeline, tmp_path):
        """ingest_directory() should process all .txt files in a folder."""
        # Create two sample files
        (tmp_path / "doc1.txt").write_text("Document one content about security. " * 5)
        (tmp_path / "doc2.txt").write_text("Document two content about onboarding. " * 5)

        results = pipeline.ingest_directory(tmp_path)
        assert "doc1.txt" in results
        assert "doc2.txt" in results
        assert all(v > 0 for v in results.values())

    def test_ask_with_source_filter(self, pipeline):
        """source_filter should restrict retrieval to the specified document."""
        pipeline.ingest_text("Security: MFA is mandatory.", "security.txt")
        pipeline.ingest_text("HR: 25 days annual leave.", "hr.txt")

        # Ask a security question but filter to HR — should still return something
        # (the mock LLM doesn't validate sources, but the filter should be applied)
        response = pipeline.ask("What is the leave policy?", source_filter="hr.txt")
        # All sources should be from hr.txt if any are returned
        for source in response.sources:
            assert source.source == "hr.txt"

    def test_pretty_print_does_not_raise(self, pipeline, capsys):
        """RAGResponse.pretty_print() should output to stdout without errors."""
        pipeline.ingest_text("Passwords must be 12 characters long.", "policy.txt")
        response = pipeline.ask("What is the password policy?")
        response.pretty_print()  # Should not raise any exception

        captured = capsys.readouterr()
        assert "QUESTION" in captured.out
        assert "ANSWER" in captured.out

    def test_ollama_not_running_raises_error(self, pipeline):
        """If Ollama is not available, ask() should raise a RuntimeError."""
        # Replace the mock LLM with one that reports unavailable
        pipeline.llm.is_available = lambda: False
        pipeline.ingest_text("test", "test.txt")

        with pytest.raises(RuntimeError, match="Ollama is not running"):
            pipeline.ask("test question")
