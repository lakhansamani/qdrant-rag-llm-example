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
from unittest.mock import MagicMock

from src.authz import AuthorizationError
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
    p.authz = None           # No permission enforcement by default (legacy mode)

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


# ── FGA (permission-aware) Pipeline Tests ─────────────────────────────────────

class MockAuthz:
    """Scriptable stand-in for AuthzClient."""

    def __init__(self, allowed: list[str], verify: bool = True):
        self._allowed = allowed
        self._verify = verify
        self.allowed_calls: list[str] = []
        self.verify_calls: list[list[str]] = []

    def allowed_documents(self, user_token: str) -> list[str]:
        self.allowed_calls.append(user_token)
        return list(self._allowed)

    def verify_sources(self, user_token: str, sources: list[str]) -> bool:
        self.verify_calls.append(list(sources))
        return self._verify


class TestFGAPipeline:
    """ask() with an AuthzClient configured: enforcement, fail-closed, refusal."""

    @pytest.fixture
    def fga_pipeline(self, pipeline):
        """The standard mocked pipeline, with two docs and authz enabled."""
        pipeline.ingest_text("Security: MFA is mandatory.", "security_policy.txt")
        pipeline.ingest_text("Onboarding: meet your buddy.", "onboarding_guide.txt")
        return pipeline

    def test_token_required_when_authz_configured(self, fga_pipeline):
        """No silent bypass: authz configured + no token = error, not open query."""
        fga_pipeline.authz = MockAuthz(allowed=["onboarding_guide.txt"])
        with pytest.raises(AuthorizationError, match="requires the user's access token"):
            fga_pipeline.ask("What is the security policy?")

    def test_sources_restricted_to_allowed_documents(self, fga_pipeline):
        """A user who can only view onboarding never gets security chunks."""
        authz = MockAuthz(allowed=["onboarding_guide.txt"])
        fga_pipeline.authz = authz
        response = fga_pipeline.ask("Tell me everything", user_token="bob-jwt")

        assert authz.allowed_calls == ["bob-jwt"]
        assert response.sources, "expected at least one permitted chunk"
        assert all(s.source == "onboarding_guide.txt" for s in response.sources)

    def test_no_grants_refuses_without_calling_llm(self, fga_pipeline):
        """Empty allow-list → refusal answer; the LLM must never be invoked."""
        fga_pipeline.authz = MockAuthz(allowed=[])
        fga_pipeline.llm.generate = MagicMock(side_effect=AssertionError("LLM called"))

        response = fga_pipeline.ask("What is the security policy?", user_token="jwt")
        assert response.sources == []
        assert "don't have access" in response.answer

    def test_revocation_mid_request_blocks_answer(self, fga_pipeline):
        """If the post-generation re-check fails, the answer must not be returned."""
        fga_pipeline.authz = MockAuthz(allowed=["onboarding_guide.txt"], verify=False)
        with pytest.raises(AuthorizationError, match="revoked"):
            fga_pipeline.ask("Tell me everything", user_token="jwt")

    def test_cited_sources_are_reverified(self, fga_pipeline):
        """The defense-in-depth check runs against the actually cited sources."""
        authz = MockAuthz(allowed=["onboarding_guide.txt", "security_policy.txt"])
        fga_pipeline.authz = authz
        response = fga_pipeline.ask("Tell me everything", user_token="jwt")

        assert len(authz.verify_calls) == 1
        assert set(authz.verify_calls[0]) == {s.source for s in response.sources}

    def test_no_authz_preserves_legacy_behaviour(self, fga_pipeline):
        """Without an AuthzClient, ask() works exactly as before (no token needed)."""
        assert fga_pipeline.authz is None
        response = fga_pipeline.ask("What is the security policy?")
        assert isinstance(response, RAGResponse)
