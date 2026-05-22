"""
tests/test_embedder.py
----------------------
Unit tests for the Embedder class.

These tests use the real FastEmbed model — they require network access on the
first run to download the model weights (~25MB). Subsequent runs are offline.

Run with:
    pytest tests/test_embedder.py -v
"""

import numpy as np
import pytest
from src.embedder import Embedder, DEFAULT_MODEL


@pytest.fixture(scope="module")
def embedder():
    """
    Create a shared Embedder instance for all tests in this module.

    Using scope="module" ensures the model is only loaded once per test session,
    which saves significant time since model loading takes a few seconds.
    """
    return Embedder()


class TestEmbedder:
    """Tests for the Embedder class."""

    def test_embed_returns_correct_number_of_vectors(self, embedder):
        """embed() should return exactly one vector per input text."""
        texts = ["Hello world", "Another sentence", "Third text"]
        vectors = embedder.embed(texts)
        assert len(vectors) == 3

    def test_embed_returns_numpy_arrays(self, embedder):
        """Each embedding should be a numpy ndarray."""
        vectors = embedder.embed(["test sentence"])
        assert isinstance(vectors[0], np.ndarray)

    def test_embed_correct_dimension(self, embedder):
        """
        Embeddings should have the expected dimensionality.
        BAAI/bge-small-en-v1.5 produces 384-dimensional vectors.
        """
        vectors = embedder.embed(["dimension check"])
        assert vectors[0].shape == (384,), (
            f"Expected 384-dim embeddings, got {vectors[0].shape}"
        )

    def test_vector_size_property(self, embedder):
        """vector_size property should match actual embedding dimension."""
        assert embedder.vector_size == 384

    def test_embed_query_single_vector(self, embedder):
        """embed_query() should return a single numpy array, not a list."""
        vector = embedder.embed_query("What is the security policy?")
        assert isinstance(vector, np.ndarray)
        assert vector.ndim == 1
        assert vector.shape[0] == 384

    def test_embed_empty_list_returns_empty(self, embedder):
        """embed([]) should return an empty list without errors."""
        result = embedder.embed([])
        assert result == []

    def test_similar_texts_have_higher_similarity(self, embedder):
        """
        Semantically similar sentences should have higher cosine similarity
        than semantically different sentences.

        This is the core property that makes RAG work: the vector space captures
        semantic meaning, not just word overlap.
        """
        # These are semantically similar
        v1 = embedder.embed_query("How do I report a security incident?")
        v2 = embedder.embed_query("What is the procedure for security breach reporting?")

        # This is semantically different
        v3 = embedder.embed_query("What is the annual leave policy?")

        # Compute cosine similarities
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_same_topic = cosine_sim(v1, v2)
        sim_diff_topic = cosine_sim(v1, v3)

        assert sim_same_topic > sim_diff_topic, (
            f"Expected similar texts to have higher similarity "
            f"({sim_same_topic:.3f}) than different texts ({sim_diff_topic:.3f})"
        )

    def test_embedding_is_deterministic(self, embedder):
        """The same text should always produce the same embedding."""
        text = "deterministic embedding test"
        v1 = embedder.embed_query(text)
        v2 = embedder.embed_query(text)
        np.testing.assert_array_equal(v1, v2)

    def test_different_texts_produce_different_vectors(self, embedder):
        """Two different texts should produce meaningfully different embeddings."""
        v1 = embedder.embed_query("apple")
        v2 = embedder.embed_query("quantum mechanics")
        # They should not be identical
        assert not np.array_equal(v1, v2)

    def test_embed_iter_yields_all_vectors(self, embedder):
        """embed_iter() should yield one vector per input text."""
        texts = [f"document {i}" for i in range(10)]
        results = list(embedder.embed_iter(texts, batch_size=4))
        assert len(results) == 10

    def test_default_model_name(self, embedder):
        """Default model should be bge-small-en-v1.5."""
        assert embedder.model_name == DEFAULT_MODEL
