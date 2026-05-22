"""
embedder.py
-----------
Wraps FastEmbed's TextEmbedding to produce dense vector embeddings locally.

FastEmbed uses quantized ONNX models — no GPU required, no API key needed.
Default model: BAAI/bge-small-en-v1.5  (384-dim, ~25MB, fast & accurate)

Usage:
    embedder = Embedder()
    vectors = embedder.embed(["Hello world", "Another document"])
"""

from typing import Iterator
import numpy as np
from fastembed import TextEmbedding


# The embedding model to use. bge-small-en-v1.5 gives a good accuracy/speed balance.
# It produces 384-dimensional vectors and outperforms OpenAI Ada-002 on many benchmarks.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    """
    Local embedding engine powered by FastEmbed.

    On first use, the model (~25 MB) is downloaded and cached in ~/.cache/fastembed.
    All subsequent calls are fully offline.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """
        Initialise the embedding model.

        Args:
            model_name: HuggingFace model identifier supported by FastEmbed.
                        See https://qdrant.github.io/fastembed/examples/Supported_Models/
        """
        self.model_name = model_name
        # TextEmbedding downloads the ONNX model on first call.
        # Subsequent instantiations reuse the cached model from disk.
        self._model = TextEmbedding(model_name=model_name)

    @property
    def vector_size(self) -> int:
        """
        Return the dimensionality of the embeddings produced by this model.

        BAAI/bge-small-en-v1.5 → 384 dimensions.
        You must create your Qdrant collection with this exact size.
        """
        # Embed a dummy sentence to determine the output dimension
        dummy = list(self._model.embed(["dim check"]))
        return dummy[0].shape[0]

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """
        Embed a list of text strings into dense vectors.

        Args:
            texts: List of strings to embed. Can be documents or queries —
                   bge models work well for both without needing prefix tokens.

        Returns:
            List of numpy arrays, one per input text, shape (vector_size,).

        Example:
            embedder = Embedder()
            vecs = embedder.embed(["What is our security policy?"])
            print(vecs[0].shape)  # (384,)
        """
        if not texts:
            return []

        # .embed() returns a generator — materialise it into a list
        embeddings: list[np.ndarray] = list(self._model.embed(texts))
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Convenience wrapper to embed a single query string.

        Args:
            query: The user's search query.

        Returns:
            A single numpy array of shape (vector_size,).
        """
        return self.embed([query])[0]

    def embed_iter(self, texts: list[str], batch_size: int = 64) -> Iterator[np.ndarray]:
        """
        Embed texts lazily in batches — useful for large document sets.

        Args:
            texts:      All texts to embed.
            batch_size: Number of texts to process at once (controls RAM usage).

        Yields:
            numpy arrays one at a time.
        """
        yield from self._model.embed(texts, batch_size=batch_size)
