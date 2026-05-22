"""
retriever.py
------------
Handles document ingestion (chunking + embedding + indexing) and retrieval.

The key idea of RAG:
  1. INGEST: Split documents into chunks → embed → store in Qdrant
  2. RETRIEVE: Embed the user query → find the most similar chunks

Why chunk instead of embedding whole documents?
  - Embedding models have a token limit (~512 tokens for bge-small-en).
  - Smaller chunks give more precise similarity matches.
  - Chunks overlap slightly so context isn't lost at boundaries.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

from src.embedder import Embedder
from src.vector_store import VectorStore


@dataclass
class DocumentChunk:
    """
    Represents a single chunk of text extracted from a source document.

    Attributes:
        text:        The raw text content of this chunk.
        source:      Filename or identifier of the original document.
        chunk_index: Position of this chunk within the document (0-based).
    """
    text: str
    source: str
    chunk_index: int


@dataclass
class RetrievedChunk:
    """
    A document chunk returned by a similarity search, including its score.

    Attributes:
        text:        The chunk's text content.
        source:      Which document this came from.
        chunk_index: Position within that document.
        score:       Cosine similarity score (0–1). Higher is more relevant.
    """
    text: str
    source: str
    chunk_index: int
    score: float


class Retriever:
    """
    Manages document ingestion and semantic retrieval.

    Typical flow:
        retriever = Retriever(store=my_store, embedder=my_embedder)
        retriever.ingest_file(Path("docs/policy.txt"))
        chunks = retriever.retrieve("How do I report a security incident?")
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
    ) -> None:
        """
        Args:
            store:         The VectorStore to write to and read from.
            embedder:      The Embedder used to convert text to vectors.
            chunk_size:    Maximum number of characters per chunk.
                           Keep below the model's token limit (~1500 chars for bge-small-en).
            chunk_overlap: Number of characters to overlap between consecutive chunks.
                           Prevents losing context at chunk boundaries.
        """
        self.store = store
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Chunking ────────────────────────────────────────────────────────────

    def _split_text(self, text: str, source: str) -> list[DocumentChunk]:
        """
        Split a long document into overlapping character-level chunks.

        Simple sliding-window approach:
          - Start at position 0
          - Take chunk_size characters
          - Advance by (chunk_size - chunk_overlap) characters
          - Repeat until end of document

        Args:
            text:   Full document text.
            source: Document identifier (e.g. filename).

        Returns:
            List of DocumentChunk objects.
        """
        chunks: list[DocumentChunk] = []
        step = self.chunk_size - self.chunk_overlap
        idx = 0

        while idx < len(text):
            chunk_text = text[idx: idx + self.chunk_size].strip()
            if chunk_text:  # Skip empty chunks
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    source=source,
                    chunk_index=len(chunks),
                ))
            idx += step

        return chunks

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest_text(self, text: str, source: str) -> int:
        """
        Chunk, embed, and index a raw string.

        Args:
            text:   The document content.
            source: A label for this document (e.g. "security_policy.txt").

        Returns:
            Number of chunks ingested.
        """
        chunks = self._split_text(text, source)
        if not chunks:
            return 0

        # Embed all chunk texts in a single batched call (efficient)
        texts = [c.text for c in chunks]
        vectors = self.embedder.embed(texts)

        # Build payloads — the data stored alongside each vector in Qdrant
        payloads = [
            {
                "text": c.text,
                "source": c.source,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        self.store.upsert(vectors=vectors, payloads=payloads)
        return len(chunks)

    def ingest_file(self, path: Path) -> int:
        """
        Read a plain-text file and ingest its contents.

        Args:
            path: Path to a .txt file (UTF-8 encoded).

        Returns:
            Number of chunks ingested.
        """
        text = path.read_text(encoding="utf-8")
        return self.ingest_text(text, source=path.name)

    def ingest_directory(self, directory: Path, glob: str = "*.txt") -> dict[str, int]:
        """
        Ingest all matching files from a directory.

        Args:
            directory: Path to folder containing documents.
            glob:      File pattern to match (default: all .txt files).

        Returns:
            Dict mapping filename → number of chunks ingested.
        """
        results: dict[str, int] = {}
        for file_path in sorted(directory.glob(glob)):
            count = self.ingest_file(file_path)
            results[file_path.name] = count
            print(f"  ✓ Ingested {file_path.name}: {count} chunks")
        return results

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: float = 0.3,
        source_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Find the most relevant document chunks for a given query.

        Steps:
          1. Embed the query using the same model used during ingestion.
          2. Run cosine similarity search in Qdrant.
          3. Return the top-k chunks above the score threshold.

        Args:
            query:           The user's question in natural language.
            top_k:           How many chunks to return.
            score_threshold: Minimum similarity score (0–1). Raise this to reduce noise.
            source_filter:   Optional: restrict search to a specific document filename.

        Returns:
            List of RetrievedChunk, sorted by similarity score (best first).
        """
        # Embed the query — same model, same vector space as the documents
        query_vector = self.embedder.embed_query(query)

        # Build optional filter (e.g. search only within security_policy.txt)
        filter_by = {"source": source_filter} if source_filter else None

        # Execute the vector similarity search in Qdrant
        raw_results = self.store.search(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_by=filter_by,
        )

        # Convert Qdrant ScoredPoint objects to our cleaner RetrievedChunk dataclass
        return [
            RetrievedChunk(
                text=r.payload["text"],
                source=r.payload["source"],
                chunk_index=r.payload["chunk_index"],
                score=r.score,
            )
            for r in raw_results
        ]

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into a single context string for the LLM prompt.

        Each chunk is labelled with its source file so the LLM can cite it.

        Args:
            chunks: Retrieved chunks from self.retrieve().

        Returns:
            A formatted string ready to inject into the LLM prompt.
        """
        if not chunks:
            return "No relevant documents found."

        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(
                f"[Source {i}: {chunk.source} | relevance: {chunk.score:.2f}]\n"
                f"{chunk.text}"
            )

        return "\n\n---\n\n".join(parts)
