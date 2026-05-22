"""
pipeline.py
-----------
The RAG Pipeline — ties together Embedder, VectorStore, Retriever, and LLM.

This is the single entry point for the application:
  1. Call pipeline.ingest_directory(path) to load your knowledge base.
  2. Call pipeline.ask(question) to get an LLM answer grounded in your docs.

The pipeline implements the full RAG loop:
  Query → Embed query → Search Qdrant → Format context → Prompt LLM → Return answer
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever, RetrievedChunk
from src.llm_client import OllamaClient


@dataclass
class RAGResponse:
    """
    The result of a RAG query, including the answer and full audit trail.

    Attributes:
        question:         The original user question.
        answer:           The LLM-generated answer, grounded in the retrieved docs.
        sources:          List of document chunks the LLM used as context.
        context_used:     The raw formatted context string sent to the LLM.
        model_used:       The Ollama model that generated the answer.
    """
    question: str
    answer: str
    sources: list[RetrievedChunk]
    context_used: str
    model_used: str

    def pretty_print(self) -> None:
        """Print a nicely formatted summary of the RAG response."""
        print(f"\n{'='*60}")
        print(f"QUESTION: {self.question}")
        print(f"{'='*60}")
        print(f"\nANSWER:\n{self.answer}")
        print(f"\n{'─'*60}")
        print("SOURCES USED:")
        for i, chunk in enumerate(self.sources, 1):
            print(f"  {i}. {chunk.source} (score: {chunk.score:.3f})")
        print(f"{'='*60}\n")


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    The pipeline uses:
      - FastEmbed (local ONNX model) for embeddings — no API key
      - Qdrant (in-memory or local file) for vector search
      - Ollama (local server) for LLM inference — no API key

    All data stays on your machine. No external API calls unless you choose to.

    Example:
        pipeline = RAGPipeline(llm_model="llama3.2")
        pipeline.ingest_directory(Path("data/knowledge_base"))
        response = pipeline.ask("What is our policy on data residency?")
        response.pretty_print()
    """

    def __init__(
        self,
        collection: str = "knowledge_base",
        storage_path: str = ":memory:",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        llm_model: str = "llama3.2",
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        top_k: int = 4,
        score_threshold: float = 0.3,
    ) -> None:
        """
        Args:
            collection:      Qdrant collection name.
            storage_path:    ":memory:" for volatile storage, or a path for persistence.
                             Example: "./qdrant_data" stores embeddings between runs.
            embedding_model: FastEmbed model for embeddings (384-dim by default).
            llm_model:       Ollama model name (must be pulled first: `ollama pull <model>`).
            chunk_size:      Characters per document chunk.
            chunk_overlap:   Overlap between consecutive chunks (preserves context).
            top_k:           Number of chunks to retrieve per query.
            score_threshold: Minimum cosine similarity for a chunk to be included.
        """
        self.top_k = top_k
        self.score_threshold = score_threshold

        # Step 1: Initialise the embedding model (downloads on first run)
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = Embedder(model_name=embedding_model)

        # Step 2: Initialise the vector store (creates collection if needed)
        print(f"Initialising Qdrant collection: '{collection}' at '{storage_path}'")
        self.store = VectorStore(
            collection=collection,
            vector_size=self.embedder.vector_size,
            path=storage_path,
        )

        # Step 3: Set up the retriever (handles chunking + indexing + search)
        self.retriever = Retriever(
            store=self.store,
            embedder=self.embedder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Step 4: Set up the LLM client (connects to local Ollama server)
        print(f"Using LLM: {llm_model} (via Ollama)")
        self.llm = OllamaClient(model=llm_model)

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest_directory(self, directory: Path, glob: str = "*.txt") -> dict[str, int]:
        """
        Load all documents from a directory into the knowledge base.

        Args:
            directory: Folder containing .txt documents.
            glob:      File pattern to match.

        Returns:
            Dict of {filename: chunk_count} for logging.
        """
        print(f"\nIngesting documents from: {directory}")
        results = self.retriever.ingest_directory(directory, glob=glob)
        total = sum(results.values())
        print(f"✅ Ingestion complete: {len(results)} files, {total} total chunks\n")
        return results

    def ingest_text(self, text: str, source: str) -> int:
        """
        Ingest a raw string directly (useful for programmatic ingestion).

        Args:
            text:   Document content.
            source: Label for this document.

        Returns:
            Number of chunks ingested.
        """
        return self.retriever.ingest_text(text, source)

    # ── Querying ─────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        source_filter: str | None = None,
        verbose: bool = False,
    ) -> RAGResponse:
        """
        Ask a question and get an LLM answer grounded in the knowledge base.

        Full RAG loop:
          1. Embed the question
          2. Retrieve top-k relevant chunks from Qdrant
          3. Format chunks as context
          4. Build the LLM prompt (context + question)
          5. Call Ollama to generate the answer
          6. Return a structured RAGResponse

        Args:
            question:      The user's question in natural language.
            source_filter: Optionally restrict search to one document (e.g. "security_policy.txt").
            verbose:       If True, print retrieved chunks before calling the LLM.

        Returns:
            RAGResponse with the answer, sources, and metadata.
        """
        if not self.llm.is_available():
            raise RuntimeError(
                "Ollama is not running. Start it with: ollama serve\n"
                f"Then pull your model: ollama pull {self.llm.model}"
            )

        # ── Step 1: Retrieve relevant chunks from Qdrant ──────────────────
        chunks = self.retriever.retrieve(
            query=question,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            source_filter=source_filter,
        )

        if verbose:
            print(f"Retrieved {len(chunks)} chunks:")
            for c in chunks:
                print(f"  [{c.score:.3f}] {c.source}: {c.text[:80]}...")

        # ── Step 2: Format context for the LLM ───────────────────────────
        context = self.retriever.format_context(chunks)

        # ── Step 3: Build the RAG prompt ──────────────────────────────────
        prompt = self.llm.build_rag_prompt(context=context, question=question)

        # ── Step 4: Generate the LLM answer ───────────────────────────────
        answer = self.llm.generate(prompt=prompt)

        return RAGResponse(
            question=question,
            answer=answer,
            sources=chunks,
            context_used=context,
            model_used=self.llm.model,
        )

    @property
    def document_count(self) -> int:
        """Number of chunks currently indexed in the vector store."""
        return self.store.count()
