"""
scripts/ingest.py
-----------------
Standalone script to ingest documents into a persistent Qdrant collection.

Use this when you want to pre-load your knowledge base before starting the UI,
or to add new documents without restarting the app.

Usage:
    # Ingest the default knowledge base
    python scripts/ingest.py

    # Ingest from a custom directory, persist to disk
    python scripts/ingest.py --data /path/to/docs --storage ./qdrant_data

    # Use a different embedding model
    python scripts/ingest.py --model BAAI/bge-base-en-v1.5
"""

import argparse
import sys
import time
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument(
        "--data",
        default="data/knowledge_base",
        help="Directory containing .txt documents to ingest",
    )
    parser.add_argument(
        "--storage",
        default="./qdrant_data",
        help="Path for persistent Qdrant storage (use ':memory:' for ephemeral)",
    )
    parser.add_argument(
        "--collection",
        default="knowledge_base",
        help="Name of the Qdrant collection",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5",
        help="FastEmbed model for embeddings",
    )
    parser.add_argument(
        "--glob",
        default="*.txt",
        help="File pattern to match (default: *.txt)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        sys.exit(1)

    print("=" * 60)
    print("RAG LOCAL DEMO — Document Ingestion")
    print("=" * 60)
    print(f"Data directory:  {data_path.resolve()}")
    print(f"Storage:         {args.storage}")
    print(f"Collection:      {args.collection}")
    print(f"Embedding model: {args.model}")
    print()

    start_time = time.time()

    # Initialise components
    print("⏳ Loading embedding model (downloads ~25MB on first run)...")
    embedder = Embedder(model_name=args.model)
    print(f"   Vector size: {embedder.vector_size} dimensions")

    print("⏳ Connecting to Qdrant...")
    store = VectorStore(
        collection=args.collection,
        vector_size=embedder.vector_size,
        path=args.storage,
    )

    retriever = Retriever(store=store, embedder=embedder)

    # Ingest documents
    print(f"\n⏳ Ingesting documents from '{data_path}'...")
    results = retriever.ingest_directory(data_path, glob=args.glob)

    elapsed = time.time() - start_time
    total_chunks = sum(results.values())

    print(f"\n{'=' * 60}")
    print("✅ Ingestion complete!")
    print(f"   Files processed: {len(results)}")
    print(f"   Total chunks:    {total_chunks}")
    print(f"   Total points:    {store.count()}")
    print(f"   Time elapsed:    {elapsed:.1f}s")
    print(f"   Storage path:    {args.storage}")
    print("=" * 60)
    print("\n💡 To start the UI using this persistent store:")
    print(f"   python src/app.py --storage {args.storage}\n")


if __name__ == "__main__":
    main()
