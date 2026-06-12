"""
vector_store.py
---------------
Wraps Qdrant's Python client to store, index, and search document embeddings.

Qdrant is a high-performance vector database built in Rust. It supports:
  - Cosine, Euclidean, and dot-product similarity
  - Payload filters (metadata-based filtering on top of vector search)
  - In-memory mode (no Docker needed for local testing)
  - Persistent local storage (file-based, no server required)
  - Server mode (via Docker, for production or large datasets)

This module uses LOCAL (file-based) storage by default so the demo runs
with zero infrastructure — just Python.
"""

import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    ScoredPoint,
    VectorParams,
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
)


class VectorStore:
    """
    A simple vector store backed by Qdrant.

    Supports three modes:
      - ":memory:"            → in-process, data lost on exit (great for tests)
      - "http(s)://host:port" → connect to a running Qdrant server (Docker, cloud)
      - path (str)            → local file storage, data persists between runs

    Example:
        store = VectorStore(collection="kb", vector_size=384)
        store.upsert(vectors=[v1, v2], payloads=[{"text": "..."}, ...])
        results = store.search(query_vector=qv, top_k=5)
    """

    def __init__(
        self,
        collection: str,
        vector_size: int,
        path: str = ":memory:",
        distance: Distance = Distance.COSINE,
    ) -> None:
        """
        Initialise the Qdrant client and ensure the collection exists.

        Args:
            collection:  Name of the Qdrant collection (like a DB table).
            vector_size: Dimensionality of the embedding vectors (must match your model).
            path:        ":memory:" for in-process storage, a URL like
                         "http://localhost:6333" to connect to a running Qdrant
                         server (e.g. the official Docker image, which also serves
                         the dashboard UI), or a file-system path for embedded
                         persistent storage (e.g. "./qdrant_data").
            distance:    Similarity metric. COSINE is recommended for text embeddings.
        """
        self.collection = collection
        self.vector_size = vector_size

        if path == ":memory:":
            self._client = QdrantClient(location=":memory:")
        elif path.startswith(("http://", "https://")):
            self._client = QdrantClient(url=path)
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=path)

        # Create the collection if it does not already exist.
        # Calling recreate_collection would wipe existing data — we avoid that.
        self._ensure_collection(distance)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _ensure_collection(self, distance: Distance) -> None:
        """Create the Qdrant collection if it doesn't exist yet."""
        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection not in existing:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.vector_size, distance=distance),
            )

    # ── Public API ──────────────────────────────────────────────────────────

    def upsert(
        self,
        vectors: list,            # list[np.ndarray]
        payloads: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> None:
        """
        Insert or update points in the collection.

        Args:
            vectors:  List of embedding vectors (one per document chunk).
            payloads: List of dicts with metadata — stored alongside the vector.
                      Typical keys: "text", "source", "chunk_index".
            ids:      Optional list of string IDs. Auto-generated UUIDs if omitted.

        The payload is critical: it lets you retrieve the original document text
        and metadata alongside the search score.
        """
        if not vectors:
            return

        # Auto-generate UUIDs if the caller didn't provide IDs.
        # Qdrant in-memory mode requires IDs to be valid UUID strings or integers.
        point_ids = ids or [str(uuid.uuid4()) for _ in vectors]

        points = [
            PointStruct(
                id=pid,
                vector=vec.tolist(),   # Qdrant expects a plain Python list
                payload=payload,
            )
            for pid, vec, payload in zip(point_ids, vectors, payloads)
        ]

        self._client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        query_vector,              # np.ndarray
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_by: dict[str, Any] | None = None,
        allowed_sources: list[str] | None = None,
    ) -> list[ScoredPoint]:
        """
        Find the most similar vectors to the query.

        Args:
            query_vector:    The embedded query as a numpy array.
            top_k:           Maximum number of results to return.
            score_threshold: Minimum cosine similarity score (0–1). Filter out low matches.
            filter_by:       Optional dict of payload key→value to filter results.
                             E.g. {"source": "security_policy.txt"} to search only that file.
            allowed_sources: Permission allow-list for the "source" payload field.
                             None  → no restriction (single-user / legacy mode).
                             []    → caller may see nothing; returns [] without searching.
                             [...] → only chunks from these sources are candidates —
                                     the filter applies DURING the vector search, so
                                     forbidden chunks are never scored and top_k stays
                                     meaningful.

        Returns:
            List of ScoredPoint objects with .score, .payload, and .id.
        """
        # An empty allow-list is a decision, not an absence of one: fail closed.
        if allowed_sources is not None and not allowed_sources:
            return []

        # Build a Qdrant payload filter if requested
        conditions: list[Any] = []
        if allowed_sources is not None:
            conditions.append(
                FieldCondition(key="source", match=MatchAny(any=allowed_sources))
            )
        if filter_by:
            conditions += [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_by.items()
            ]
        qdrant_filter = Filter(must=conditions) if conditions else None

        # query_points replaces the deprecated .search() in qdrant-client >= 1.7
        result = self._client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=top_k,
            score_threshold=score_threshold if score_threshold > 0 else None,
            query_filter=qdrant_filter,
            with_payload=True,   # Always return the payload so we can read the text
        )
        return result.points

    def count(self) -> int:
        """Return the number of vectors stored in the collection."""
        info = self._client.get_collection(self.collection)
        return info.points_count or 0

    def delete_collection(self) -> None:
        """Drop the entire collection. Useful in tests for teardown."""
        self._client.delete_collection(self.collection)
