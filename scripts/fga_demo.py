"""
scripts/fga_demo.py
-------------------
End-to-end walk-through of permission-aware RAG: two users ask the same
questions and get different answers, because Qdrant only searches the
documents each user is allowed to see (enforced by Authorizer's embedded
OpenFGA engine — see src/authz.py).

It also demonstrates live revocation: Bob is granted engineering access
mid-run (one tuple write), immediately sees the tech-stack doc, and loses
it again the moment the tuple is deleted. No re-ingestion, no re-login.

Prerequisites:
    docker compose up -d            # Qdrant + Authorizer
    python scripts/fga_seed.py      # users, model, grants

Usage:
    python scripts/fga_demo.py            # retrieval-only (no Ollama needed)
    python scripts/fga_demo.py --llm      # also generate answers via Ollama
"""

import argparse
import sys
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from authorizer import (
    AuthorizerAdminClient,
    FgaTupleInput,
    FgaWriteTuplesRequest,
    PaginatedRequest,
    PaginationRequest,
)

from scripts.fga_seed import DEMO_PASSWORD
from src.authz import AuthzClient
from src.pipeline import RAGPipeline

QUESTIONS = [
    "What tech stack do we use for the frontend?",
    "What was our Q4 revenue and cash runway?",
]


def show_user_turn(
    pipeline: RAGPipeline,
    authz: AuthzClient,
    email: str,
    token: str,
    use_llm: bool,
) -> None:
    """Ask every demo question as one user and print what they can see."""
    allowed = authz.allowed_documents(token)
    print(f"\n👤 {email}")
    print(f"   may view: {', '.join(sorted(allowed)) if allowed else '(nothing)'}")

    for question in QUESTIONS:
        print(f"\n   ❓ {question}")
        if use_llm:
            response = pipeline.ask(question, user_token=token)
            sources = sorted({s.source for s in response.sources})
            print(f"   📎 sources: {', '.join(sources) if sources else '(none)'}")
            print(f"   💬 {response.answer.strip()[:300]}")
        else:
            chunks = pipeline.retriever.retrieve(
                query=question,
                top_k=pipeline.top_k,
                score_threshold=pipeline.score_threshold,
                allowed_sources=allowed,
            )
            sources = sorted({c.source for c in chunks})
            print(f"   📎 retrievable sources: {', '.join(sources) if sources else '(none — permission-filtered)'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Permission-aware RAG demo")
    parser.add_argument("--authorizer", default="http://localhost:8080",
                        help="Authorizer server URL")
    parser.add_argument("--admin-secret", default="admin",
                        help="Admin secret (used only for the revocation demo)")
    parser.add_argument("--client-id", default="123456",
                        help="Authorizer client id")
    parser.add_argument("--storage", default="http://localhost:6333",
                        help="Qdrant storage URL or file path (must already be ingested via fga_seed.py)")
    parser.add_argument("--model", default="llama3.2", help="Ollama model (with --llm)")
    parser.add_argument("--llm", action="store_true",
                        help="Generate answers via Ollama (default: retrieval-only)")
    args = parser.parse_args()

    print("=" * 60)
    print("RAG LOCAL DEMO — Fine-Grained Permissions (FGA)")
    print("Authorizer (embedded OpenFGA) + Qdrant" + (" + Ollama" if args.llm else ""))
    print("=" * 60)

    authz = AuthzClient(args.authorizer, args.client_id)
    admin = AuthorizerAdminClient(args.authorizer, args.admin_secret, protocol="rest")

    pipeline = RAGPipeline(
        storage_path=args.storage,
        llm_model=args.model,
        authz=authz,
    )

    print("⏳ Logging in demo users (created by scripts/fga_seed.py)...")
    tokens = {
        email: authz.login(email, DEMO_PASSWORD)
        for email in ("bob@example.com", "alice@example.com", "carol@example.com")
    }

    # ── Act 1: same questions, different answers ──────────────────────────
    # Note the finance question: only carol (finance) gets an answer; alice
    # (engineering) and bob are blocked — the financial report is never
    # retrieved for them, so the LLM cannot leak it.
    print("\n" + "─" * 60)
    print("ACT 1 — Same questions, permission-filtered retrieval")
    print("─" * 60)
    for email, token in tokens.items():
        show_user_turn(pipeline, authz, email, token, args.llm)

    # ── Act 2: grant, observe, revoke, observe ────────────────────────────
    print("\n" + "─" * 60)
    print("ACT 2 — Live grant & revocation (no re-ingestion, no re-login)")
    print("─" * 60)

    users_res = admin.users(PaginatedRequest(pagination=PaginationRequest(limit=50)))
    bob_id = next(u.id for u in users_res.users if u.email == "bob@example.com")
    grant = FgaTupleInput(user=f"user:{bob_id}", relation="member", object="team:engineering")

    print("\n⚡ Granting: bob → member → team:engineering (one tuple write)")
    admin.fga_write_tuples(FgaWriteTuplesRequest(tuples=[grant]))
    show_user_turn(pipeline, authz, "bob@example.com", tokens["bob@example.com"], args.llm)

    print("\n⚡ Revoking the same tuple (offboarding in one call)")
    admin.fga_delete_tuples(FgaWriteTuplesRequest(tuples=[grant]))
    show_user_turn(pipeline, authz, "bob@example.com", tokens["bob@example.com"], args.llm)

    print(f"\n{'=' * 60}")
    print("✅ Demo complete.")
    print("   Permissions lived in tuples, not the index: granting and")
    print("   revoking access never touched Qdrant or the embeddings.")
    print("=" * 60)


if __name__ == "__main__":
    main()
