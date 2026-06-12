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

from scripts.fga_seed import DEMO_PASSWORD, AdminClient
from src.authz import AuthzClient
from src.pipeline import RAGPipeline

QUESTIONS = [
    "What tech stack do we use for the frontend?",
    "How do I report a security incident?",
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
    parser.add_argument("--storage", default=":memory:",
                        help="Qdrant storage (':memory:', a path, or a server URL)")
    parser.add_argument("--data", default="data/knowledge_base",
                        help="Directory of .txt documents")
    parser.add_argument("--model", default="llama3.2", help="Ollama model (with --llm)")
    parser.add_argument("--llm", action="store_true",
                        help="Generate answers via Ollama (default: retrieval-only)")
    args = parser.parse_args()

    print("=" * 60)
    print("RAG LOCAL DEMO — Fine-Grained Permissions (FGA)")
    print("Authorizer (embedded OpenFGA) + Qdrant" + (" + Ollama" if args.llm else ""))
    print("=" * 60)

    authz = AuthzClient(args.authorizer)
    admin = AdminClient(args.authorizer, args.admin_secret)

    pipeline = RAGPipeline(
        storage_path=args.storage,
        llm_model=args.model,
        authz=authz,
    )
    pipeline.ingest_directory(Path(args.data))

    print("⏳ Logging in demo users (created by scripts/fga_seed.py)...")
    tokens = {
        email: authz.login(email, DEMO_PASSWORD)
        for email in ("bob@example.com", "alice@example.com")
    }

    # ── Act 1: same questions, different answers ──────────────────────────
    print("\n" + "─" * 60)
    print("ACT 1 — Same questions, permission-filtered retrieval")
    print("─" * 60)
    for email, token in tokens.items():
        show_user_turn(pipeline, authz, email, token, args.llm)

    # ── Act 2: grant, observe, revoke, observe ────────────────────────────
    print("\n" + "─" * 60)
    print("ACT 2 — Live grant & revocation (no re-ingestion, no re-login)")
    print("─" * 60)

    bob_id_query = """query users { _users(params: { pagination: { limit: 50 } }) {
        users { id email } } }"""
    bob_id = next(
        u["id"] for u in admin.graphql(bob_id_query)["_users"]["users"]
        if u["email"] == "bob@example.com"
    )
    grant = {"user": f"user:{bob_id}", "relation": "member", "object": "team:engineering"}

    print("\n⚡ Granting: bob → member → team:engineering (one tuple write)")
    admin.graphql(
        """mutation w($params: FgaWriteTuplesInput!) {
             _fga_write_tuples(params: $params) { message } }""",
        {"params": {"tuples": [grant]}},
    )
    show_user_turn(pipeline, authz, "bob@example.com", tokens["bob@example.com"], args.llm)

    print("\n⚡ Revoking the same tuple (offboarding in one call)")
    admin.graphql(
        """mutation d($params: FgaWriteTuplesInput!) {
             _fga_delete_tuples(params: $params) { message } }""",
        {"params": {"tuples": [grant]}},
    )
    show_user_turn(pipeline, authz, "bob@example.com", tokens["bob@example.com"], args.llm)

    print(f"\n{'=' * 60}")
    print("✅ Demo complete.")
    print("   Permissions lived in tuples, not the index: granting and")
    print("   revoking access never touched Qdrant or the embeddings.")
    print("=" * 60)


if __name__ == "__main__":
    main()
