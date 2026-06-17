"""
scripts/fga_seed.py
-------------------
One-shot, idempotent setup for the permission-aware (FGA) demo.

Against a running Authorizer server (see docker-compose.yml) this script:
  1. Creates the demo users (skipped if they already exist):
       alice@example.com — engineering team member
       bob@example.com   — new hire, no team
       carol@example.com — finance team member
  2. Installs the authorization model (skipped if it is already active).
  3. Writes the relationship tuples that grant document access
     (only the missing ones — safe to re-run).

The resulting access matrix over data/knowledge_base/:

    document                 alice   bob    carol   why
    ---------------------    -----   ----   -----   ----------------------------
    onboarding_guide.txt     yes     yes    yes     public (user:* viewer)
    tech_stack.txt           yes     no     no      team:engineering#member viewer
    financial_report.txt     no      no     yes     team:finance#member viewer
    security_policy.txt      no      no     no      team:security#member viewer

So an engineer (alice) is BLOCKED from the financial report; only finance
(carol) can read it. Asking alice about Q4 revenue returns nothing — the
document is never retrieved, so the LLM can't leak it.

Usage:
    python scripts/fga_seed.py
    python scripts/fga_seed.py --authorizer http://localhost:8080 --admin-secret admin
"""

import argparse
import sys
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import Embedder
from src.retriever import Retriever
from src.vector_store import VectorStore

from authorizer import (
    AuthorizerAdminClient,
    AuthorizerClient,
    FgaReadTuplesRequest,
    FgaTupleInput,
    FgaWriteModelRequest,
    FgaWriteTuplesRequest,
    PaginatedRequest,
    PaginationRequest,
    SignUpRequest,
)
from authorizer.exceptions import AuthorizerError

from src.authz import AuthorizationError, AuthzClient

DEMO_PASSWORD = "Demo@Pass123"  # demo-only; satisfies the strong-password policy

DEMO_USERS = ["alice@example.com", "bob@example.com", "carol@example.com"]

# The authorization model. Documents are FGA objects named after the chunk
# payload's `source` field (the filename) — see src/authz.py.
MODEL_DSL = """model
  schema 1.1

type user

type team
  relations
    define member: [user]

type document
  relations
    define owner: [user]
    define viewer: [user, user:*, team#member]
    define blocked: [user]
    define can_view: (viewer or owner) but not blocked
"""

# (user, relation, object) grants. {alice}/{carol} are replaced with user ids.
TUPLES = [
    ("user:*", "viewer", "document:onboarding_guide.txt"),
    ("user:{alice}", "member", "team:engineering"),
    ("user:{carol}", "member", "team:finance"),
    ("team:engineering#member", "viewer", "document:tech_stack.txt"),
    ("team:finance#member", "viewer", "document:financial_report.txt"),
    ("team:security#member", "viewer", "document:security_policy.txt"),
]


# ── Seed steps ────────────────────────────────────────────────────────────────

def ensure_user(
    admin: AuthorizerAdminClient,
    user_client: AuthorizerClient,
    authz: AuthzClient,
    email: str,
) -> str:
    """Sign the user up (tolerating 'already exists'), return their user id."""
    try:
        res = user_client.signup(SignUpRequest(
            email=email,
            password=DEMO_PASSWORD,
            confirm_password=DEMO_PASSWORD,
        ))
        if res.user and res.user.id:
            print(f"  ✓ Created user {email}")
            return res.user.id
    except AuthorizerError as signup_error:
        # The server's signup error is deliberately generic (it doesn't reveal
        # whether the account exists). A successful login with the demo
        # password is the reliable "already seeded" signal.
        try:
            authz.login(email, DEMO_PASSWORD)
        except AuthorizationError:
            raise RuntimeError(f"signup failed for {email}: {signup_error}") from signup_error

    users_res = admin.users(PaginatedRequest(pagination=PaginationRequest(limit=50)))
    for user in users_res.users:
        if user.email == email:
            print(f"  ✓ User {email} already exists")
            return user.id
    raise RuntimeError(f"could not resolve id for existing user {email}")


def ensure_model(admin: AuthorizerAdminClient) -> None:
    """Install MODEL_DSL unless the active model already matches it."""
    def _normalise(dsl: str) -> list[str]:
        # The server returns the DSL with relations alphabetised within each
        # type, so compare the sorted set of meaningful lines instead of the
        # exact text.
        return sorted(line.strip() for line in dsl.splitlines() if line.strip())

    try:
        current = admin.fga_get_model()
        if _normalise(current.dsl) == _normalise(MODEL_DSL):
            print("  ✓ Authorization model already active")
            return
    except AuthorizerError:
        pass  # No model yet (or FGA freshly enabled) — write one.

    model = admin.fga_write_model(FgaWriteModelRequest(dsl=MODEL_DSL))
    print(f"  ✓ Installed authorization model {model.id}")


def ensure_tuples(admin: AuthorizerAdminClient, alice_id: str, carol_id: str) -> None:
    """Write the demo grants that don't exist yet (idempotent)."""
    existing: set[tuple[str, str, str]] = set()
    read_res = admin.fga_read_tuples(FgaReadTuplesRequest(page_size=100))
    for t in read_res.tuples:
        existing.add((t.user, t.relation, t.object))

    wanted = [
        (user.format(alice=alice_id, carol=carol_id), relation, obj)
        for user, relation, obj in TUPLES
    ]
    missing = [t for t in wanted if t not in existing]
    if not missing:
        print("  ✓ All grants already in place")
        return

    admin.fga_write_tuples(FgaWriteTuplesRequest(tuples=[
        FgaTupleInput(user=u, relation=r, object=o) for u, r, o in missing
    ]))
    for u, r, o in missing:
        print(f"  ✓ Granted: {u}  {r}  {o}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Authorizer FGA for the demo")
    parser.add_argument("--authorizer", default="http://localhost:8080",
                        help="Authorizer server URL")
    parser.add_argument("--admin-secret", default="admin",
                        help="Authorizer admin secret (see docker-compose.yml)")
    parser.add_argument("--client-id", default="123456",
                        help="Authorizer client id")
    parser.add_argument("--storage", default="http://localhost:6333",
                        help="Qdrant storage URL, file path, or ':memory:'")
    parser.add_argument("--data", default="data/knowledge_base",
                        help="Directory of .txt documents to ingest")
    args = parser.parse_args()

    admin = AuthorizerAdminClient(args.authorizer, args.admin_secret, protocol="rest")
    user_client = AuthorizerClient(args.client_id, args.authorizer, protocol="rest")
    authz = AuthzClient(args.authorizer, args.client_id)

    print("=" * 60)
    print("RAG LOCAL DEMO — FGA Seed (Authorizer + OpenFGA)")
    print("=" * 60)

    print("\n⏳ Ensuring demo users exist...")
    user_ids = {
        email: ensure_user(admin, user_client, authz, email)
        for email in DEMO_USERS
    }

    print("\n⏳ Ensuring authorization model is active...")
    ensure_model(admin)

    print("\n⏳ Ensuring relationship tuples...")
    ensure_tuples(
        admin,
        alice_id=user_ids["alice@example.com"],
        carol_id=user_ids["carol@example.com"],
    )

    print("\n⏳ Ingesting knowledge base documents...")
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"❌ Data directory not found: {data_path.resolve()}")
    embedder = Embedder()
    store = VectorStore(
        collection="knowledge_base",
        vector_size=embedder.vector_size,
        path=args.storage,
    )
    retriever = Retriever(store=store, embedder=embedder)
    retriever.ingest_directory(data_path)
    print(f"  ✓ {store.count()} chunks indexed in Qdrant")

    print(f"\n{'=' * 60}")
    print("✅ Seed complete. Demo credentials:")
    for email in DEMO_USERS:
        print(f"   {email}  /  {DEMO_PASSWORD}")
    print("\n💡 Next:")
    print("   python scripts/fga_demo.py --storage " + args.storage)
    print("   python src/app.py --authorizer " + args.authorizer + " --storage " + args.storage)
    print("=" * 60)


if __name__ == "__main__":
    main()
