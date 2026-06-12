"""
scripts/fga_seed.py
-------------------
One-shot, idempotent setup for the permission-aware (FGA) demo.

Against a running Authorizer server (see docker-compose.yml) this script:
  1. Creates two demo users (skipped if they already exist):
       alice@example.com — engineering team member
       bob@example.com   — new hire, no team
  2. Installs the authorization model (skipped if it is already active).
  3. Writes the relationship tuples that grant document access
     (only the missing ones — safe to re-run).

The resulting access matrix over data/knowledge_base/:

    document                 alice   bob    why
    ---------------------    -----   ----   ------------------------------
    onboarding_guide.txt     yes     yes    public (user:* viewer)
    tech_stack.txt           yes     no     team:engineering#member viewer
    security_policy.txt      no      no     team:security#member viewer

Usage:
    python scripts/fga_seed.py
    python scripts/fga_seed.py --authorizer http://localhost:8080 --admin-secret admin
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.authz import AuthorizationError, AuthzClient

DEMO_PASSWORD = "Demo@Pass123"  # demo-only; satisfies the strong-password policy

DEMO_USERS = ["alice@example.com", "bob@example.com"]

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

# (user, relation, object) grants. {alice} is replaced with alice's user id.
TUPLES = [
    ("user:*", "viewer", "document:onboarding_guide.txt"),
    ("user:{alice}", "member", "team:engineering"),
    ("team:engineering#member", "viewer", "document:tech_stack.txt"),
    ("team:security#member", "viewer", "document:security_policy.txt"),
]


class AdminClient:
    """Minimal admin-side GraphQL client (X-Authorizer-Admin-Secret auth)."""

    def __init__(self, base_url: str, admin_secret: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.admin_secret = admin_secret
        self.timeout = timeout

    def graphql(self, query: str, variables: dict[str, Any] | None = None,
                admin: bool = True) -> dict[str, Any]:
        """POST a GraphQL request; raises RuntimeError on any error."""
        # Authorizer's CSRF guard requires Origin/Referer on POSTs.
        headers = {"Content-Type": "application/json", "Origin": self.base_url}
        if admin:
            headers["X-Authorizer-Admin-Secret"] = self.admin_secret
        payload = json.dumps({"query": query, "variables": variables or {}})
        request = urllib.request.Request(
            f"{self.base_url}/graphql",
            data=payload.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"Authorizer unreachable at {self.base_url}: {e}") from e
        if body.get("errors"):
            raise RuntimeError(body["errors"][0].get("message", "GraphQL error"))
        return body.get("data") or {}


# ── Seed steps ────────────────────────────────────────────────────────────────

def ensure_user(admin: AdminClient, authz: AuthzClient, email: str) -> str:
    """Sign the user up (tolerating 'already exists'), return their user id."""
    try:
        data = admin.graphql(
            """mutation signup($params: SignUpRequest!) {
                 signup(params: $params) { user { id } }
               }""",
            {"params": {
                "email": email,
                "password": DEMO_PASSWORD,
                "confirm_password": DEMO_PASSWORD,
            }},
            admin=False,
        )
        user_id = data["signup"]["user"]["id"]
        print(f"  ✓ Created user {email}")
        return user_id
    except RuntimeError as signup_error:
        # The server's signup error is deliberately generic (it doesn't reveal
        # whether the account exists). A successful login with the demo
        # password is the reliable "already seeded" signal.
        try:
            authz.login(email, DEMO_PASSWORD)
        except AuthorizationError:
            raise RuntimeError(f"signup failed for {email}: {signup_error}") from signup_error
    data = admin.graphql(
        """query users { _users(params: { pagination: { limit: 50 } }) {
             users { id email } } }"""
    )
    for user in data["_users"]["users"]:
        if user["email"] == email:
            print(f"  ✓ User {email} already exists")
            return user["id"]
    raise RuntimeError(f"could not resolve id for existing user {email}")


def ensure_model(admin: AdminClient) -> None:
    """Install MODEL_DSL unless the active model already matches it."""
    def _normalise(dsl: str) -> list[str]:
        # The server returns the DSL with relations alphabetised within each
        # type, so compare the sorted set of meaningful lines instead of the
        # exact text.
        return sorted(line.strip() for line in dsl.splitlines() if line.strip())

    try:
        current = admin.graphql("query { _fga_get_model { id dsl } }")
        if _normalise(current["_fga_get_model"]["dsl"]) == _normalise(MODEL_DSL):
            print("  ✓ Authorization model already active")
            return
    except RuntimeError:
        pass  # No model yet (or FGA freshly enabled) — write one.

    data = admin.graphql(
        """mutation writeModel($params: FgaWriteModelInput!) {
             _fga_write_model(params: $params) { id }
           }""",
        {"params": {"dsl": MODEL_DSL}},
    )
    print(f"  ✓ Installed authorization model {data['_fga_write_model']['id']}")


def ensure_tuples(admin: AdminClient, alice_id: str) -> None:
    """Write the demo grants that don't exist yet (idempotent)."""
    existing: set[tuple[str, str, str]] = set()
    data = admin.graphql(
        """query readTuples($params: FgaReadTuplesInput!) {
             _fga_read_tuples(params: $params) { tuples { user relation object } }
           }""",
        {"params": {"page_size": 100}},
    )
    for t in data["_fga_read_tuples"]["tuples"]:
        existing.add((t["user"], t["relation"], t["object"]))

    wanted = [
        (user.format(alice=alice_id), relation, obj)
        for user, relation, obj in TUPLES
    ]
    missing = [t for t in wanted if t not in existing]
    if not missing:
        print("  ✓ All grants already in place")
        return

    admin.graphql(
        """mutation writeTuples($params: FgaWriteTuplesInput!) {
             _fga_write_tuples(params: $params) { message }
           }""",
        {"params": {"tuples": [
            {"user": u, "relation": r, "object": o} for u, r, o in missing
        ]}},
    )
    for u, r, o in missing:
        print(f"  ✓ Granted: {u}  {r}  {o}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Authorizer FGA for the demo")
    parser.add_argument("--authorizer", default="http://localhost:8080",
                        help="Authorizer server URL")
    parser.add_argument("--admin-secret", default="admin",
                        help="Authorizer admin secret (see docker-compose.yml)")
    args = parser.parse_args()

    admin = AdminClient(args.authorizer, args.admin_secret)
    authz = AuthzClient(args.authorizer)

    print("=" * 60)
    print("RAG LOCAL DEMO — FGA Seed (Authorizer + OpenFGA)")
    print("=" * 60)

    print("\n⏳ Ensuring demo users exist...")
    user_ids = {email: ensure_user(admin, authz, email) for email in DEMO_USERS}

    print("\n⏳ Ensuring authorization model is active...")
    ensure_model(admin)

    print("\n⏳ Ensuring relationship tuples...")
    ensure_tuples(admin, alice_id=user_ids["alice@example.com"])

    print(f"\n{'=' * 60}")
    print("✅ Seed complete. Demo credentials:")
    for email in DEMO_USERS:
        print(f"   {email}  /  {DEMO_PASSWORD}")
    print("\n💡 Next:")
    print("   python scripts/fga_demo.py          # CLI walk-through")
    print("   python src/app.py --authorizer " + args.authorizer)
    print("=" * 60)


if __name__ == "__main__":
    main()
