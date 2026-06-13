"""
authz.py
--------
Fine-grained authorization (FGA) gateway, built on the official Authorizer
Python SDK (authorizer-py: https://github.com/authorizerdev/authorizer-py).

Authorizer embeds OpenFGA — the open-source implementation of Google's Zanzibar
relationship-based access control (ReBAC). This module is the thin, fail-closed
bridge between the RAG pipeline and that permission engine:

    login(email, password)    → an access token for the user.
    allowed_documents(token)  → which documents may THIS user retrieve?
    verify_sources(token, …)  → double-check citations before showing them.

The two permission calls authenticate with the END USER's token, never an admin
key. The server pins the permission subject from the token, so a prompt-injected
agent has no credential to escalate with — it asks as the user, it gets the
user's answer.

Every failure mode here is FAIL CLOSED: a network error, a GraphQL error, or a
truncated permission list raises AuthorizationError (or returns False for
verify_sources). "Auth is down" must never degrade to "everything is visible".

The SDK already sends the CSRF `Origin` header required by Authorizer >= v2.3.0
(defaulting to the server's own origin), so server-side calls are not rejected.
"""

from authorizer import (
    AuthorizerClient,
    CheckPermissionsRequest,
    ListPermissionsRequest,
    LoginRequest,
    PermissionCheckInput,
)
from authorizer.exceptions import AuthorizerError

# The default Authorizer server address. See docker-compose.yml.
AUTHORIZER_BASE_URL = "http://localhost:8080"
# The demo's client id (value of the server's --client-id flag).
DEFAULT_CLIENT_ID = "123456"

# Documents are modelled as FGA objects named after the chunk payload's
# `source` field (the filename), e.g. "document:security_policy.txt".
# This makes the Qdrant payload the exact join key — no mapping table.
DOCUMENT_TYPE = "document"
VIEW_RELATION = "can_view"


class AuthorizationError(Exception):
    """Raised when a permission decision cannot be made safely (fail closed)."""


class AuthzClient:
    """
    Wraps the Authorizer Python SDK for login and FGA checks.

    Example:
        authz = AuthzClient()                      # http://localhost:8080
        token = authz.login("bob@example.com", "Demo@Pass123")
        authz.allowed_documents(token)             # ["onboarding_guide.txt"]
    """

    def __init__(
        self,
        base_url: str = AUTHORIZER_BASE_URL,
        client_id: str = DEFAULT_CLIENT_ID,
    ) -> None:
        """
        Args:
            base_url:  URL of the Authorizer server.
            client_id: Authorizer client id (value of the server's --client-id).
        """
        self.base_url = base_url.rstrip("/")
        self._client = AuthorizerClient(client_id, self.base_url)

    # ── Public API ──────────────────────────────────────────────────────────

    def login(self, email: str, password: str) -> str:
        """
        Authenticate a user and return their access token.

        Raises:
            AuthorizationError: On bad credentials or an unreachable server.
        """
        try:
            res = self._client.login(LoginRequest(email=email, password=password))
        except AuthorizerError as e:
            raise AuthorizationError(str(e)) from e
        if not res.access_token:
            raise AuthorizationError("login succeeded but returned no access token")
        return res.access_token

    def allowed_documents(self, user_token: str) -> list[str]:
        """
        Return the `source` names of every document the caller may view.

        Asks Authorizer's `list_permissions` for all objects of type `document`
        the token's subject holds `can_view` on, then strips the "document:"
        prefix so the result plugs straight into Qdrant's payload filter.

        Args:
            user_token: The END USER's access token (subject is pinned
                        server-side from this token).

        Returns:
            List of source names, e.g. ["onboarding_guide.txt"]. Empty list
            means the user may see nothing — a valid, enforceable answer.

        Raises:
            AuthorizationError: On any error, or if the permission list was
                                truncated (a partial allow-list must not be
                                mistaken for the full one).
        """
        try:
            res = self._client.list_permissions(
                ListPermissionsRequest(
                    relation=VIEW_RELATION, object_type=DOCUMENT_TYPE
                ),
                headers=self._bearer(user_token),
            )
        except AuthorizerError as e:
            raise AuthorizationError(str(e)) from e
        if res.truncated:
            raise AuthorizationError(
                "permission list truncated (>1000 grants); refusing partial view"
            )
        prefix = f"{DOCUMENT_TYPE}:"
        return [
            o[len(prefix):] if o.startswith(prefix) else o
            for o in res.objects
        ]

    def verify_sources(self, user_token: str, sources: list[str]) -> bool:
        """
        Batch-verify that the caller may view every cited source document.

        Defense in depth: called after generation, before the answer is shown,
        so a grant revoked mid-request can't leak through stale retrieval.

        Args:
            user_token: The END USER's access token.
            sources:    `source` values of the retrieved chunks.

        Returns:
            True only if EVERY source is allowed. Any error returns False.
        """
        if not sources:
            return True
        unique = sorted(set(sources))
        checks = [
            PermissionCheckInput(relation=VIEW_RELATION, object=f"{DOCUMENT_TYPE}:{s}")
            for s in unique
        ]
        try:
            res = self._client.check_permissions(
                CheckPermissionsRequest(checks=checks),
                headers=self._bearer(user_token),
            )
        except AuthorizerError:
            return False
        if len(res.results) != len(checks):
            return False
        return all(r.allowed is True for r in res.results)

    def close(self) -> None:
        """Release the underlying SDK HTTP client."""
        self._client.close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _bearer(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}
