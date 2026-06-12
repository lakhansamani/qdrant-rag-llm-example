"""
authz.py
--------
Fine-grained authorization (FGA) client for Authorizer's embedded OpenFGA engine.

Authorizer (https://github.com/authorizerdev/authorizer) is an open-source,
self-hosted auth server that embeds OpenFGA — the open-source implementation of
Google's Zanzibar relationship-based access control (ReBAC). This module is the
bridge between the RAG pipeline and that permission engine:

    allowed_documents(token)  → which documents may THIS user retrieve?
    verify_sources(token, …)  → double-check citations before showing them.

Both calls authenticate with the END USER's token, never an admin key. The
server pins the permission subject from the token, so a prompt-injected agent
has no credential to escalate with — it asks as the user, it gets the user's
answer.

Every failure mode here is FAIL CLOSED: a network error, a GraphQL error, or a
truncated permission list raises AuthorizationError (or returns False for
verify_sources). "Auth is down" must never degrade to "everything is visible".

Uses only Python's built-in urllib — no extra dependencies required.
"""

import json
import urllib.error
import urllib.request
from typing import Any

# The default Authorizer server address. See docker-compose.yml.
AUTHORIZER_BASE_URL = "http://localhost:8080"

# Documents are modelled as FGA objects named after the chunk payload's
# `source` field (the filename), e.g. "document:security_policy.txt".
# This makes the Qdrant payload the exact join key — no mapping table.
DOCUMENT_TYPE = "document"
VIEW_RELATION = "can_view"

_LOGIN_MUTATION = """
mutation login($params: LoginRequest!) {
  login(params: $params) { access_token }
}"""

_LIST_PERMISSIONS_QUERY = """
query listPermissions($params: ListPermissionsInput!) {
  list_permissions(params: $params) { objects truncated }
}"""

_CHECK_PERMISSIONS_QUERY = """
query checkPermissions($params: CheckPermissionsInput!) {
  check_permissions(params: $params) { results { relation object allowed } }
}"""


class AuthorizationError(Exception):
    """Raised when a permission decision cannot be made safely (fail closed)."""


class AuthzClient:
    """
    Talks to an Authorizer server's GraphQL API for login and FGA checks.

    Example:
        authz = AuthzClient()                      # http://localhost:8080
        token = authz.login("bob@example.com", "Password@123")
        authz.allowed_documents(token)             # ["onboarding_guide.txt"]
    """

    def __init__(
        self,
        base_url: str = AUTHORIZER_BASE_URL,
        timeout: float = 10.0,
    ) -> None:
        """
        Args:
            base_url: URL of the Authorizer server (the GraphQL endpoint is
                      derived as <base_url>/graphql).
            timeout:  Per-request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ── Internal helpers ────────────────────────────────────────────────────

    def _graphql(
        self,
        query: str,
        variables: dict[str, Any],
        token: str | None = None,
    ) -> dict[str, Any]:
        """
        POST a GraphQL request and return the `data` object.

        Raises:
            AuthorizationError: On any transport or GraphQL error. Callers must
                                treat this as "no access", never "all access".
        """
        payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/graphql",
            data=payload,
            headers={
                "Content-Type": "application/json",
                # Authorizer's CSRF guard requires Origin/Referer on POSTs.
                "Origin": self.base_url,
                **({"Authorization": f"Bearer {token}"} if token else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            raise AuthorizationError(f"Authorizer unreachable: {e}") from e

        if body.get("errors"):
            raise AuthorizationError(body["errors"][0].get("message", "GraphQL error"))
        return body.get("data") or {}

    # ── Public API ──────────────────────────────────────────────────────────

    def login(self, email: str, password: str) -> str:
        """
        Authenticate a user and return their access token.

        Args:
            email:    The user's email address.
            password: The user's password.

        Returns:
            A JWT access token to pass to allowed_documents / verify_sources.

        Raises:
            AuthorizationError: On bad credentials or an unreachable server.
        """
        data = self._graphql(
            _LOGIN_MUTATION,
            {"params": {"email": email, "password": password}},
        )
        token = (data.get("login") or {}).get("access_token")
        if not token:
            raise AuthorizationError("login succeeded but returned no access token")
        return token

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
        data = self._graphql(
            _LIST_PERMISSIONS_QUERY,
            {"params": {"relation": VIEW_RELATION, "object_type": DOCUMENT_TYPE}},
            token=user_token,
        )
        result = data.get("list_permissions") or {}
        if result.get("truncated"):
            raise AuthorizationError(
                "permission list truncated (>1000 grants); refusing partial view"
            )
        prefix = f"{DOCUMENT_TYPE}:"
        return [o.removeprefix(prefix) for o in result.get("objects") or []]

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
        checks = [
            {"relation": VIEW_RELATION, "object": f"{DOCUMENT_TYPE}:{s}"}
            for s in sorted(set(sources))
        ]
        try:
            data = self._graphql(
                _CHECK_PERMISSIONS_QUERY,
                {"params": {"checks": checks}},
                token=user_token,
            )
        except AuthorizationError:
            return False
        results = (data.get("check_permissions") or {}).get("results") or []
        if len(results) != len(checks):
            return False
        return all(r.get("allowed") is True for r in results)
