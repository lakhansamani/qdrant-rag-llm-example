"""
tests/test_authz.py
-------------------
Unit tests for the AuthzClient FGA wrapper around the official Authorizer
Python SDK (authorizer-py).

These tests inject a fake SDK client — no Authorizer server is needed. They pin
down the property the whole feature rests on: every failure mode is FAIL
CLOSED. A server error, an SDK exception, a truncated permission list, or a
short result set must never widen access.

Run with:
    pytest tests/test_authz.py -v
"""

import pytest

from authorizer.exceptions import AuthorizerConnectionError, AuthorizerError
from authorizer.types import (
    AuthToken,
    CheckPermissionsResponse,
    ListPermissionsResponse,
    PermissionCheckResult,
)

from src.authz import AuthorizationError, AuthzClient


# ── Fake SDK client ─────────────────────────────────────────────────────────

class FakeSDK:
    """Stand-in for authorizer.AuthorizerClient. Scriptable per method."""

    def __init__(self, *, login=None, list_resp=None, check_resp=None, raises=None):
        self._login = login
        self._list = list_resp
        self._check = check_resp
        self._raises = raises  # an AuthorizerError instance to raise, or None
        self.list_headers = None
        self.check_headers = None

    def login(self, req):
        if isinstance(self._raises, AuthorizerError):
            raise self._raises
        return self._login

    def list_permissions(self, req, headers=None):
        self.list_headers = headers
        if isinstance(self._raises, AuthorizerError):
            raise self._raises
        return self._list

    def check_permissions(self, req, headers=None):
        self.check_headers = headers
        if isinstance(self._raises, AuthorizerError):
            raise self._raises
        return self._check


def make_client(**fake_kwargs):
    """Build an AuthzClient with its SDK client replaced by a FakeSDK."""
    c = AuthzClient(base_url="http://localhost:8080")
    c._client = FakeSDK(**fake_kwargs)
    return c


# ── login ─────────────────────────────────────────────────────────────────────

class TestLogin:

    def test_login_returns_access_token(self):
        c = make_client(login=AuthToken(access_token="jwt-123"))
        assert c.login("a@example.com", "pw") == "jwt-123"

    def test_login_bad_credentials_raises(self):
        c = make_client(raises=AuthorizerError("bad user credentials"))
        with pytest.raises(AuthorizationError, match="credentials"):
            c.login("a@example.com", "wrong")

    def test_login_missing_token_raises(self):
        c = make_client(login=AuthToken(access_token=None))
        with pytest.raises(AuthorizationError, match="no access token"):
            c.login("a@example.com", "pw")


# ── allowed_documents ────────────────────────────────────────────────────────

class TestAllowedDocuments:

    def test_strips_document_prefix(self):
        c = make_client(list_resp=ListPermissionsResponse(
            objects=["document:onboarding_guide.txt", "document:tech_stack.txt"],
            truncated=False,
        ))
        assert c.allowed_documents("jwt") == ["onboarding_guide.txt", "tech_stack.txt"]

    def test_empty_grant_list_is_valid_not_error(self):
        """No grants is an enforceable answer (deny all), not a failure."""
        c = make_client(list_resp=ListPermissionsResponse(objects=[], truncated=False))
        assert c.allowed_documents("jwt") == []

    def test_truncated_list_fails_closed(self):
        """A partial allow-list must never be mistaken for the full one."""
        c = make_client(list_resp=ListPermissionsResponse(
            objects=["document:a.txt"], truncated=True,
        ))
        with pytest.raises(AuthorizationError, match="truncated"):
            c.allowed_documents("jwt")

    def test_sdk_error_fails_closed(self):
        c = make_client(raises=AuthorizerError("unauthorized"))
        with pytest.raises(AuthorizationError):
            c.allowed_documents("expired-jwt")

    def test_server_unreachable_fails_closed(self):
        """'Auth is down' must raise, never degrade to an open query."""
        c = make_client(raises=AuthorizerConnectionError("connection refused"))
        with pytest.raises(AuthorizationError):
            c.allowed_documents("jwt")

    def test_sends_user_token_as_bearer_header(self):
        """The permission call must carry the END USER's token."""
        c = make_client(list_resp=ListPermissionsResponse(objects=[], truncated=False))
        c.allowed_documents("user-jwt")
        assert c._client.list_headers == {"Authorization": "Bearer user-jwt"}


# ── verify_sources ───────────────────────────────────────────────────────────

class TestVerifySources:

    def test_all_allowed_returns_true(self):
        c = make_client(check_resp=CheckPermissionsResponse(results=[
            PermissionCheckResult(relation="can_view", object="document:a.txt", allowed=True),
            PermissionCheckResult(relation="can_view", object="document:b.txt", allowed=True),
        ]))
        assert c.verify_sources("jwt", ["a.txt", "b.txt"]) is True

    def test_one_denied_returns_false(self):
        c = make_client(check_resp=CheckPermissionsResponse(results=[
            PermissionCheckResult(relation="can_view", object="document:a.txt", allowed=True),
            PermissionCheckResult(relation="can_view", object="document:b.txt", allowed=False),
        ]))
        assert c.verify_sources("jwt", ["a.txt", "b.txt"]) is False

    def test_error_returns_false_not_raise(self):
        """Post-generation check degrades to deny, never to allow."""
        c = make_client(raises=AuthorizerError("fga is not enabled"))
        assert c.verify_sources("jwt", ["a.txt"]) is False

    def test_short_result_set_returns_false(self):
        """Fewer results than checks must not pass silently."""
        c = make_client(check_resp=CheckPermissionsResponse(results=[
            PermissionCheckResult(relation="can_view", object="document:a.txt", allowed=True),
        ]))
        assert c.verify_sources("jwt", ["a.txt", "b.txt"]) is False

    def test_duplicate_sources_checked_once(self):
        """Chunks from the same document produce a single check."""
        c = make_client(check_resp=CheckPermissionsResponse(results=[
            PermissionCheckResult(relation="can_view", object="document:a.txt", allowed=True),
        ]))
        assert c.verify_sources("jwt", ["a.txt", "a.txt", "a.txt"]) is True

    def test_no_sources_is_trivially_true(self):
        c = make_client()
        assert c.verify_sources("jwt", []) is True
