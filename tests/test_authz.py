"""
tests/test_authz.py
-------------------
Unit tests for the AuthzClient (Authorizer / OpenFGA permission gateway).

These tests mock the HTTP layer — no Authorizer server is needed. They pin
down the property the whole feature rests on: every failure mode is FAIL
CLOSED. An unreachable server, a GraphQL error, a truncated permission list,
or a short result set must never widen access.

Run with:
    pytest tests/test_authz.py -v
"""

import io
import json
import urllib.error
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from src.authz import AuthorizationError, AuthzClient


# ── Helpers ───────────────────────────────────────────────────────────────────

def _http_response(body: dict):
    """Build a context-manager mimicking urllib's HTTP response object."""
    @contextmanager
    def _cm(*args, **kwargs):
        yield io.BytesIO(json.dumps(body).encode("utf-8"))
    return _cm


def _patch_urlopen(body: dict):
    """Patch urllib.request.urlopen inside src.authz to return `body` as JSON."""
    return patch("src.authz.urllib.request.urlopen", new=_http_response(body))


@pytest.fixture
def client():
    return AuthzClient(base_url="http://localhost:8080")


# ── login ─────────────────────────────────────────────────────────────────────

class TestLogin:

    def test_login_returns_access_token(self, client):
        body = {"data": {"login": {"access_token": "jwt-123"}}}
        with _patch_urlopen(body):
            assert client.login("a@example.com", "pw") == "jwt-123"

    def test_login_bad_credentials_raises(self, client):
        body = {"errors": [{"message": "bad user credentials"}]}
        with _patch_urlopen(body), pytest.raises(AuthorizationError, match="credentials"):
            client.login("a@example.com", "wrong")

    def test_login_missing_token_raises(self, client):
        body = {"data": {"login": {}}}
        with _patch_urlopen(body), pytest.raises(AuthorizationError, match="no access token"):
            client.login("a@example.com", "pw")


# ── allowed_documents ────────────────────────────────────────────────────────

class TestAllowedDocuments:

    def test_strips_document_prefix(self, client):
        body = {"data": {"list_permissions": {
            "objects": ["document:onboarding_guide.txt", "document:tech_stack.txt"],
            "truncated": False,
        }}}
        with _patch_urlopen(body):
            assert client.allowed_documents("jwt") == [
                "onboarding_guide.txt", "tech_stack.txt",
            ]

    def test_empty_grant_list_is_valid_not_error(self, client):
        """No grants is an enforceable answer (deny all), not a failure."""
        body = {"data": {"list_permissions": {"objects": [], "truncated": False}}}
        with _patch_urlopen(body):
            assert client.allowed_documents("jwt") == []

    def test_truncated_list_fails_closed(self, client):
        """A partial allow-list must never be mistaken for the full one."""
        body = {"data": {"list_permissions": {
            "objects": ["document:a.txt"], "truncated": True,
        }}}
        with _patch_urlopen(body), pytest.raises(AuthorizationError, match="truncated"):
            client.allowed_documents("jwt")

    def test_graphql_error_fails_closed(self, client):
        body = {"errors": [{"message": "unauthorized"}]}
        with _patch_urlopen(body), pytest.raises(AuthorizationError):
            client.allowed_documents("expired-jwt")

    def test_server_unreachable_fails_closed(self, client):
        def _raise(*args, **kwargs):
            raise urllib.error.URLError("connection refused")
        with patch("src.authz.urllib.request.urlopen", new=_raise):
            with pytest.raises(AuthorizationError, match="unreachable"):
                client.allowed_documents("jwt")

    def test_sends_user_token_as_bearer_header(self, client):
        """The permission call must carry the END USER's token."""
        captured = {}

        @contextmanager
        def _capture(request, timeout=None):
            captured["auth"] = request.get_header("Authorization")
            body = {"data": {"list_permissions": {"objects": [], "truncated": False}}}
            yield io.BytesIO(json.dumps(body).encode("utf-8"))

        with patch("src.authz.urllib.request.urlopen", new=_capture):
            client.allowed_documents("user-jwt")
        assert captured["auth"] == "Bearer user-jwt"


# ── verify_sources ───────────────────────────────────────────────────────────

class TestVerifySources:

    def test_all_allowed_returns_true(self, client):
        body = {"data": {"check_permissions": {"results": [
            {"relation": "can_view", "object": "document:a.txt", "allowed": True},
            {"relation": "can_view", "object": "document:b.txt", "allowed": True},
        ]}}}
        with _patch_urlopen(body):
            assert client.verify_sources("jwt", ["a.txt", "b.txt"]) is True

    def test_one_denied_returns_false(self, client):
        body = {"data": {"check_permissions": {"results": [
            {"relation": "can_view", "object": "document:a.txt", "allowed": True},
            {"relation": "can_view", "object": "document:b.txt", "allowed": False},
        ]}}}
        with _patch_urlopen(body):
            assert client.verify_sources("jwt", ["a.txt", "b.txt"]) is False

    def test_error_returns_false_not_raise(self, client):
        """Post-generation check degrades to deny, never to allow."""
        body = {"errors": [{"message": "fga is not enabled"}]}
        with _patch_urlopen(body):
            assert client.verify_sources("jwt", ["a.txt"]) is False

    def test_short_result_set_returns_false(self, client):
        """Fewer results than checks must not pass silently."""
        body = {"data": {"check_permissions": {"results": [
            {"relation": "can_view", "object": "document:a.txt", "allowed": True},
        ]}}}
        with _patch_urlopen(body):
            assert client.verify_sources("jwt", ["a.txt", "b.txt"]) is False

    def test_duplicate_sources_checked_once(self, client):
        """Chunks from the same document produce a single check."""
        body = {"data": {"check_permissions": {"results": [
            {"relation": "can_view", "object": "document:a.txt", "allowed": True},
        ]}}}
        with _patch_urlopen(body):
            assert client.verify_sources("jwt", ["a.txt", "a.txt", "a.txt"]) is True

    def test_no_sources_is_trivially_true(self, client):
        assert client.verify_sources("jwt", []) is True
