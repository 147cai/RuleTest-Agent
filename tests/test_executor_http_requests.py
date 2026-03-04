"""
Integration tests: MCP http_client_mcp server loading and stateful HTTP.

Tests 1-2 (mcp_loader / toolkit) require agentscope + http_client_mcp installed.
Tests 3-5 (stateful HTTP) require http_client_mcp + network access to httpbin.org.

Run with:
    cd TestAgent && pytest tests/test_executor_http_requests.py -v
Skip network tests:
    pytest tests/test_executor_http_requests.py -v -m "not network"
"""
import json
import sys
import pytest

# ── Dependency guards ──────────────────────────────────────────────────────────

# Skip the entire module if http_client_mcp is not installed
pytest.importorskip("http_client_mcp", reason="http_client_mcp not installed")

import requests as _requests  # noqa: E402 (needed by server module)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _reset_server_state() -> None:
    """Reset process-level state in http_client_mcp.server between tests."""
    from http_client_mcp import server as srv
    srv._session = _requests.Session()
    srv._config = {"base_url": "", "timeout": 30, "verify_ssl": True, "headers": {}}
    srv._history = []


HTTPBIN = "https://httpbin.org"


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
async def mcp_clients(mcp_config):
    """
    Function-scoped: start http_client_mcp via mcp_loader per test, then close.
    Function scope avoids asyncio event-loop mismatch with module-scoped fixtures.
    Skipped if agentscope is not installed.
    """
    pytest.importorskip("agentscope", reason="agentscope not installed")
    from mcp_loader import load_mcp_servers, close_mcp_servers

    clients = await load_mcp_servers(mcp_config)
    yield clients
    await close_mcp_servers(clients)


# ── MCP Loader Tests ──────────────────────────────────────────────────────────


def test_mcp_server_connects_via_import():
    """
    Sanity: http_client_mcp.server is importable and exposes 4 tool functions.
    Does NOT start a subprocess — just verifies the package is present.
    """
    from http_client_mcp import server as srv
    for name in ("http_configure", "http_request", "http_get_state", "http_clear_session"):
        assert callable(getattr(srv, name, None)), f"Missing tool function: {name}"


@pytest.mark.asyncio
async def test_mcp_loader_connects(mcp_config):
    """mcp_loader.load_mcp_servers connects to http_client_mcp and returns client dict."""
    pytest.importorskip("agentscope", reason="agentscope not installed")
    from mcp_loader import load_mcp_servers, close_mcp_servers

    clients = await load_mcp_servers(mcp_config)
    try:
        assert "http_client" in clients, f"Expected 'http_client' key, got: {list(clients.keys())}"
    finally:
        await close_mcp_servers(clients)


@pytest.mark.asyncio
async def test_mcp_tools_registered_count(mcp_clients, mcp_config):
    """
    After registering the http_client MCP client into a Toolkit,
    exactly 4 HTTP tools are available (http_configure, http_request,
    http_get_state, http_clear_session).
    """
    assert "http_client" in mcp_clients, (
        "MCP client not connected — check test_mcp_loader_connects for root cause."
    )

    pytest.importorskip("agentscope", reason="agentscope not installed")
    from agentscope.tool import Toolkit
    from tool_registry import _register_mcp_tools, _ensure_tool_group

    toolkit = Toolkit()
    group_name = "http_client_tools"
    _ensure_tool_group(toolkit, group_name, "HTTP Client Tools (MCP)")

    cfg_with_group = {
        "http_client": dict(mcp_config["http_client"], group=group_name)
    }
    await _register_mcp_tools(toolkit, mcp_clients, cfg_with_group)

    expected = {"http_configure", "http_request", "http_get_state", "http_clear_session"}

    # Probe common agentscope Toolkit attribute names for registered tool functions
    registered: set = set()
    for attr in ("_tool_functions", "_functions", "_tools", "tools"):
        container = getattr(toolkit, attr, None)
        if isinstance(container, dict) and container:
            registered = {name for name in container if name in expected}
            if registered:
                break
        elif isinstance(container, list) and container:
            registered = {
                (getattr(t, "name", None) or str(t))
                for t in container
                if (getattr(t, "name", None) or str(t)) in expected
            }
            if registered:
                break

    if not registered:
        # Last resort: check via dir() (callable attrs matching expected names)
        registered = {name for name in expected if callable(getattr(toolkit, name, None))}

    assert registered == expected, (
        f"Expected tools {expected}, got {registered}.\n"
        f"Toolkit public attrs: {[a for a in dir(toolkit) if not a.startswith('__')]}"
    )


# ── Stateful HTTP Tests (direct server calls, no agentscope needed) ───────────


@pytest.mark.network
def test_http_configure_and_get():
    """http_configure sets base_url; http_request GET /get returns HTTP 200."""
    from http_client_mcp.server import http_configure, http_request
    _reset_server_state()

    cfg = json.loads(http_configure(base_url=HTTPBIN, timeout=15))
    assert cfg["status"] == "ok", f"http_configure failed: {cfg}"
    assert cfg["config"]["base_url"] == HTTPBIN

    resp = json.loads(http_request(method="GET", url="/get"))
    assert resp["status"] == "success", f"http_request failed: {resp}"
    assert resp["status_code"] == 200, f"Unexpected status code: {resp['status_code']}"


@pytest.mark.network
def test_session_cookie_persistence():
    """
    KEY TEST: cookies set by /cookies/set are visible in the next /cookies call.

    httpbin /cookies/set?user=test redirects (302 → /cookies).
    requests.Session follows the redirect AND retains the Set-Cookie header,
    so the second GET /cookies should see user=test in the response body.
    """
    from http_client_mcp.server import http_configure, http_request, http_clear_session
    _reset_server_state()
    http_configure(base_url=HTTPBIN, timeout=15)

    # Step 1: set cookie (httpbin follows redirect automatically)
    set_resp = json.loads(http_request(method="GET", url="/cookies/set", params={"user": "test"}))
    assert set_resp["status_code"] == 200, (
        f"Cookie set request failed (status={set_resp['status_code']}): {set_resp}"
    )

    # Step 2: read cookies
    get_resp = json.loads(http_request(method="GET", url="/cookies"))
    assert get_resp["status_code"] == 200

    body = get_resp["body"]
    if isinstance(body, str):
        body = json.loads(body)
    cookies = body.get("cookies", {})
    assert cookies.get("user") == "test", (
        f"Cookie 'user' not persisted across requests. Got cookies: {cookies}"
    )


@pytest.mark.network
def test_http_clear_resets_cookies():
    """After http_clear_session(), http_get_state() shows cookies == {}."""
    from http_client_mcp.server import (
        http_configure, http_request, http_clear_session, http_get_state
    )
    _reset_server_state()
    http_configure(base_url=HTTPBIN, timeout=15)

    # Set a cookie first
    http_request(method="GET", url="/cookies/set", params={"session": "abc"})

    # Now clear
    clear_result = json.loads(http_clear_session())
    assert clear_result["status"] == "ok"

    # Verify cookies are gone
    state = json.loads(http_get_state())
    assert state["cookies"] == {}, (
        f"Cookies not cleared after http_clear_session(). Got: {state['cookies']}"
    )
