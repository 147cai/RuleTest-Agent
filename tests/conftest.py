"""
Pytest fixtures for TestAgent integration tests.
"""
import sys
import os
import pytest

# Make agent/ importable from tests.
# Two possible layouts:
#   Development (WSL):  TestAgent/tests/ → TestAgent/Client/agent/
#   Container:          /app/tests/       → /app/agent/
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_HERE, "..", "Client", "agent"),  # WSL dev layout
    os.path.join(_HERE, "..", "agent"),             # container layout (/app/agent)
]
for _candidate in _CANDIDATES:
    _candidate = os.path.abspath(_candidate)
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)


@pytest.fixture
def mcp_python() -> str:
    """
    Path to the Python interpreter that has http_client_mcp installed.
    Inside the container this is /usr/bin/python3; locally falls back to sys.executable.
    """
    candidates = ["/usr/bin/python3", sys.executable]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


@pytest.fixture
def mcp_config(mcp_python: str) -> dict:
    """
    MCP server configuration dict for the http_client_mcp server,
    matching the format expected by mcp_loader / agentscope MCPManager.
    """
    return {
        "http_client": {
            "command": mcp_python,
            "args": ["-m", "http_client_mcp.server"],
            "env": {},
        }
    }
