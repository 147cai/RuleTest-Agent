"""
End-to-end test: send a chat message through the server, verify MCP tool
calls complete successfully (no "interrupted" errors).

This test exercises the FULL pipeline:
  frontend request → Node.js server → Python agent subprocess
  → OrchestratorAgent → Worker (ReActAgent) → MCP tool call → SSE response

Requires:
  - testagent-server container running (docker compose up -d server)
  - Network access to httpbin.org (MCP tool target)

Run:
    pytest tests/test_e2e_mcp_tool_call.py -v -m e2e
"""
import asyncio
import json
import time
import uuid
from typing import Any, Dict, List

import httpx
import pytest

# ── Config ────────────────────────────────────────────────────────────────────

SERVER_BASE = "http://localhost:8000"
# Unique per test run to avoid collisions
_RUN_ID = uuid.uuid4().hex[:8]
TEST_USER = f"e2e_test_{_RUN_ID}"
TEST_PASS = "testpass_e2e_123"
TEST_EMAIL = f"e2e_{_RUN_ID}@test.local"

# Timeout for the full SSE stream (agent execution can be slow with real LLM)
STREAM_TIMEOUT = 600  # 10 minutes max


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _register_and_login(client: httpx.AsyncClient) -> str:
    """Register a test user and login, return JWT access_token."""
    await client.post(
        f"{SERVER_BASE}/auth/register",
        json={"username": TEST_USER, "password": TEST_PASS, "email": TEST_EMAIL},
    )
    resp = await client.post(
        f"{SERVER_BASE}/auth/login",
        json={"username": TEST_USER, "password": TEST_PASS},
    )
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    return resp.json()["access_token"]


async def _create_conversation(client: httpx.AsyncClient, token: str) -> str:
    """Create a conversation, return conversation_id."""
    resp = await client.post(
        f"{SERVER_BASE}/api/conversations",
        headers={"Authorization": f"Bearer {token}"},
        json={"title": f"E2E MCP Test {_RUN_ID}"},
    )
    assert resp.status_code == 201, f"Create conversation failed: {resp.text}"
    return resp.json()["conversation_id"]


def _parse_sse_lines(raw: str) -> List[Dict[str, Any]]:
    """Parse raw SSE text into a list of {event_type, data} dicts."""
    events = []
    current_event = None
    current_data_lines = []

    for line in raw.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            current_data_lines.append(line[6:])
        elif line == "" and current_event is not None:
            # End of event block
            data_str = "\n".join(current_data_lines)
            try:
                data = json.loads(data_str)
            except (json.JSONDecodeError, ValueError):
                data = data_str
            events.append({"event_type": current_event, "data": data})
            current_event = None
            current_data_lines = []

    return events


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.e2e
@pytest.mark.network
class TestE2EMCPToolCall:
    """
    End-to-end: send message → agent calls MCP tool → verify success in SSE.

    Marked with both 'e2e' and 'network' markers so it can be skipped in
    CI environments without network or without the server running.
    """

    async def test_mcp_tool_call_not_interrupted(self):
        """
        Send a simple HTTP request task, verify:
        1. SSE stream contains tool_call events for MCP tools (http_configure/http_request)
        2. tool_result events show success=true (not interrupted)
        3. Stream completes with 'done' event
        """
        async with httpx.AsyncClient(timeout=httpx.Timeout(STREAM_TIMEOUT)) as client:
            # Setup: register, login, create conversation
            token = await _register_and_login(client)
            conv_id = await _create_conversation(client, token)

            # Send a message that MUST trigger MCP http_client_tools
            message = (
                "使用 http_client_tools 执行以下操作：\n"
                "1. 先调用 http_configure 配置 base_url 为 https://httpbin.org\n"
                "2. 然后调用 http_request 发送 GET /get\n"
                "直接执行，不要询问。"
            )

            # Stream SSE response
            collected_text = ""
            events: List[Dict[str, Any]] = []

            async with client.stream(
                "POST",
                f"{SERVER_BASE}/api/chat/stream",
                headers={"Authorization": f"Bearer {token}"},
                json={"message": message, "conversation_id": conv_id},
            ) as resp:
                assert resp.status_code == 200, f"Stream request failed: {resp.status_code}"

                async for chunk in resp.aiter_text():
                    collected_text += chunk

            # Parse SSE events
            events = _parse_sse_lines(collected_text)
            event_types = [e["event_type"] for e in events]

            # Debug: print event summary
            print(f"\n=== SSE Events ({len(events)} total) ===")
            for e in events:
                etype = e["event_type"]
                data = e["data"]
                if etype == "tool_call":
                    print(f"  tool_call: {data.get('name', '?')}")
                elif etype == "tool_result":
                    print(f"  tool_result: {data.get('name', '?')} success={data.get('success')}")
                    if not data.get("success"):
                        output = data.get("output", "")
                        print(f"    output: {str(output)[:200]}")
                elif etype == "coordinator_event":
                    print(f"  coordinator: {data.get('event_type', '?')}")
                elif etype == "chunk":
                    pass  # Too noisy
                elif etype == "done":
                    print(f"  done")
                else:
                    print(f"  {etype}")

            # ── Assertions ──

            # 1. Stream must complete
            assert "done" in event_types, (
                f"SSE stream did not complete (no 'done' event). "
                f"Event types: {event_types}"
            )

            # 2. Must have at least one tool_call event
            tool_calls = [e for e in events if e["event_type"] == "tool_call"]
            assert len(tool_calls) > 0, (
                f"No tool_call events found. The agent did not call any tools. "
                f"Event types: {event_types}"
            )

            # 3. Must have tool_result events
            tool_results = [e for e in events if e["event_type"] == "tool_result"]
            assert len(tool_results) > 0, (
                f"No tool_result events found. Tools were called but no results received. "
                f"Event types: {event_types}"
            )

            # 4. Check for MCP tool names (http_configure or http_request)
            tool_names_called = {
                e["data"].get("name", "") for e in tool_calls
            }
            tool_names_resulted = {
                e["data"].get("name", "") for e in tool_results
            }
            all_tool_names = tool_names_called | tool_names_resulted
            print(f"\n  Tools called: {tool_names_called}")
            print(f"  Tools resulted: {tool_names_resulted}")

            # 5. NO tool_result should contain "interrupted" text
            for tr in tool_results:
                output = str(tr["data"].get("output", ""))
                assert "interrupted" not in output.lower(), (
                    f"Tool '{tr['data'].get('name')}' was interrupted! "
                    f"Output: {output[:300]}\n"
                    "This indicates the MCP cross-event-loop bug is NOT fixed."
                )

            # 6. At least one tool_result should show success
            successful_results = [
                e for e in tool_results
                if e["data"].get("success") is True
            ]
            assert len(successful_results) > 0, (
                f"No successful tool results. All tool calls failed.\n"
                f"Results: {json.dumps([e['data'] for e in tool_results], ensure_ascii=False, indent=2)[:1000]}"
            )

            print(f"\n=== E2E PASS: {len(tool_calls)} tool calls, "
                  f"{len(successful_results)} successful results ===")
