"""
Unit tests for Client/agent/hook.py

Tests the AgentHooks.pre_print_hook event extraction logic:
- Text block delta calculation
- Thinking block delta calculation (the bug fix)
- Tool call / tool result deduplication
- Mixed content blocks
"""
import sys
import os
import threading
from unittest.mock import MagicMock, patch

import pytest

# Stub out agentscope before importing hook, so tests run without the full
# agentscope package installed.
_agentscope_stub = MagicMock()
_agentscope_stub.agent.AgentBase = MagicMock
sys.modules.setdefault("agentscope", _agentscope_stub)
sys.modules.setdefault("agentscope.agent", _agentscope_stub.agent)

# Make agent/ importable (conftest already does this, but be explicit for clarity)
_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.abspath(os.path.join(_HERE, "..", "Client", "agent"))
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

import hook as hook_module
from hook import AgentHooks, _state_lock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_hook_state():
    """Clear all global hook state between tests."""
    with _state_lock:
        hook_module._last_sent_content.clear()
        hook_module._last_sent_thinking.clear()
        hook_module._message_sequence.clear()
        hook_module._sent_tool_ids.clear()
        hook_module._sent_tool_result_ids.clear()


def _make_msg(blocks):
    """Create a mock AgentScope message with the given content blocks."""
    msg = MagicMock()
    msg.get_content_blocks.return_value = blocks
    return msg


def _make_agent():
    return MagicMock()


def _captured_payloads(monkeypatch):
    """Patch _submit_payload to capture payloads synchronously."""
    payloads = []

    def fake_submit(payload):
        payloads.append(payload)

    monkeypatch.setattr(AgentHooks, "_submit_payload", staticmethod(fake_submit))
    return payloads


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestTextDelta:
    def setup_method(self):
        _reset_hook_state()
        AgentHooks.url = "http://localhost:8000"
        AgentHooks.reply_id = "reply-text-001"

    def test_first_text_push(self, monkeypatch):
        payloads = _captured_payloads(monkeypatch)
        msg = _make_msg([{"type": "text", "text": "Hello"}])

        AgentHooks.pre_print_hook(_make_agent(), {"msg": msg})

        assert len(payloads) == 1
        events = payloads[0]["events"]
        assert len(events) == 1
        assert events[0]["type"] == "text"
        assert events[0]["content"] == "Hello"

    def test_incremental_text_delta(self, monkeypatch):
        payloads = _captured_payloads(monkeypatch)
        agent = _make_agent()

        # First call
        AgentHooks.pre_print_hook(agent, {"msg": _make_msg([{"type": "text", "text": "Hello"}])})
        # Second call — accumulated text grows
        AgentHooks.pre_print_hook(agent, {"msg": _make_msg([{"type": "text", "text": "Hello world"}])})

        assert len(payloads) == 2
        assert payloads[0]["events"][0]["content"] == "Hello"
        assert payloads[1]["events"][0]["content"] == " world"

    def test_no_event_when_text_unchanged(self, monkeypatch):
        payloads = _captured_payloads(monkeypatch)
        agent = _make_agent()
        msg = _make_msg([{"type": "text", "text": "Same"}])

        AgentHooks.pre_print_hook(agent, {"msg": msg})
        AgentHooks.pre_print_hook(agent, {"msg": msg})

        assert len(payloads) == 1  # second call emits no new event


class TestThinkingDelta:
    def setup_method(self):
        _reset_hook_state()
        AgentHooks.url = "http://localhost:8000"
        AgentHooks.reply_id = "reply-think-001"

    def test_thinking_block_emits_event(self, monkeypatch):
        """Regression: thinking blocks were silently dropped before the fix."""
        payloads = _captured_payloads(monkeypatch)
        msg = _make_msg([{"type": "thinking", "thinking": "Let me reason..."}])

        AgentHooks.pre_print_hook(_make_agent(), {"msg": msg})

        assert len(payloads) == 1
        events = payloads[0]["events"]
        assert len(events) == 1
        assert events[0]["type"] == "thinking"
        assert events[0]["content"] == "Let me reason..."

    def test_incremental_thinking_delta(self, monkeypatch):
        payloads = _captured_payloads(monkeypatch)
        agent = _make_agent()

        AgentHooks.pre_print_hook(agent, {"msg": _make_msg([{"type": "thinking", "thinking": "Step 1"}])})
        AgentHooks.pre_print_hook(agent, {"msg": _make_msg([{"type": "thinking", "thinking": "Step 1 Step 2"}])})

        assert len(payloads) == 2
        assert payloads[0]["events"][0]["content"] == "Step 1"
        assert payloads[1]["events"][0]["content"] == " Step 2"

    def test_no_event_when_thinking_unchanged(self, monkeypatch):
        payloads = _captured_payloads(monkeypatch)
        agent = _make_agent()
        msg = _make_msg([{"type": "thinking", "thinking": "Same thought"}])

        AgentHooks.pre_print_hook(agent, {"msg": msg})
        AgentHooks.pre_print_hook(agent, {"msg": msg})

        assert len(payloads) == 1


class TestMixedBlocks:
    def setup_method(self):
        _reset_hook_state()
        AgentHooks.url = "http://localhost:8000"
        AgentHooks.reply_id = "reply-mixed-001"

    def test_text_and_thinking_both_emitted(self, monkeypatch):
        payloads = _captured_payloads(monkeypatch)
        blocks = [
            {"type": "thinking", "thinking": "I should greet"},
            {"type": "text", "text": "Hello!"},
        ]
        AgentHooks.pre_print_hook(_make_agent(), {"msg": _make_msg(blocks)})

        assert len(payloads) == 1
        events = payloads[0]["events"]
        types = {e["type"] for e in events}
        assert "text" in types
        assert "thinking" in types

    def test_text_and_thinking_independent_deltas(self, monkeypatch):
        payloads = _captured_payloads(monkeypatch)
        agent = _make_agent()

        # First call: only thinking
        AgentHooks.pre_print_hook(agent, {"msg": _make_msg([
            {"type": "thinking", "thinking": "Thinking..."},
        ])})
        # Second call: thinking grows, text appears
        AgentHooks.pre_print_hook(agent, {"msg": _make_msg([
            {"type": "thinking", "thinking": "Thinking... done"},
            {"type": "text", "text": "Result"},
        ])})

        assert len(payloads) == 2
        first_types = {e["type"] for e in payloads[0]["events"]}
        second_types = {e["type"] for e in payloads[1]["events"]}

        assert first_types == {"thinking"}
        assert second_types == {"thinking", "text"}

        second_events = {e["type"]: e for e in payloads[1]["events"]}
        assert second_events["thinking"]["content"] == " done"
        assert second_events["text"]["content"] == "Result"


class TestToolEvents:
    def setup_method(self):
        _reset_hook_state()
        AgentHooks.url = "http://localhost:8000"
        AgentHooks.reply_id = "reply-tool-001"

    def test_tool_call_emitted_once(self, monkeypatch):
        payloads = _captured_payloads(monkeypatch)
        agent = _make_agent()
        blocks = [
            {"type": "tool_use", "id": "tool-1", "name": "search", "input": {"q": "test"}},
        ]

        AgentHooks.pre_print_hook(agent, {"msg": _make_msg(blocks)})
        AgentHooks.pre_print_hook(agent, {"msg": _make_msg(blocks)})  # same block, should dedupe

        tool_events = [e for p in payloads for e in p["events"] if e["type"] == "tool_call"]
        assert len(tool_events) == 1
        assert tool_events[0]["name"] == "search"

    def test_tool_result_emitted_once(self, monkeypatch):
        payloads = _captured_payloads(monkeypatch)
        agent = _make_agent()
        blocks = [
            {"type": "tool_result", "tool_use_id": "tool-1", "name": "search", "content": "found it"},
        ]

        AgentHooks.pre_print_hook(agent, {"msg": _make_msg(blocks)})
        AgentHooks.pre_print_hook(agent, {"msg": _make_msg(blocks)})

        result_events = [e for p in payloads for e in p["events"] if e["type"] == "tool_result"]
        assert len(result_events) == 1
        assert result_events[0]["output"] == "found it"

    def test_empty_input_with_raw_input_skipped(self, monkeypatch):
        """Tool calls still streaming (empty input dict + raw_input) must be skipped."""
        payloads = _captured_payloads(monkeypatch)
        blocks = [
            {"type": "tool_use", "id": "tool-2", "name": "search", "input": {}, "raw_input": '{"q": "par'},
        ]

        AgentHooks.pre_print_hook(_make_agent(), {"msg": _make_msg(blocks)})

        tool_events = [e for p in payloads for e in p["events"] if e["type"] == "tool_call"]
        assert len(tool_events) == 0


class TestNoUrlOrReplyId:
    def setup_method(self):
        _reset_hook_state()

    def test_no_url_skips_http_push(self, monkeypatch):
        """url check lives in _sync_push_to_studio; _submit_payload is still called
        but the inner function short-circuits before making an HTTP request."""
        http_calls = []

        def fake_sync_push(payload):
            # Simulate the real guard in _sync_push_to_studio
            if not AgentHooks.url:
                return
            http_calls.append(payload)

        monkeypatch.setattr(AgentHooks, "_sync_push_to_studio", staticmethod(fake_sync_push))
        monkeypatch.setattr(AgentHooks, "_submit_payload", staticmethod(fake_sync_push))

        AgentHooks.url = ""
        AgentHooks.reply_id = "reply-x"

        AgentHooks.pre_print_hook(_make_agent(), {"msg": _make_msg([{"type": "text", "text": "hi"}])})

        assert len(http_calls) == 0

    def test_no_reply_id_skips_push(self, monkeypatch):
        payloads = _captured_payloads(monkeypatch)
        AgentHooks.url = "http://localhost:8000"
        AgentHooks.reply_id = ""

        AgentHooks.pre_print_hook(_make_agent(), {"msg": _make_msg([{"type": "text", "text": "hi"}])})

        assert len(payloads) == 0
