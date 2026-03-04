"""
Integration/unit tests: orchestrator task decomposition with stateful workers.

Two orthogonal concerns:

A) Race-condition demonstrations (pure threading, no agentscope required)
   Show WHAT goes wrong when the orchestrator splits stateful steps across
   parallel workers — the scenario the "Stateful Task Grouping" prompt rule
   is designed to prevent.

B) TaskManager mechanics (agentscope required, skipped when absent)
   Verify create_task metadata, error paths, and that spawn_batch_and_wait
   actually launches tasks concurrently while spawn_and_wait is sequential.

Run all:
    cd TestAgent && pytest tests/test_orchestrator_stateful.py -v

Skip network tests only:
    pytest tests/test_orchestrator_stateful.py -v -m "not network"
"""

import asyncio
import json
import threading
import time
from typing import Dict, List
from unittest.mock import MagicMock

import pytest


# ── Shared stateful session simulation ─────────────────────────────────────


class _SharedSession:
    """
    Simulates process-level mutable state inside an MCP server subprocess
    (analogous to http_client_mcp._session / _config / _history).

    All workers in a conversation share this instance because they share
    the same toolkit → the same MCP client → the same subprocess.
    Thread-safe internally via a Lock.
    """

    def __init__(self) -> None:
        self.base_url: str = ""
        self.cookies: Dict[str, str] = {}
        self._lock = threading.Lock()

    def configure(self, base_url: str) -> None:
        with self._lock:
            self.base_url = base_url

    def set_cookie(self, key: str, value: str) -> None:
        with self._lock:
            self.cookies[key] = value

    def get_base_url(self) -> str:
        with self._lock:
            return self.base_url

    def get_cookies(self) -> Dict[str, str]:
        with self._lock:
            return dict(self.cookies)

    def reset(self) -> None:
        with self._lock:
            self.base_url = ""
            self.cookies = {}


# ── A: Race-condition demonstrations ───────────────────────────────────────


class TestRaceCondition:
    """
    Deterministic race-condition tests using threading.Barrier.

    No agentscope or LLM required — demonstrates the infrastructure-level
    problem that occurs when stateful steps are split across parallel workers.

    Each test mirrors an INCORRECT orchestrator task decomposition that the
    "Stateful Task Grouping" prompt rule explicitly forbids.
    """

    def test_parallel_reset_clears_sibling_cookie(self):
        """
        Deterministic race: Worker B's reset() executes between Worker A's
        set_cookie() and A's subsequent read — destroying A's state.

        Incorrect orchestrator decomposition (anti-pattern):
            task_A → executor: "set cookie user=alice"      ← stateful step 1
            task_B → executor: "reset the session"          ← concurrent

        Guaranteed interleaving via two barriers:
          ① A  : set_cookie("user", "alice")
          ② b1 : both threads synchronize
          ③ B  : reset()                   ← runs after A's set_cookie
          ④ b2 : both threads synchronize
          ⑤ A  : get_cookies() → {}        ← BUG: expected {"user": "alice"}
        """
        session = _SharedSession()
        b1 = threading.Barrier(2)  # A has set cookie; B may proceed
        b2 = threading.Barrier(2)  # B has reset; A may read
        results: Dict[str, dict] = {}

        def worker_a() -> None:
            session.set_cookie("user", "alice")
            b1.wait()  # let B proceed to reset
            b2.wait()  # wait until B's reset is done
            results["a"] = session.get_cookies()

        def worker_b() -> None:
            b1.wait()        # wait until A has set cookie
            session.reset()  # guaranteed to run after A's set_cookie
            b2.wait()
            results["b"] = session.get_cookies()

        threads = [threading.Thread(target=f) for f in (worker_a, worker_b)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results["a"] == {}, (
            "Race condition confirmed: Worker A's cookie was wiped by Worker B's "
            f"concurrent reset. Expected {{}}, got {results['a']}.\n"
            "Fix: assign both steps to ONE task in a single spawn_and_wait call."
        )

    def test_parallel_configure_last_writer_wins(self):
        """
        Deterministic race: two workers configure() the shared session with
        different base_urls — last writer wins, first writer's config is lost.

        Incorrect orchestrator decomposition (anti-pattern):
            task_A → executor: configure("https://api-a.example.com") + request
            task_B → executor: configure("https://api-b.example.com") + request
            → both dispatched via spawn_batch_and_wait

        Guaranteed interleaving:
          ① A  : configure("https://api-a.example.com")
          ② b1 : both threads synchronize
          ③ B  : configure("https://api-b.example.com")  ← overwrites A's config
          ④ b2 : both threads synchronize
          ⑤ A  : get_base_url() → "api-b"               ← BUG: A's request now
                                                            goes to the wrong URL
        """
        session = _SharedSession()
        b1 = threading.Barrier(2)
        b2 = threading.Barrier(2)
        observed: Dict[str, str] = {}

        def worker_a() -> None:
            session.configure("https://api-a.example.com")
            b1.wait()  # let B overwrite
            b2.wait()  # wait until B's configure is done
            observed["a"] = session.get_base_url()

        def worker_b() -> None:
            b1.wait()  # wait until A has configured
            session.configure("https://api-b.example.com")  # overwrites A
            b2.wait()
            observed["b"] = session.get_base_url()

        threads = [threading.Thread(target=f) for f in (worker_a, worker_b)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert observed["a"] == "https://api-b.example.com", (
            "Last-writer-wins confirmed: A's configure was overwritten by B.\n"
            f"A saw {observed['a']!r}, expected 'https://api-b.example.com'.\n"
            "Fix: each scenario's configure+request must be one atomic task."
        )
        assert observed["a"] == observed["b"], (
            "Both workers should see the same (last-written) base_url."
        )

    def test_single_task_sequential_steps_are_always_correct(self):
        """
        Correct pattern: ALL stateful steps for one scenario are grouped into
        a SINGLE worker task and executed sequentially — no interleaving possible.

        This is what the prompt rule enforces:
            task_A → executor: "configure + set_cookie + read_cookies"
                                (all steps in one task, one spawn_and_wait)
        """
        session = _SharedSession()

        # Simulates ONE worker task handling the entire stateful workflow
        session.configure("https://api.example.com")
        session.set_cookie("session_id", "abc123")

        assert session.get_base_url() == "https://api.example.com"
        assert session.get_cookies() == {"session_id": "abc123"}


# ── B: TaskManager mechanics ───────────────────────────────────────────────


def _make_task_manager(workers: dict = None):
    """Create a minimal TaskManager. Requires agentscope."""
    pytest.importorskip("agentscope", reason="agentscope not installed")
    from orchestrator.task_manager import TaskManager

    return TaskManager(
        progress_callback=None,
        workers=workers or {},
        toolkit=MagicMock(),
        model=MagicMock(),
    )


class TestTaskManagerCreation:
    """Pure unit tests for task registration — no worker execution needed."""

    def test_create_task_assigns_sequential_ids(self):
        tm = _make_task_manager()
        id1 = tm.create_task("Task one", "worker_a", {})
        id2 = tm.create_task("Task two", "worker_b", {})
        assert id1 == "task_001"
        assert id2 == "task_002"

    def test_create_task_stores_metadata(self):
        tm = _make_task_manager()
        task_id = tm.create_task(
            description="Configure HTTP session and verify cookies",
            worker_name="executor",
            input_data={"base_url": "https://api.example.com"},
        )
        node = tm.get_all_tasks()[task_id]
        assert node.description == "Configure HTTP session and verify cookies"
        assert node.worker_name == "executor"
        assert node.input_data["base_url"] == "https://api.example.com"
        assert node.status == "pending"
        assert node.depth == 0

    def test_create_child_task_increments_depth(self):
        tm = _make_task_manager()
        parent_id = tm.create_task("Parent task", "executor", {})
        child_id = tm.create_task("Child task", "executor", {}, parent_id=parent_id)
        tasks = tm.get_all_tasks()
        assert tasks[parent_id].depth == 0
        assert tasks[child_id].depth == 1

    def test_list_tasks_shows_pending_icon(self):
        tm = _make_task_manager()
        tm.create_task("Configure session", "executor", {})
        tm.create_task("Verify cookies", "executor", {})
        listing = tm.list_tasks()
        assert "task_001" in listing
        assert "task_002" in listing
        assert "○" in listing  # pending icon for unstarted tasks

    def test_progress_callback_fires_on_create(self):
        pytest.importorskip("agentscope", reason="agentscope not installed")
        from orchestrator.task_manager import TaskManager

        events: List[str] = []

        def callback(event_type: str, data: dict) -> None:
            events.append(event_type)

        tm = TaskManager(
            progress_callback=callback,
            workers={},
            toolkit=MagicMock(),
            model=MagicMock(),
        )
        tm.create_task("Test task", "executor", {})
        assert "task_tree_node_created" in events


class TestTaskManagerErrorPaths:
    """Error handling in spawn_and_wait / spawn_batch_and_wait."""

    async def test_spawn_unknown_task_id_returns_error(self):
        tm = _make_task_manager()
        result = await tm.spawn_and_wait("task_999")
        assert result.startswith("ERROR:")
        assert "task_999" in result

    async def test_spawn_unknown_worker_marks_task_failed(self):
        tm = _make_task_manager()
        task_id = tm.create_task("Some task", "nonexistent_worker", {})
        result = await tm.spawn_and_wait(task_id)
        assert result.startswith("ERROR:")
        assert "nonexistent_worker" in result
        assert tm.get_all_tasks()[task_id].status == "failed"

    async def test_spawn_batch_empty_list_returns_zero_counts(self):
        tm = _make_task_manager()
        result = json.loads(await tm.spawn_batch_and_wait([]))
        assert result["total"] == 0
        assert result["succeeded"] == 0
        assert result["failed"] == 0

    async def test_spawn_batch_missing_task_ids_all_fail(self):
        tm = _make_task_manager()
        result = json.loads(await tm.spawn_batch_and_wait(["task_999", "task_998"]))
        assert result["total"] == 2
        assert result["failed"] == 2
        assert result["succeeded"] == 0

    async def test_max_depth_exceeded_returns_error(self):
        pytest.importorskip("agentscope", reason="agentscope not installed")
        from orchestrator.task_manager import TaskManager

        tm = _make_task_manager()
        task_id = tm.create_task("Deep task", "executor", {})
        # Force the task to max depth to trigger the guard
        tm.get_all_tasks()[task_id].depth = TaskManager.MAX_DEPTH
        result = await tm.spawn_and_wait(task_id)
        assert result.startswith("ERROR:")
        assert "max nesting depth" in result
        assert tm.get_all_tasks()[task_id].status == "failed"


# ── C: Concurrency behavior via monkeypatched spawn_and_wait ───────────────


class TestSpawnBatchConcurrency:
    """
    Verifies spawn_batch_and_wait runs tasks concurrently via asyncio.gather
    and that this concurrency is the root cause of race conditions with
    stateful tasks.

    Uses monkeypatching (tm.spawn_and_wait = fake_fn) to bypass real worker
    execution while keeping TaskManager's async scheduling logic intact.
    """

    async def test_spawn_batch_runs_tasks_concurrently(self):
        """
        Two tasks each sleep 0.15 s via asyncio.sleep.
          - Sequential (spawn_and_wait × 2): wall time ≥ 0.30 s
          - Parallel  (spawn_batch_and_wait via asyncio.gather): wall time ≈ 0.15 s

        Asserts elapsed < 0.25 s to confirm concurrent execution.
        """
        tm = _make_task_manager(workers={"executor": MagicMock()})
        task_a = tm.create_task("task_sleep_a", "executor", {})
        task_b = tm.create_task("task_sleep_b", "executor", {})

        async def fake_spawn(task_id: str, timeout=None) -> str:
            await asyncio.sleep(0.15)
            return "done"

        tm.spawn_and_wait = fake_spawn  # monkeypatch instance method

        start = time.monotonic()
        result_json = await tm.spawn_batch_and_wait([task_a, task_b], timeout=10)
        elapsed = time.monotonic() - start

        result = json.loads(result_json)
        assert result["total"] == 2
        assert result["succeeded"] == 2, f"Expected 2 succeeded, got: {result}"
        assert elapsed < 0.25, (
            f"spawn_batch_and_wait must run tasks concurrently (≈ 0.15 s), "
            f"but took {elapsed:.2f} s — suggests sequential execution."
        )

    async def test_spawn_and_wait_calls_are_strictly_sequential(self):
        """
        spawn_and_wait must execute one task at a time — no overlap.

        This is the CORRECT pattern for stateful task sequences: each step
        completes fully before the next one begins, so no interleaving can
        corrupt shared state.
        """
        tm = _make_task_manager(workers={"executor": MagicMock()})
        task_a = tm.create_task("configure_session", "executor", {})
        task_b = tm.create_task("send_request", "executor", {})
        task_c = tm.create_task("verify_response", "executor", {})

        call_log: List[tuple] = []

        async def fake_spawn(task_id: str, timeout=None) -> str:
            node = tm.get_all_tasks()[task_id]
            call_log.append(("start", node.description))
            await asyncio.sleep(0.02)  # simulate work
            call_log.append(("end", node.description))
            return f"done: {node.description}"

        tm.spawn_and_wait = fake_spawn

        await tm.spawn_and_wait(task_a)
        await tm.spawn_and_wait(task_b)
        await tm.spawn_and_wait(task_c)

        assert call_log == [
            ("start", "configure_session"), ("end", "configure_session"),
            ("start", "send_request"),      ("end", "send_request"),
            ("start", "verify_response"),   ("end", "verify_response"),
        ], (
            "Sequential spawn_and_wait must complete each task before starting "
            f"the next. Got execution log: {call_log}"
        )

    async def test_spawn_batch_race_condition_with_shared_state(self):
        """
        REGRESSION: spawn_batch_and_wait causes deterministic state corruption
        when workers share mutable state.

        Simulates incorrect orchestrator decomposition:
            task_A (executor): set cookie "user=alice", then read cookies
            task_B (executor): reset session                          ← concurrent

        Guaranteed interleaving via asyncio events (mirrors TestRaceCondition
        tests but exercised *through* TaskManager.spawn_batch_and_wait):
          ① task_A: set_cookie("user", "alice"), signal e1
          ② task_B: wait e1, reset(), signal e2
          ③ task_A: wait e2, get_cookies() → {}        ← BUG confirmed
        """
        session = _SharedSession()
        e1 = asyncio.Event()  # A has set cookie; B may proceed
        e2 = asyncio.Event()  # B has reset; A may read
        observed: Dict[str, dict] = {}

        tm = _make_task_manager(workers={"executor": MagicMock()})
        task_a = tm.create_task("set_cookie_and_read", "executor", {})
        task_b = tm.create_task("reset_and_read", "executor", {})

        async def fake_spawn(task_id: str, timeout=None) -> str:
            node = tm.get_all_tasks()[task_id]
            desc = node.description

            if desc == "set_cookie_and_read":
                session.set_cookie("user", "alice")
                e1.set()            # let B proceed to reset
                await e2.wait()     # wait until B's reset() is done
                observed["a"] = session.get_cookies()
                return json.dumps(observed["a"])

            elif desc == "reset_and_read":
                await e1.wait()     # wait until A has set the cookie
                session.reset()     # guaranteed to run after A's set_cookie
                e2.set()
                observed["b"] = session.get_cookies()
                return json.dumps(observed["b"])

            return "unknown"

        tm.spawn_and_wait = fake_spawn

        result_json = await tm.spawn_batch_and_wait([task_a, task_b], timeout=10)
        result = json.loads(result_json)

        assert result["succeeded"] == 2, f"Both fake tasks should succeed: {result}"
        assert observed["a"] == {}, (
            "Race condition via spawn_batch_and_wait confirmed: task_A's cookie "
            "was wiped by task_B's concurrent reset.\n"
            f"Expected {{}}, got {observed['a']}.\n"
            "Fix: group both steps into ONE task dispatched with spawn_and_wait."
        )


# ── D: _filter_toolkit correctness ──────────────────────────────────────────


class TestFilterToolkit:
    """
    Verifies WorkerRunner._filter_toolkit creates a shallow copy of the
    Toolkit with only allowed tools, activates needed groups, and does not
    mutate the original toolkit.
    """

    def test_filter_keeps_only_allowed_tools(self):
        pytest.importorskip("agentscope", reason="agentscope not installed")
        from agentscope.tool import Toolkit, ToolResponse
        from worker.worker_runner import WorkerRunner
        from worker.worker_loader import WorkerConfig

        tk = Toolkit()

        def tool_a(x: str) -> ToolResponse:
            """Tool A"""
            return ToolResponse(content="a")

        def tool_b(x: str) -> ToolResponse:
            """Tool B"""
            return ToolResponse(content="b")

        def tool_c(x: str) -> ToolResponse:
            """Tool C"""
            return ToolResponse(content="c")

        tk.register_tool_function(tool_a)
        tk.register_tool_function(tool_b)
        tk.register_tool_function(tool_c)

        config = WorkerConfig.from_dict({
            "name": "test_worker",
            "tools": ["tool_a", "tool_c"],
            "mode": "react",
        })
        runner = WorkerRunner(config=config, model=MagicMock(), toolkit=tk)

        filtered = runner._filter_toolkit(tk)
        filtered_names = set(filtered.tools.keys())

        assert filtered_names == {"tool_a", "tool_c"}, (
            f"Expected only tool_a and tool_c, got {filtered_names}"
        )

    def test_filter_returns_full_toolkit_when_empty(self):
        pytest.importorskip("agentscope", reason="agentscope not installed")
        from agentscope.tool import Toolkit, ToolResponse
        from worker.worker_runner import WorkerRunner
        from worker.worker_loader import WorkerConfig

        tk = Toolkit()

        def tool_x(x: str) -> ToolResponse:
            """Tool X"""
            return ToolResponse(content="x")

        tk.register_tool_function(tool_x)

        config = WorkerConfig.from_dict({
            "name": "test_worker",
            "tools": [],
            "mode": "react",
        })
        runner = WorkerRunner(config=config, model=MagicMock(), toolkit=tk)

        result = runner._filter_toolkit(tk)
        assert result is tk, "When no tools specified, should return original toolkit"

    def test_filter_ignores_tools_not_in_toolkit(self):
        pytest.importorskip("agentscope", reason="agentscope not installed")
        from agentscope.tool import Toolkit, ToolResponse
        from worker.worker_runner import WorkerRunner
        from worker.worker_loader import WorkerConfig

        tk = Toolkit()

        def existing_tool(x: str) -> ToolResponse:
            """Existing"""
            return ToolResponse(content="ok")

        tk.register_tool_function(existing_tool)

        config = WorkerConfig.from_dict({
            "name": "test_worker",
            "tools": ["existing_tool", "nonexistent_tool"],
            "mode": "react",
        })
        runner = WorkerRunner(config=config, model=MagicMock(), toolkit=tk)

        filtered = runner._filter_toolkit(tk)
        assert set(filtered.tools.keys()) == {"existing_tool"}

    def test_filter_preserves_mcp_like_tools(self):
        """
        MCP tools share the same original_func (__call__ bound method).
        The old approach (register_tool_function) fails because names collide.
        The shallow-copy approach preserves the original ToolFunc objects.
        """
        pytest.importorskip("agentscope", reason="agentscope not installed")
        from agentscope.tool import Toolkit, ToolResponse
        from agentscope.tool._types import RegisteredToolFunction
        from worker.worker_runner import WorkerRunner
        from worker.worker_loader import WorkerConfig

        tk = Toolkit()

        # Register a normal tool first
        def normal_tool(x: str) -> ToolResponse:
            """Normal tool"""
            return ToolResponse(content="ok")

        tk.register_tool_function(normal_tool)

        # Simulate MCP tools: multiple RegisteredToolFunction entries sharing
        # the same original_func (a bound __call__ method)
        class FakeMCPClient:
            def __call__(self, **kwargs):
                pass

        client = FakeMCPClient()

        # Create tool group for MCP tools
        tk.create_tool_group("http_client_tools", "HTTP client tools", "")

        for tool_name in ("http_configure", "http_request", "http_get_cookies"):
            tf = RegisteredToolFunction(
                name=tool_name,
                group="http_client_tools",
                source="mcp_server",
                original_func=client.__call__,  # same bound method!
                json_schema={"type": "function", "function": {"name": tool_name, "parameters": {"type": "object", "properties": {}}}},
            )
            tk.tools[tool_name] = tf

        config = WorkerConfig.from_dict({
            "name": "test_worker",
            "tools": ["http_configure", "http_request"],
            "mode": "react",
        })
        runner = WorkerRunner(config=config, model=MagicMock(), toolkit=tk)

        filtered = runner._filter_toolkit(tk)
        assert set(filtered.tools.keys()) == {"http_configure", "http_request"}
        # Verify the ToolFunc objects are the exact same references (not re-registered)
        assert filtered.tools["http_configure"] is tk.tools["http_configure"]
        assert filtered.tools["http_request"] is tk.tools["http_request"]

    def test_filter_activates_needed_groups(self):
        """Filtered toolkit should activate groups that contain retained tools."""
        pytest.importorskip("agentscope", reason="agentscope not installed")
        from agentscope.tool import Toolkit, ToolResponse
        from agentscope.tool._types import RegisteredToolFunction
        from worker.worker_runner import WorkerRunner
        from worker.worker_loader import WorkerConfig

        tk = Toolkit()

        # Create two groups, both inactive by default
        tk.create_tool_group("group_a", "Group A tools", "")
        tk.create_tool_group("group_b", "Group B tools", "")
        assert not tk.groups["group_a"].active
        assert not tk.groups["group_b"].active

        # Add tools to each group
        for name, group in [("ta1", "group_a"), ("ta2", "group_a"), ("tb1", "group_b")]:
            tf = RegisteredToolFunction(
                name=name,
                group=group,
                source="function_group",
                original_func=lambda: None,
                json_schema={"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}},
            )
            tk.tools[name] = tf

        # Filter to only group_a tools
        config = WorkerConfig.from_dict({
            "name": "test_worker",
            "tools": ["ta1", "ta2"],
            "mode": "react",
        })
        runner = WorkerRunner(config=config, model=MagicMock(), toolkit=tk)
        filtered = runner._filter_toolkit(tk)

        assert filtered.groups["group_a"].active, "group_a should be activated"
        assert not filtered.groups["group_b"].active, "group_b should remain inactive"

    def test_filter_does_not_mutate_original(self):
        """Shallow copy must not affect the original toolkit's groups state."""
        pytest.importorskip("agentscope", reason="agentscope not installed")
        from agentscope.tool import Toolkit, ToolResponse
        from agentscope.tool._types import RegisteredToolFunction
        from worker.worker_runner import WorkerRunner
        from worker.worker_loader import WorkerConfig

        tk = Toolkit()
        tk.create_tool_group("my_group", "My tools", "")

        tf = RegisteredToolFunction(
            name="my_tool",
            group="my_group",
            source="function_group",
            original_func=lambda: None,
            json_schema={"type": "function", "function": {"name": "my_tool", "parameters": {"type": "object", "properties": {}}}},
        )
        tk.tools["my_tool"] = tf

        assert not tk.groups["my_group"].active, "Precondition: group inactive"

        config = WorkerConfig.from_dict({
            "name": "test_worker",
            "tools": ["my_tool"],
            "mode": "react",
        })
        runner = WorkerRunner(config=config, model=MagicMock(), toolkit=tk)
        filtered = runner._filter_toolkit(tk)

        # filtered should have group active
        assert filtered.groups["my_group"].active
        # original must remain untouched
        assert not tk.groups["my_group"].active, "Original toolkit groups must not be mutated"
        # original tools dict must be untouched
        assert "my_tool" in tk.tools


# ── E: TaskManager worker failure propagation ───────────────────────────────


class TestTaskManagerFailurePropagation:
    """
    Verifies TaskManager.spawn_and_wait correctly propagates worker failures
    instead of silently returning empty strings.
    """

    async def test_worker_failure_returns_error_string(self):
        """When worker fails, spawn_and_wait should return ERROR: message"""
        pytest.importorskip("agentscope", reason="agentscope not installed")
        from orchestrator.task_manager import TaskManager

        # Create TaskManager with a real worker entry so we pass the
        # "unknown worker" check and reach the _run_worker coroutine
        tm = _make_task_manager(workers={"executor": MagicMock()})
        task_id = tm.create_task("failing_task", "executor", {})

        # Monkeypatch _run_worker indirectly: replace spawn_and_wait's
        # internal coroutine by patching the instance method to simulate
        # the except branch catching a RuntimeError
        original_spawn = tm.spawn_and_wait

        async def spawn_that_exercises_except(tid, timeout=None):
            """Call real spawn_and_wait but inject a failure in _run_worker"""
            # The real spawn_and_wait will try to import WorkerRunner which
            # may not exist in test env — the except branch will catch it
            return await original_spawn(tid, timeout)

        # Just call the real spawn_and_wait — since WorkerRunner import will
        # fail in the test environment, it should hit the except branch
        result = await tm.spawn_and_wait(task_id)

        assert result.startswith("ERROR:"), f"Expected ERROR: prefix, got: {result}"
        assert tm.get_all_tasks()[task_id].status == "failed"
