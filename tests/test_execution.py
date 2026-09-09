"""Sandbox, tool bridge, stub generation, and the convergence loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from tally.execution.bridge import ToolBridge
from tally.execution.convergence import ConvergenceLoop, Escalation, parse_failure
from tally.execution.registry import ToolParam, ToolRegistry, ToolSpec
from tally.execution.sandbox import LocalSandbox, SandboxLimits, build_sandbox
from tally.execution.stubgen import params_from_json_schema, specs_from_mcp, write_package
from tally.execution.workspace import Workspace


# --- sandbox --------------------------------------------------------------
def test_sandbox_runs_code_and_persists_artifacts(tmp_path: Path):
    sb = LocalSandbox(workspace=tmp_path)
    result = sb.run("print('hi')\nopen('out.txt','w').write('kept')")
    assert result.ok
    assert result.stdout.strip() == "hi"
    assert (tmp_path / "out.txt").read_text() == "kept"


def test_sandbox_reports_its_isolation_level(tmp_path: Path):
    """A result must always carry the conditions it was produced under."""
    sb = LocalSandbox(workspace=tmp_path)
    result = sb.run("pass")
    assert result.backend == "local-subprocess"
    assert result.isolation == "process-rlimits-only"
    assert result.to_json()["isolation"] == "process-rlimits-only"


def test_local_sandbox_refuses_network_imports(tmp_path: Path):
    sb = LocalSandbox(workspace=tmp_path)
    result = sb.run("import socket")
    assert not result.ok
    assert result.exit_code == 126
    assert "socket" in result.stderr


def test_sandbox_enforces_a_wall_clock(tmp_path: Path):
    sb = LocalSandbox(workspace=tmp_path)
    result = sb.run("while True: pass", limits=SandboxLimits(wall_clock_s=2))
    assert result.timed_out and result.exit_code == 124


def test_traceback_line_numbers_match_the_agents_own_code(tmp_path: Path):
    """The bootstrap must not shift the lines the feedback loop quotes back."""
    sb = LocalSandbox(workspace=tmp_path)
    result = sb.run("a = 1\nb = 2\nc = b / 0\n")
    assert 'line 3' in result.stderr


def test_build_sandbox_falls_back_when_docker_is_absent(tmp_path: Path):
    sb = build_sandbox(workspace=tmp_path, prefer_docker=False)
    assert sb.backend == "local-subprocess"


# --- bridge ---------------------------------------------------------------
def test_bridged_tool_reaches_the_host_from_a_network_free_sandbox(tmp_path: Path):
    ws, tools = tmp_path / "ws", tmp_path / "tools"
    ws.mkdir()
    tools.mkdir()
    seen: list[str] = []

    def fetch(cik: str) -> dict:
        seen.append(cik)
        return {"cik": cik, "revenue": 1}

    registry = ToolRegistry().add(ToolSpec(
        module="sec", func="fetch", summary="fetch", params=[ToolParam("cik", "str")],
        returns="dict", requires_network=True, handler=fetch,
    ))
    write_package(tools, registry)
    bridge = ToolBridge(rpc_dir=ws / ".rpc").register_registry(registry)
    sb = LocalSandbox(workspace=ws, tools_dir=tools)

    with bridge.serving():
        result = sb.run("from tools import sec\nprint(sec.fetch(cik='0000320193'))")
    assert result.ok, result.stderr
    assert seen == ["0000320193"]
    assert bridge.stats()["calls"] == 1


def test_bridge_rejects_unregistered_tools(tmp_path: Path):
    """An unknown name is an error, never a passthrough."""
    ws, tools = tmp_path / "ws", tmp_path / "tools"
    ws.mkdir()
    tools.mkdir()
    registry = ToolRegistry().add(ToolSpec(
        module="sec", func="fetch", summary="fetch", params=[], returns="dict",
        requires_network=True, handler=lambda: {},
    ))
    write_package(tools, registry)
    bridge = ToolBridge(rpc_dir=ws / ".rpc")  # nothing registered
    sb = LocalSandbox(workspace=ws, tools_dir=tools)
    with bridge.serving():
        result = sb.run(
            "from tools import sec\n"
            "try:\n    sec.fetch()\nexcept Exception as e:\n    print('ERR', e)"
        )
    assert "unknown tool" in result.stdout


def test_bridge_surfaces_handler_exceptions_to_the_agent(tmp_path: Path):
    ws, tools = tmp_path / "ws", tmp_path / "tools"
    ws.mkdir()
    tools.mkdir()

    def boom() -> dict:
        raise LookupError("no filing for that year")

    registry = ToolRegistry().add(ToolSpec(
        module="sec", func="boom", summary="fails", params=[], returns="dict",
        requires_network=True, handler=boom,
    ))
    write_package(tools, registry)
    bridge = ToolBridge(rpc_dir=ws / ".rpc").register_registry(registry)
    sb = LocalSandbox(workspace=ws, tools_dir=tools)
    with bridge.serving():
        result = sb.run(
            "from tools import sec\n"
            "try:\n    sec.boom()\nexcept Exception as e:\n    print('CAUGHT', e)"
        )
    assert "no filing for that year" in result.stdout
    assert bridge.stats()["failures"] == 1


def test_rpc_plumbing_stays_out_of_the_agents_workspace_view(tmp_path: Path):
    ws_root = tmp_path / "ws"
    ws_root.mkdir()
    ws = Workspace(root=ws_root)
    (ws_root / ".rpc").mkdir()
    (ws_root / ".rpc" / "audit.jsonl").write_text("{}\n")
    ws.write_text("l1.json", "{}")
    assert [a.rel for a in ws.artifacts()] == ["l1.json"]
    # But the checkpoint manifest must see everything, or drift detection lies.
    assert len(ws.snapshot_manifest()["files"]) == 2


# --- stub generation / MCP ------------------------------------------------
def test_mcp_schema_becomes_an_importable_module(tmp_path: Path):
    specs = specs_from_mcp([{
        "name": "search-filings",
        "description": "Search filings.\nSupports forms.",
        "inputSchema": {"type": "object", "properties": {
            "cik": {"type": "string", "description": "CIK"},
            "form": {"type": "string"}}, "required": ["cik"]},
    }], module="sec")
    registry = ToolRegistry().extend(specs)
    write_package(tmp_path, registry)
    source = (tmp_path / "tools" / "sec.py").read_text()
    assert "def search_filings(cik: str, form: str = None)" in source
    assert "Search filings." in source


def test_required_params_precede_defaulted_ones():
    """Otherwise the generated module is a SyntaxError."""
    params = params_from_json_schema({
        "type": "object",
        "properties": {"zebra": {"type": "string"}, "alpha": {"type": "string"}},
        "required": ["zebra"],
    })
    assert params[0].name == "zebra" and params[0].default is None
    assert params[1].name == "alpha" and params[1].default == "None"


def test_mcp_names_are_sanitised_into_python_identifiers():
    specs = specs_from_mcp(
        [{"name": "a.weird-tool/name", "inputSchema": {}}], module="x"
    )
    assert specs[0].func == "name"


def test_signature_block_is_far_cheaper_than_json_schemas():
    """The claim the tools slot rests on, measured."""
    from tally.context.tokenizer import count_tokens

    registry = ToolRegistry().extend([
        ToolSpec(module="m", func=f"tool_{i}",
                 summary=f"Does thing number {i} with several inputs",
                 params=[ToolParam("alpha", "str", doc="the alpha input"),
                         ToolParam("beta", "int", doc="the beta input")],
                 returns="dict",
                 detail="Extended description that would appear in a JSON schema.")
        for i in range(30)
    ])
    costs = registry.token_cost_comparison(count_tokens)
    assert costs["signature_lines"] < costs["full_json_schemas"] / 2


# --- convergence ----------------------------------------------------------
def test_the_same_mistake_twice_escalates_and_thrice_asks_a_human(tmp_path: Path):
    registry = ToolRegistry().add(ToolSpec(
        module="sec", func="fetch_filing", summary="fetch",
        params=[ToolParam("cik", "str")], returns="dict", local_source="return {}",
    ))
    write_package(tmp_path, registry)
    sb = LocalSandbox(workspace=tmp_path / "ws", tools_dir=tmp_path)
    loop = ConvergenceLoop(registry=registry)
    code = "from tools import sec\nprint(sec.fetch())"

    first = loop.observe(sb.run(code), code=code)
    second = loop.observe(sb.run(code), code=code)
    third = loop.observe(sb.run(code), code=code)
    assert first.escalation is Escalation.NONE
    assert second.escalation is Escalation.MODEL
    assert third.escalation is Escalation.HITL


def test_different_errors_in_a_row_are_progress_not_a_loop(tmp_path: Path):
    sb = LocalSandbox(workspace=tmp_path)
    loop = ConvergenceLoop()
    a = loop.observe(sb.run("print(undefined_name)"), code="print(undefined_name)")
    b = loop.observe(sb.run("x = 1/0"), code="x = 1/0")
    assert a.escalation is Escalation.NONE and b.escalation is Escalation.NONE
    assert a.failure.signature != b.failure.signature


def test_failure_signature_ignores_the_message_text(tmp_path: Path):
    """`name 'df' is not defined` and `name 'tbl' ...` are one mistake repeated."""
    sb = LocalSandbox(workspace=tmp_path)
    loop = ConvergenceLoop()
    loop.observe(sb.run("print(df)"), code="print(df)")
    second = loop.observe(sb.run("print(tbl)"), code="print(tbl)")
    assert second.repeat_count == 2
    assert second.escalation is Escalation.MODEL


def test_feedback_reinjects_the_real_tool_surface(tmp_path: Path):
    registry = ToolRegistry().extend([
        ToolSpec(module="sec", func="fetch_filing", summary="Fetch a filing",
                 params=[ToolParam("cik", "str")], returns="dict", local_source="return {}"),
        ToolSpec(module="sec", func="extract_tables", summary="Extract tables",
                 params=[ToolParam("path", "str")], returns="list", local_source="return []"),
    ])
    write_package(tmp_path, registry)
    sb = LocalSandbox(workspace=tmp_path / "ws", tools_dir=tmp_path)
    loop = ConvergenceLoop(registry=registry)
    code = "from tools import sec\nsec.fetch()"
    feedback = loop.observe(sb.run(code), code=code)
    assert "fetch_filing" in feedback.text and "extract_tables" in feedback.text
    assert ">>" in feedback.text, "the failing line should be quoted back"


@pytest.mark.parametrize(
    ("code", "kind"),
    [
        ("print(nope)", "NameError"),
        ("x = 1/0", "ZeroDivisionError"),
        ("import nonexistent_pkg_xyz", "ModuleNotFoundError"),
        ("{}['missing']", "KeyError"),
    ],
)
def test_failure_classification(tmp_path: Path, code: str, kind: str):
    sb = LocalSandbox(workspace=tmp_path)
    failure = parse_failure(sb.run(code))
    assert failure is not None and failure.kind == kind
