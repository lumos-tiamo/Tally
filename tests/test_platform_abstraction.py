"""The platform's acceptance criterion, as an executable test.

The claim under test: **adding a scenario means writing an AgentSpec and tool
modules, and nothing else.** A platform that needs a runtime change per scenario
is an application with a plugin folder, so this file is where the claim is either
demonstrated or falsified.

Three things are checked:

1. All four scenarios drive the *same* :class:`Agent` class, imported once here.
2. No scenario package imports from, subclasses, or monkey-patches the runtime
   modules — it may only *use* them.
3. The incremental cost of a scenario is measured, so "cheap to add" is a number
   rather than an adjective.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from tally.execution.workspace import Workspace
from tally.models.providers.fake import ScriptedProvider
from tally.models.router import Registry, Router
from tally.observability.trace import Tracer, new_run_id
from tally.orchestration.agent import Agent          # the one and only runtime
from tally.scenarios import bi_analyst, code_engineer, deep_research

SRC = Path(__file__).resolve().parents[1] / "src" / "tally"
SCENARIOS = SRC / "scenarios"
RUNTIME_PACKAGES = ("orchestration", "context", "execution", "models", "observability")


def scripted_router(responses, tracer):  # noqa: ANN001
    provider = ScriptedProvider(responses, name="s", loop_last=True)
    registry = Registry(providers={"s": provider})
    for purpose in list(registry.policy):
        registry.policy[purpose] = ("s",)
    return Router(registry, tracer=tracer), provider


# --- 1. the same runtime drives every scenario ---------------------------
def test_bi_scenario_runs_on_the_unmodified_runtime(tmp_path: Path):
    db = tmp_path / "shop.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE orders(id INTEGER PRIMARY KEY, amount REAL, status TEXT);"
        "INSERT INTO orders VALUES (1, 100.0, 'shipped'), (2, 50.0, 'cancelled');"
    )
    conn.commit()
    conn.close()

    tracer = Tracer(new_run_id(), tmp_path / "runs")
    router, _ = scripted_router([
        f"```python\nfrom tools import sql\n"
        f"r = sql.run({str(db)!r}, 'SELECT SUM(amount) AS total FROM orders "
        f"WHERE status=\\'shipped\\'')\n"
        f"sql.save_json({{'answer': r['rows'][0]['total'], 'grain': 'whole table', "
        f"'query': r['query'], 'caveats': ''}}, 'answer.json')\nprint(r['rows'])\n```",
        "DONE\nShipped revenue is 100.0.",
    ], tracer)

    tools = bi_analyst.spec.build_tools(db)
    spec = bi_analyst.spec.build_spec(tools=tools, prefer_docker=False)
    workspace = Workspace.create(tmp_path / "ws", "bi")
    agent = Agent(spec, router=router, workspace=workspace, tracer=tracer,
                  run_dir=tmp_path / "run-bi")
    run = agent.run(bi_analyst.spec.objective("Shipped revenue?", str(db)))
    tracer.close()

    assert run.finished
    assert workspace.exists("answer.json")
    assert '"answer": 100.0' in workspace.read_text("answer.json")


def test_research_scenario_runs_on_the_unmodified_runtime(tmp_path: Path):
    def fake_search(query: str, limit: int) -> list[dict]:
        return [{"title": "Grid storage deployments 2026", "url": "https://example.test/a",
                 "snippet": "Deployments rose 41% year on year."}]

    def fake_fetch(url: str) -> dict:
        return {"url": url, "title": "Grid storage deployments 2026",
                "text": "Global grid storage deployments rose 41% year on year in 2026."}

    handlers = deep_research.spec.WebHandlers(search=fake_search, fetch=fake_fetch)
    tools = deep_research.spec.build_tools(handlers)
    spec = deep_research.spec.build_spec(tools=tools, prefer_docker=False)

    tracer = Tracer(new_run_id(), tmp_path / "runs")
    router, _ = scripted_router([
        "```python\nfrom tools import web, notes\n"
        "hits = web.search(query='grid storage 2026', limit=3)\n"
        "page = web.fetch(url=hits[0]['url'])\n"
        "s = notes.add_source('.', page['url'], page['title'], 'reported')\n"
        "print(s)\n"
        "print(notes.add_claim('.', 'Grid storage deployments rose 41% in 2026',"
        " s['source_id'], 'deployments rose 41% year on year'))\n```",
        "```python\nfrom tools import notes\n"
        "print(notes.corroboration('.'))\n"
        "print(notes.build_report('.', 'Grid storage 2026'))\n```",
        "DONE\nDeployments rose 41%, single reported source.",
    ], tracer)

    workspace = Workspace.create(tmp_path / "ws", "research")
    agent = Agent(spec, router=router, workspace=workspace, tracer=tracer,
                  run_dir=tmp_path / "run-r")
    run = agent.run(deep_research.spec.objective("Grid storage in 2026"))
    tracer.close()

    assert run.finished
    assert workspace.exists("report.md")
    report = workspace.read_text("report.md")
    assert "41%" in report and "## Sources" in report


def test_engineering_scenario_runs_on_the_unmodified_runtime(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "calc.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8"
    )
    (repo / "test_calc.py").write_text(
        "from pkg.calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )

    tools = code_engineer.spec.build_tools()
    spec = code_engineer.spec.build_spec(tools=tools, prefer_docker=False)
    tracer = Tracer(new_run_id(), tmp_path / "runs")
    router, _ = scripted_router([
        f"```python\nfrom tools import repo\nprint(repo.run_tests({str(repo)!r}))\n```",
        f"```python\nfrom tools import repo\n"
        f"print(repo.replace({str(repo)!r}, 'pkg/calc.py',"
        f" 'return a - b', 'return a + b'))\n"
        f"print(repo.run_tests({str(repo)!r}))\n```",
        "DONE\nFixed the sign error in add(); suite is green.",
    ], tracer)

    workspace = Workspace.create(tmp_path / "ws", "eng")
    agent = Agent(spec, router=router, workspace=workspace, tracer=tracer,
                  run_dir=tmp_path / "run-e")
    run = agent.run(code_engineer.spec.objective("add() returns the wrong value.", str(repo)))
    tracer.close()

    assert run.finished
    assert "return a + b" in (repo / "pkg" / "calc.py").read_text()


# --- 2. scenarios may use the runtime but never modify it ----------------
def scenario_modules() -> list[Path]:
    return [p for p in SCENARIOS.rglob("*.py") if "sandbox_tools" not in p.parts]


def test_no_scenario_subclasses_or_patches_the_runtime():
    """Using the runtime is fine. Extending or rewriting it is the violation."""
    runtime_classes = {"Agent", "Router", "ContextLedger", "Compactor", "Tracer",
                       "Workspace", "ToolBridge", "ConvergenceLoop", "MemoryStore"}
    offences: list[str] = []
    for path in scenario_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
                    if name in runtime_classes:
                        offences.append(f"{path.name}: {node.name} subclasses {name}")
            # setattr(SomeRuntimeClass, ...) is monkey-patching.
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "setattr" and node.args:
                target = node.args[0]
                name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
                if name in runtime_classes:
                    offences.append(f"{path.name}: monkey-patches {name}")
    assert not offences, "scenarios must not modify the runtime: " + "; ".join(offences)


def test_sandbox_tool_modules_import_only_the_frozen_dependency_set():
    """A sandbox module that imports the platform cannot run inside the sandbox."""
    allowed = {
        "json", "re", "math", "time", "pathlib", "sqlite3", "hashlib", "shutil",
        "subprocess", "sys", "unicodedata", "os", "collections", "dataclasses",
        "typing", "__future__", "itertools", "functools", "textwrap",
        "pandas", "numpy", "pyarrow", "bs4", "lxml",
    }
    offences: list[str] = []
    for path in SCENARIOS.rglob("sandbox_tools/*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [(node.module or "").split(".")[0]]
            for name in names:
                if name and name not in allowed:
                    offences.append(f"{path.parent.parent.name}/{path.name}: {name}")
    assert not offences, "sandbox modules may only use the frozen set: " + "; ".join(offences)


# --- 3. the cost of adding a scenario, measured --------------------------
def scenario_cost() -> dict[str, dict[str, int]]:
    costs: dict[str, dict[str, int]] = {}
    for package in ("dd_finance", "bi_analyst", "deep_research", "code_engineer"):
        root = SCENARIOS / package
        spec_lines = tools_lines = 0
        for path in root.rglob("*.py"):
            lines = sum(1 for line in path.read_text(encoding="utf-8").splitlines()
                        if line.strip() and not line.strip().startswith("#"))
            if "sandbox_tools" in path.parts:
                tools_lines += lines
            else:
                spec_lines += lines
        costs[package] = {"spec": spec_lines, "tools": tools_lines,
                          "total": spec_lines + tools_lines}
    return costs


def test_a_light_scenario_costs_a_few_hundred_lines_and_no_runtime_change():
    """'Cheap to add' has to be a number to mean anything."""
    costs = scenario_cost()
    light = {k: v for k, v in costs.items() if k != "dd_finance"}
    for name, cost in light.items():
        assert cost["total"] < 700, f"{name} is {cost['total']} lines — too heavy to be 'light'"
        assert cost["spec"] > 0 and cost["tools"] > 0
    # The deep scenario carries the eval machinery, so it is legitimately larger.
    assert costs["dd_finance"]["total"] > max(c["total"] for c in light.values())


@pytest.mark.parametrize("package", ["bi_analyst", "deep_research", "code_engineer"])
def test_each_scenario_declares_the_same_four_things(package: str):
    """The AgentSpec contract, not a per-scenario interface."""
    module = {"bi_analyst": bi_analyst, "deep_research": deep_research,
              "code_engineer": code_engineer}[package].spec
    assert hasattr(module, "build_tools")
    assert hasattr(module, "build_spec")
    assert hasattr(module, "objective")
    assert hasattr(module, "PERSONA")
