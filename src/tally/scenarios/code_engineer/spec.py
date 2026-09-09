"""Code engineering scenario: change a repository until its tests pass.

Same contract, different pressure on the platform:

* **The sandbox filesystem is the task**, not a side effect. Every other scenario
  writes artefacts; this one edits its inputs.
* **The loop has a real oracle.** ``repo.run_tests`` tells the agent whether it
  succeeded, so the convergence loop gets a hard signal instead of an error
  message — the termination condition is external and objective.
* **Iterative convergence.** Steps are change-verify pairs, and the interesting
  failure is oscillation: fixing one test by breaking another. The persona and
  the skill both target that.

There is no bridged tool here at all. Combined with the BI scenario, that shows
the host bridge is an option the platform offers, not a dependency it imposes.
"""

from __future__ import annotations

from pathlib import Path

from tally.context.ledger import Budget
from tally.context.skills import Skill, SkillLibrary, default_library
from tally.execution.registry import ToolParam, ToolRegistry, ToolSpec
from tally.execution.sandbox import SandboxLimits
from tally.orchestration.agent import AgentSpec

SANDBOX_TOOLS = Path(__file__).parent / "sandbox_tools"

PERSONA = """\
You are a software engineer changing an existing repository.

* You read before you write. `repo.grep` and `repo.read` with line ranges, never
  whole files — a file you dumped into context is budget you cannot spend on
  reasoning.
* You edit by exact-match replacement with enough surrounding context to be
  unambiguous. If a replacement is refused for being ambiguous, add context
  rather than trying a different anchor.
* You run the tests after every change, and you read the failure count, not just
  the exit code. A change that fixes one test and breaks two is a regression.
* You stop when the suite is green, and you say what you changed and why. If you
  cannot get it green, you report the remaining failure honestly rather than
  weakening the test.\
"""

ENGINEERING_SKILLS = (
    Skill(
        name="change-then-verify",
        description="Run the tests after every edit and compare failure counts",
        when_to_use="after any repository modification",
        body=(
            "One edit, then `repo.run_tests(root)`. Compare `failed` against the\n"
            "previous run:\n"
            "* fewer failures — keep going;\n"
            "* same failures — your change did not address the cause; re-read the\n"
            "  failing test before editing again;\n"
            "* more failures — revert with `repo.revert(root, backup)` before trying\n"
            "  something else. Stacking a second guess on a bad first one makes both\n"
            "  hard to unwind.\n"
            "Never edit a test to make it pass unless the task explicitly asks for it."
        ),
        tags=("test", "verify", "run", "regression", "revert", "iterate", "failures"),
    ),
    Skill(
        name="unique-anchor-edits",
        description="Give replacements enough context to match exactly once",
        when_to_use="before calling repo.replace",
        body=(
            "`repo.replace` requires the old text to occur exactly once. When it\n"
            "reports multiple occurrences, do not switch to a shorter anchor —\n"
            "extend it upward to include the enclosing function signature or the\n"
            "preceding line.\n"
            "When it reports the text was not found, the file has changed since you\n"
            "read it: re-read the range and rebuild the anchor from what is there now."
        ),
        tags=("edit", "replace", "anchor", "unique", "patch", "context"),
    ),
)


def build_tools() -> ToolRegistry:
    """Entirely local: the repository is mounted, so nothing needs the network."""
    P = ToolParam
    registry = ToolRegistry()
    registry.attach_source_file("repo", SANDBOX_TOOLS / "repo.py")
    registry.extend([
        ToolSpec(module="repo", func="tree", returns="dict",
                 summary="List matching files with sizes and line counts",
                 params=[P("root", "str"), P("pattern", "str", "'*.py'"),
                         P("limit", "int", "200")], local_source="pass",
                 tags=("tree", "files", "list", "repository", "structure")),
        ToolSpec(module="repo", func="read", returns="dict",
                 summary="Read a line range from a file, with line numbers",
                 params=[P("root", "str"), P("rel", "str"), P("start", "int", "1"),
                         P("end", "int", "0")], local_source="pass",
                 tags=("read", "file", "lines", "source", "view")),
        ToolSpec(module="repo", func="grep", returns="dict",
                 summary="Search the repository for a pattern, returning file:line hits",
                 params=[P("root", "str"), P("pattern", "str"),
                         P("glob", "str", "'*.py'"), P("limit", "int", "40")],
                 local_source="pass",
                 tags=("grep", "search", "find", "pattern", "symbol", "usage")),
        ToolSpec(module="repo", func="replace", returns="dict",
                 summary="Replace an exact snippet that must occur exactly once",
                 params=[P("root", "str"), P("rel", "str"), P("old", "str"),
                         P("new", "str")], local_source="pass",
                 tags=("replace", "edit", "patch", "modify", "change", "fix")),
        ToolSpec(module="repo", func="write_file", returns="dict",
                 summary="Create or overwrite a file, backing up any existing version",
                 params=[P("root", "str"), P("rel", "str"), P("content", "str")],
                 local_source="pass",
                 tags=("write", "create", "file", "new", "overwrite")),
        ToolSpec(module="repo", func="revert", returns="dict",
                 summary="Restore a file from a backup produced by an earlier edit",
                 params=[P("root", "str"), P("backup_rel", "str")], local_source="pass",
                 tags=("revert", "undo", "restore", "backup", "rollback")),
        ToolSpec(module="repo", func="run_tests", returns="dict",
                 summary="Run pytest and return pass/fail counts with the output tail",
                 params=[P("root", "str"), P("target", "str", "''"),
                         P("timeout_s", "int", "120")], local_source="pass",
                 tags=("test", "pytest", "run", "verify", "suite", "green", "failures")),
        ToolSpec(module="repo", func="save_json", returns="dict",
                 summary="Persist a result to the workspace and return its digest",
                 params=[P("obj", "object"), P("path", "str")], local_source="pass",
                 tags=("save", "persist", "json", "write")),
    ])
    return registry


def build_skills() -> SkillLibrary:
    return SkillLibrary().extend(default_library().skills.values()).extend(ENGINEERING_SKILLS)


def objective(task: str, repo_root: str) -> str:
    return (
        f"Repository: `{repo_root}`\n\nTask:\n  {task}\n\n"
        "Work as follows:\n"
        "  1. `repo.run_tests(root)` first, to see the starting state.\n"
        "  2. Locate the relevant code with `repo.grep`, read only the ranges you need.\n"
        "  3. Make one change, then run the tests again and compare failure counts.\n"
        "  4. Repeat until green.\n\n"
        "Write a summary to `change_report.json` (a relative path — your working "
        "directory is already the workspace) as:\n"
        '  {"green": <bool>, "files_changed": [...], "tests_passed": <int>,\n'
        '   "tests_failed": <int>, "summary": "<what you changed and why>"}\n\n'
        "Then reply DONE. If you could not reach green, say which failure remains "
        "and what you tried — do not weaken the test to pass it."
    )


def build_spec(*, tools: ToolRegistry, window: int = 32_000,
               max_steps: int = 16, prefer_docker: bool = True) -> AgentSpec:
    return AgentSpec(
        name="code-engineer",
        scenario="code_engineer",
        persona=PERSONA,
        tools=tools,
        skills=build_skills(),
        budget=Budget(window=window, max_output=2_560),
        max_steps=max_steps,
        always_tools=("repo.run_tests", "repo.replace"),
        tool_k=8,
        skill_k=2,
        memory_k=5,
        sandbox_limits=SandboxLimits(wall_clock_s=180.0, memory_mb=2048),
        prefer_docker=prefer_docker,
    )
