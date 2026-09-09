"""BI analyst scenario: answer a business question against a SQL database.

Written to the same contract as every other scenario — an :class:`AgentSpec`
plus tool modules, no runtime code. What it stresses in the platform is
different from the diligence scenario, which is the point of having it:

* **Schema as workspace state.** The agent discovers the database through
  ``sql.schema``, and the schema digest occupies the workspace slot. A wide
  warehouse schema is itself a context problem.
* **Execution safety at the tool boundary.** Read-only connections, a
  write/DDL blocklist, forced limits and a query timeout. The diligence scenario
  never executes anything the model composed; here everything is.
* **Short horizons, tight budgets.** Most questions resolve in two or three
  steps, which exercises the opposite end of the loop from a forty-step
  diligence run.
"""

from __future__ import annotations

from pathlib import Path

from teamclaw.context.ledger import Budget
from teamclaw.context.skills import Skill, SkillLibrary, default_library
from teamclaw.execution.registry import ToolParam, ToolRegistry, ToolSpec
from teamclaw.execution.sandbox import SandboxLimits
from teamclaw.orchestration.agent import AgentSpec

SANDBOX_TOOLS = Path(__file__).parent / "sandbox_tools"

PERSONA = """\
You are a data analyst answering business questions against a SQL database.

* You do not guess at schema. Call `sql.schema(db)` first, and `sql.sample` when
  you need to see what values in a column actually look like.
* You answer with a number and the query that produced it. An answer without its
  query cannot be checked, and an unchecked number is not an answer.
* You state the grain of your result — per what, over what period — because a
  correct aggregate at the wrong grain is the most common way a dashboard lies.
* If the schema cannot answer the question, say what is missing rather than
  answering a nearby question instead.\
"""

BI_SKILLS = (
    Skill(
        name="schema-first",
        description="Discover the schema before composing any query",
        when_to_use="at the start of any database question",
        body=(
            "Call `sql.schema(db)` and read the table and column names before\n"
            "writing SQL. Then `sql.sample(db, table)` on the one or two tables that\n"
            "look relevant — column names lie about their contents often enough that\n"
            "a five-row sample is always worth one step.\n"
            "Check the join keys exist before joining on them."
        ),
        tags=("schema", "database", "discover", "tables", "columns", "sql"),
    ),
    Skill(
        name="state-the-grain",
        description="Report what each row of a result represents",
        when_to_use="whenever an aggregate is reported",
        body=(
            "Every aggregate has a grain: per customer, per month, per order line.\n"
            "State it explicitly with the number.\n"
            "Watch for fan-out: joining a one-to-many relationship before summing\n"
            "multiplies the measure. If a total looks too large by a suspicious\n"
            "factor, count the rows before and after the join."
        ),
        tags=("grain", "aggregate", "sum", "join", "fanout", "duplicate"),
    ),
)


def build_tools(db_path: Path | str) -> ToolRegistry:
    """Local-only registry: SQL runs in the sandbox against a read-only file.

    No bridged tools at all. The database is mounted, not fetched, so this
    scenario needs no host-side egress — which also demonstrates that the bridge
    is optional rather than load-bearing.
    """
    P = ToolParam
    registry = ToolRegistry()
    registry.attach_source_file("sql", SANDBOX_TOOLS / "sql.py")
    registry.extend([
        ToolSpec(module="sql", func="schema", returns="dict",
                 summary="Tables, columns and row counts for the database",
                 params=[P("db_path", "str")], local_source="pass",
                 tags=("schema", "tables", "columns", "database", "discover")),
        ToolSpec(module="sql", func="run", returns="dict",
                 summary="Execute a read-only SELECT and return rows",
                 params=[P("db_path", "str"), P("query", "str"), P("limit", "int", "200")],
                 local_source="pass",
                 detail="Rejects non-SELECT statements and forces a LIMIT.",
                 tags=("query", "select", "run", "execute", "sql")),
        ToolSpec(module="sql", func="check", returns="dict",
                 summary="Static safety check on a query before running it",
                 params=[P("query", "str")], local_source="pass",
                 tags=("check", "safety", "validate", "query")),
        ToolSpec(module="sql", func="sample", returns="dict",
                 summary="A few rows from one table to learn its value shapes",
                 params=[P("db_path", "str"), P("table", "str"), P("n", "int", "5")],
                 local_source="pass",
                 tags=("sample", "rows", "peek", "table", "values")),
        ToolSpec(module="sql", func="save_json", returns="dict",
                 summary="Persist a result to the workspace and return its digest",
                 params=[P("obj", "object"), P("path", "str")], local_source="pass",
                 tags=("save", "persist", "json", "write")),
    ])
    return registry


def build_skills() -> SkillLibrary:
    return SkillLibrary().extend(default_library().skills.values()).extend(BI_SKILLS)


def objective(question: str, db_path: str) -> str:
    return (
        f"Answer this business question using the SQLite database at `{db_path}`:\n\n"
        f"  {question}\n\n"
        "Write your answer to `workspace/answer.json` as:\n"
        '  {"answer": <number or string>, "grain": "<what one row represents>",\n'
        '   "query": "<the SQL that produced it>", "caveats": "<or empty>"}\n\n'
        "If the schema cannot answer the question, write "
        '{"answer": null, "reason": "<what is missing>"} instead of answering a\n'
        "nearby question. Then reply DONE with a one-line summary."
    )


def build_spec(*, tools: ToolRegistry, window: int = 24_000,
               max_steps: int = 8, prefer_docker: bool = True) -> AgentSpec:
    return AgentSpec(
        name="bi-analyst",
        scenario="bi_analyst",
        persona=PERSONA,
        tools=tools,
        skills=build_skills(),
        budget=Budget(window=window, max_output=1_536),
        max_steps=max_steps,
        always_tools=("sql.schema", "sql.run"),
        tool_k=6,
        skill_k=2,
        memory_k=4,
        sandbox_limits=SandboxLimits(wall_clock_s=60.0, memory_mb=1024),
        prefer_docker=prefer_docker,
    )
