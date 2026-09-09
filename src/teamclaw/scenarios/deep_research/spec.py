"""Deep research scenario: parallel sub-agents with enforced provenance.

Same contract as every scenario — AgentSpec plus tool modules. What it stresses
in the platform:

* **Sub-agent context isolation.** Each sub-topic is delegated; the parent
  receives a digest and file paths, never the child's transcript. This is the
  scenario that would break a delegation design that merely relocates tokens.
* **Provenance as a tool invariant.** ``notes.add_claim`` refuses a claim whose
  source is not registered, so citation discipline does not depend on the model
  remembering to cite.
* **Source credibility as an explicit tier**, and corroboration counted rather
  than asserted.

Web access is intentionally bridged rather than local, so every outbound fetch
passes the host broker's audit — which is what makes a research run reviewable
after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from teamclaw.context.ledger import Budget
from teamclaw.context.skills import Skill, SkillLibrary, default_library
from teamclaw.execution.registry import ToolParam, ToolRegistry, ToolSpec
from teamclaw.execution.sandbox import SandboxLimits
from teamclaw.orchestration.agent import AgentSpec

SANDBOX_TOOLS = Path(__file__).parent / "sandbox_tools"

PERSONA = """\
You are a research analyst producing a sourced brief.

* Every claim you record must name the source it came from and quote the text
  that supports it. The note tool will refuse a claim without one, and that
  refusal is not an obstacle to work around — it is the standard.
* You classify each source: primary (filings, official statistics, standards),
  reported (established outlets), secondary (named analysis), unattributed
  (forums, anonymous). A finding that rests only on unattributed sources is
  reported as exactly that.
* You separate what is corroborated from what rests on one source, and you say
  which is which in the brief.
* You delegate a sub-topic when it needs its own extended search, and you read
  back the sub-agent's artefacts rather than its reasoning.\
"""

RESEARCH_SKILLS = (
    Skill(
        name="source-before-claim",
        description="Register the source, then record the claim against it",
        when_to_use="whenever a finding is about to be written down",
        body=(
            "Order matters: `notes.add_source(root, url, title, tier)` returns a\n"
            "`source_id`, and `notes.add_claim(root, claim, source_id, quote)` needs\n"
            "it. A claim with no registered source cannot be recorded, so it cannot\n"
            "reach the report.\n"
            "Quote the supporting sentence, not your paraphrase of it: the quote is\n"
            "what lets a reader check you."
        ),
        tags=("source", "claim", "citation", "provenance", "record", "note"),
    ),
    Skill(
        name="corroborate-then-conclude",
        description="Count independent sources before stating a finding as settled",
        when_to_use="before writing any conclusion in a brief",
        body=(
            "Run `notes.corroboration(root)` before assembling the report.\n"
            "A finding with one source is a lead, not a fact; say so.\n"
            "Two sources that both cite the same original are one source — check\n"
            "whether they are independent before counting them twice."
        ),
        tags=("corroborate", "independent", "confirm", "conclusion", "finding"),
    ),
)


@dataclass
class WebHandlers:
    """Host-side web access. Injected so a test can supply a fake fetcher.

    The signature is deliberately narrow — search and fetch, nothing else — so
    the audit log of a research run is a complete record of what it touched.
    """

    search: Callable[[str, int], list[dict[str, Any]]]
    fetch: Callable[[str], dict[str, Any]]

    def web_search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Search the web and return titles, urls and snippets."""
        return self.search(query, int(limit))

    def web_fetch(self, url: str) -> dict[str, Any]:
        """Fetch one page as text. Returns a digest and the extracted body."""
        return self.fetch(url)


def build_tools(handlers: WebHandlers) -> ToolRegistry:
    P = ToolParam
    registry = ToolRegistry()
    registry.extend([
        ToolSpec(module="web", func="search", summary="Search the web for a query",
                 params=[P("query", "str"), P("limit", "int", "8")], returns="list",
                 requires_network=True, handler=handlers.web_search,
                 tags=("search", "web", "query", "find", "google", "results")),
        ToolSpec(module="web", func="fetch", summary="Fetch one page and extract its text",
                 params=[P("url", "str")], returns="dict",
                 requires_network=True, handler=handlers.web_fetch,
                 tags=("fetch", "page", "url", "download", "read", "article")),
    ])
    registry.attach_source_file("notes", SANDBOX_TOOLS / "notes.py")
    registry.extend([
        ToolSpec(module="notes", func="add_source", returns="dict",
                 summary="Register a source and get the id claims must cite",
                 params=[P("root", "str"), P("url", "str"), P("title", "str"),
                         P("tier", "str", "'secondary'")], local_source="pass",
                 detail="Tiers: primary, reported, secondary, unattributed.",
                 tags=("source", "register", "tier", "credibility", "citation")),
        ToolSpec(module="notes", func="add_claim", returns="dict",
                 summary="Record a claim against a registered source, with a quote",
                 params=[P("root", "str"), P("claim", "str"), P("source_id", "str"),
                         P("quote", "str", "''")], local_source="pass",
                 tags=("claim", "record", "finding", "note", "quote", "evidence")),
        ToolSpec(module="notes", func="load", returns="dict",
                 summary="All recorded claims and sources, with counts by tier",
                 params=[P("root", "str")], local_source="pass",
                 tags=("load", "claims", "sources", "review", "notes")),
        ToolSpec(module="notes", func="corroboration", returns="dict",
                 summary="Which findings have more than one independent source",
                 params=[P("root", "str")], local_source="pass",
                 tags=("corroboration", "independent", "confirm", "agreement")),
        ToolSpec(module="notes", func="build_report", returns="dict",
                 summary="Assemble a cited markdown brief from the recorded claims",
                 params=[P("root", "str"), P("title", "str"),
                         P("out_path", "str", "'report.md'")], local_source="pass",
                 tags=("report", "brief", "assemble", "markdown", "write", "output")),
    ])
    return registry


def build_skills() -> SkillLibrary:
    return SkillLibrary().extend(default_library().skills.values()).extend(RESEARCH_SKILLS)


def objective(topic: str, questions: tuple[str, ...] = ()) -> str:
    asks = "\n".join(f"  - {q}" for q in questions) or "  - (define the sub-questions yourself)"
    return (
        f"Produce a sourced research brief on: {topic}\n\n"
        f"Questions to answer:\n{asks}\n\n"
        "Method:\n"
        "  1. Search, then fetch the pages worth reading.\n"
        "  2. Register each source with a credibility tier, then record claims\n"
        "     against it with a supporting quote.\n"
        "  3. Check corroboration before concluding anything.\n"
        "  4. Build the brief to `report.md` — paths are relative to your working\n"
        "     directory, which is already the workspace.\n\n"
        "Then reply DONE with the headline findings and how well sourced each is."
    )


def build_spec(*, tools: ToolRegistry, window: int = 32_000,
               max_steps: int = 14, prefer_docker: bool = True) -> AgentSpec:
    return AgentSpec(
        name="research-analyst",
        scenario="deep_research",
        persona=PERSONA,
        tools=tools,
        skills=build_skills(),
        budget=Budget(window=window, max_output=2_048),
        max_steps=max_steps,
        always_tools=("notes.add_claim", "web.search"),
        tool_k=7,
        skill_k=2,
        memory_k=6,
        sandbox_limits=SandboxLimits(wall_clock_s=90.0, memory_mb=1024),
        prefer_docker=prefer_docker,
    )
