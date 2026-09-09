"""The diligence scenario: declarative spec, tools, and its eval case runner.

Nothing in this module touches the runtime. That is the platform's acceptance
criterion — a scenario is an :class:`AgentSpec` plus tool modules — and this file
is where it is honoured or broken.

The one rule that makes the eval meaningful
-------------------------------------------
**The agent is never given XBRL.** ``companyfacts`` is the ground truth; a tool
that exposed it would turn the task from "read a 300-page filing" into "call the
answer key". :data:`FORBIDDEN_TOOLS` names what must never be bridged, and
:func:`assert_no_truth_leak` is asserted by the tests so the constraint cannot be
quietly relaxed later.

So the agent gets exactly two network-backed abilities — find the annual report,
download it — and everything else is local text and arithmetic work on the file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from teamclaw.context.ledger import Budget
from teamclaw.context.skills import Skill, SkillLibrary, default_library
from teamclaw.execution.registry import ToolParam, ToolRegistry, ToolSpec
from teamclaw.execution.sandbox import SandboxLimits
from teamclaw.execution.workspace import Workspace
from teamclaw.orchestration.agent import AgentSpec
from teamclaw.scenarios.dd_finance.fields import (
    L1_FIELDS,
    L1_KEYS,
    L2_KEYS,
    field_schema,
)
from teamclaw.scenarios.dd_finance.sec_client import SecClient

SANDBOX_TOOLS = Path(__file__).parent / "sandbox_tools"

# Tool names that would hand the agent the answer key.
FORBIDDEN_TOOLS = frozenset({
    "sec.company_facts", "sec.xbrl", "sec.xbrl_concept", "xbrl.facts",
    "truth.l1", "truth.l2",
})


def assert_no_truth_leak(registry: ToolRegistry) -> None:
    """Fail loudly if any registered tool could serve XBRL to the agent."""
    leaked = sorted(set(registry.names()) & FORBIDDEN_TOOLS)
    if leaked:
        raise AssertionError(
            f"tools would leak ground truth into the task: {leaked}. XBRL company "
            "facts are the eval's truth source and must never be reachable from "
            "the agent."
        )
    for name, spec in registry.tools.items():
        blob = f"{name} {spec.summary} {spec.detail}".lower()
        if "companyfacts" in blob or "xbrl" in blob:
            raise AssertionError(
                f"tool {name!r} mentions XBRL/companyfacts; if it exposes them the "
                "eval is invalid, and if it does not the wording should not imply it"
            )


AGENT_PERSONA = """\
You are a diligence analyst reading SEC annual reports.

Three commitments define your work:

1. **You never report a figure you did not locate in the filing.** Every number
   comes with the page or item it was found in and a short verbatim quote of the
   surrounding text. If a figure is not disclosed, you say so explicitly rather
   than estimating it. An honest gap is a correct answer; a plausible invention
   is the worst possible one.

2. **You compute, you do not recall.** Ratios are derived in code from figures
   you extracted, and you report the numerator and denominator alongside the
   result so it can be re-derived. You do not reproduce figures from memory of
   the company, even when you are confident — the filing in front of you is the
   only source that counts.

3. **You check your own work.** Assets should equal liabilities plus equity. A
   revenue figure should be the same order of magnitude as the prior year.
   Statement tables are usually stated "in millions" or "in thousands" and the
   heading that says so is above the table, not next to the number. When a check
   fails, you investigate before reporting.

Filings are long. Read them by locating what you need, never by loading them
whole.\
"""

DD_SKILLS: tuple[Skill, ...] = (
    Skill(
        name="read-a-10k",
        description="Navigate a 10-K by item structure instead of scanning it",
        when_to_use="before extracting anything from a filing document",
        body=(
            "A 10-K has a fixed structure. Use it:\n"
            "* **Item 7** — MD&A: management's own discussion of the numbers, and the\n"
            "  best place for context on a trend.\n"
            "* **Item 8** — financial statements: the income statement, balance sheet\n"
            "  and cash-flow statement. Every L1 figure lives here.\n"
            "* **Item 1A** — risk factors.\n"
            "Call `doc.outline(path)` first to get item offsets, then\n"
            "`doc.find_number(path, label)` with the caption you expect, e.g.\n"
            "'Total net sales', 'Total cost of sales', 'Total current assets'.\n"
            "Captions vary between filers: if one label finds nothing, try a synonym\n"
            "(`doc.grep` with a pattern) rather than concluding the figure is absent."
        ),
        tags=("filing", "structure", "navigation", "10-k"),
    ),
    Skill(
        name="statement-scale",
        description="Resolve whether a table is in units, thousands or millions",
        when_to_use="whenever a figure is read out of a financial statement table",
        body=(
            "Statement tables state their scale in a heading above the table:\n"
            "'(in millions, except per share amounts)'. The number 391,035 in such a\n"
            "table means 391,035,000,000.\n\n"
            "`doc.find_number` reports the `scale` it found near the figure, and\n"
            "`fin.rescale(value, scale)` applies it. Do both explicitly.\n\n"
            "If no scale heading is found, do not assume units — search backwards\n"
            "with `doc.grep(path, r'in (thousands|millions|billions)')` and use the\n"
            "nearest heading before the table's offset. A mis-scaled figure is wrong\n"
            "by a factor of a thousand and looks entirely plausible."
        ),
        tags=("scale", "units", "thousands", "millions", "table"),
    ),
    Skill(
        name="abstain-on-absence",
        description="Recognise fields a filer structurally does not report",
        when_to_use="when a field cannot be found after a real search",
        body=(
            "Some fields genuinely do not exist in some filings:\n"
            "* Banks and insurers use an **unclassified balance sheet** — there is no\n"
            "  'total current assets' or 'total current liabilities' line, so the\n"
            "  current ratio is undefined, not zero.\n"
            "* Financial and real-estate filers often report **no cost of revenue**,\n"
            "  so gross margin does not exist for them.\n"
            "* Some filers publish no **total liabilities** subtotal, only\n"
            "  liabilities-and-equity.\n\n"
            "For these, emit `{\"value\": null, \"reason\": \"not_disclosed\"}`. Do not\n"
            "derive the figure by subtraction and present it as reported, and do not\n"
            "report 0. Search properly first — two synonyms and a grep — then abstain\n"
            "with confidence."
        ),
        tags=("abstention", "banks", "insurance", "unclassified", "missing"),
    ),
)


def build_skills() -> SkillLibrary:
    return SkillLibrary().extend(default_library().skills.values()).extend(DD_SKILLS)


# --- bridged (host-side) tools -------------------------------------------
@dataclass
class SecHandlers:
    """Host-side SEC access, scoped to one workspace.

    Downloads land in the workspace so the sandbox — which has no network — can
    read them as ordinary files.
    """

    client: SecClient
    workspace: Workspace

    def find_annual_report(self, ticker: str, fiscal_year: int) -> dict[str, Any]:
        """Locate a company's annual report for a fiscal year."""
        cik = self.client.cik_for_ticker(ticker)
        for filing in self.client.annual_filings(cik, limit=20):
            if filing.fiscal_year == int(fiscal_year):
                return {
                    "ticker": ticker.upper(),
                    "cik": cik,
                    "accession": filing.accession,
                    "form": filing.form,
                    "fiscal_year": filing.fiscal_year,
                    "fiscal_year_end": filing.report_date.isoformat()
                    if filing.report_date else None,
                    "filed": filing.filing_date.isoformat(),
                }
        available = sorted(
            {f.fiscal_year for f in self.client.annual_filings(cik, limit=20) if f.fiscal_year},
            reverse=True,
        )
        raise LookupError(
            f"no annual report for {ticker} FY{fiscal_year}; available: {available[:8]}"
        )

    def download_filing(self, ticker: str, fiscal_year: int) -> dict[str, Any]:
        """Download the annual report into the workspace and report a digest.

        Returns the path, not the text: the document is 1-10MB of HTML and
        putting any of it in the reply would defeat the point of the sandbox.
        """
        meta = self.find_annual_report(ticker, fiscal_year)
        cik = meta["cik"]
        filing = next(
            f for f in self.client.annual_filings(cik, limit=20)
            if f.accession == meta["accession"]
        )
        html = self.client.filing_document(filing)
        rel = f"filings/{meta['ticker']}_FY{meta['fiscal_year']}.html"
        artifact = self.workspace.write_text(
            rel, html, note=f"{meta['ticker']} {meta['form']} FY{meta['fiscal_year']}"
        )
        return {
            **meta,
            "path": artifact.rel,
            "bytes": artifact.size,
            "note": "HTML. Convert with doc.to_text(path) before searching.",
        }


def build_tools(client: SecClient, workspace: Workspace) -> ToolRegistry:
    handlers = SecHandlers(client=client, workspace=workspace)
    registry = ToolRegistry()

    registry.add(ToolSpec(
        module="sec", func="find_annual_report",
        summary="Locate a company's annual report for a fiscal year",
        params=[ToolParam("ticker", "str", doc="e.g. 'AAPL'"),
                ToolParam("fiscal_year", "int", doc="e.g. 2024")],
        returns="dict", requires_network=True, handler=handlers.find_annual_report,
        detail="Returns accession, form, fiscal year end and filing date.",
        tags=("sec", "filing", "annual", "report", "10-k", "search", "locate"),
    ))
    registry.add(ToolSpec(
        module="sec", func="download_filing",
        summary="Download an annual report into the workspace and return its path",
        params=[ToolParam("ticker", "str"), ToolParam("fiscal_year", "int")],
        returns="dict", requires_network=True, handler=handlers.download_filing,
        detail="Writes HTML under filings/. Convert with doc.to_text before searching.",
        tags=("sec", "download", "fetch", "filing", "document", "html"),
    ))

    # Local modules, attached as real source files.
    registry.attach_source_file("doc", SANDBOX_TOOLS / "doc.py")
    for spec in _doc_specs():
        registry.add(spec)
    registry.attach_source_file("fin", SANDBOX_TOOLS / "fin.py")
    for spec in _fin_specs():
        registry.add(spec)

    assert_no_truth_leak(registry)
    return registry


def _doc_specs() -> list[ToolSpec]:
    P = ToolParam
    return [
        ToolSpec(module="doc", func="to_text", returns="dict",
                 summary="Strip a filing's HTML to plain text and return a digest",
                 params=[P("path", "str"), P("out_path", "str", "None")],
                 local_source="pass",
                 tags=("convert", "html", "text", "strip", "plain")),
        ToolSpec(module="doc", func="outline", returns="dict",
                 summary="Item-level table of contents with character offsets",
                 params=[P("path", "str")], local_source="pass",
                 tags=("outline", "items", "structure", "toc", "sections")),
        ToolSpec(module="doc", func="locate", returns="dict",
                 summary="Find a phrase and report offset, item and quotable context",
                 params=[P("path", "str"), P("needle", "str"), P("limit", "int", "5")],
                 local_source="pass",
                 tags=("locate", "find", "phrase", "quote", "citation", "search")),
        ToolSpec(module="doc", func="find_number", returns="dict",
                 summary="Locate a labelled figure and the numbers on its line, with scale",
                 params=[P("path", "str"), P("label", "str"), P("limit", "int", "5")],
                 local_source="pass",
                 tags=("number", "figure", "extract", "label", "caption", "value", "scale")),
        ToolSpec(module="doc", func="table_scale", returns="dict",
                 summary="Find the in-thousands/millions heading governing a position",
                 params=[P("path", "str"), P("offset", "int")], local_source="pass",
                 tags=("scale", "units", "thousands", "millions", "heading", "table")),
        ToolSpec(module="doc", func="grep", returns="dict",
                 summary="Regex search returning matched lines with offsets",
                 params=[P("path", "str"), P("pattern", "str"), P("limit", "int", "20")],
                 local_source="pass",
                 tags=("grep", "regex", "search", "explore", "pattern")),
        ToolSpec(module="doc", func="section", returns="dict",
                 summary="Return the head of one Item section, capped",
                 params=[P("path", "str"), P("item", "str"), P("max_chars", "int", "4000")],
                 local_source="pass",
                 tags=("section", "item", "mdna", "read")),
        ToolSpec(module="doc", func="save_json", returns="dict",
                 summary="Persist a result to the workspace and return only its digest",
                 params=[P("obj", "object"), P("path", "str")], local_source="pass",
                 tags=("save", "persist", "json", "workspace", "write")),
    ]


def _fin_specs() -> list[ToolSpec]:
    P = ToolParam
    return [
        ToolSpec(module="fin", func="ratio", returns="dict",
                 summary="Compute one L2 metric from L1 values, with provenance",
                 params=[P("name", "str"), P("values", "dict")], local_source="pass",
                 detail=f"Defined metrics: {', '.join(L2_KEYS)}",
                 tags=("ratio", "margin", "roe", "roa", "compute", "metric", "derive")),
        ToolSpec(module="fin", func="all_ratios", returns="dict",
                 summary="Compute all eight L2 metrics, abstaining where inputs are absent",
                 params=[P("values", "dict")], local_source="pass",
                 tags=("ratios", "all", "compute", "metrics", "batch")),
        ToolSpec(module="fin", func="rescale", returns="dict",
                 summary="Convert a figure stated in thousands/millions/billions to units",
                 params=[P("value", "float"), P("scale", "str")], local_source="pass",
                 tags=("rescale", "scale", "units", "convert", "millions", "thousands")),
        ToolSpec(module="fin", func="growth", returns="dict",
                 summary="Year-over-year growth with provenance",
                 params=[P("current", "float"), P("prior", "float")], local_source="pass",
                 tags=("growth", "yoy", "change", "trend")),
        ToolSpec(module="fin", func="check_balance_sheet", returns="dict",
                 summary="Self-check that assets equal liabilities plus equity",
                 params=[P("values", "dict")], local_source="pass",
                 tags=("check", "balance", "sheet", "verify", "identity", "validate")),
        ToolSpec(module="fin", func="summarise", returns="dict",
                 summary="Digest of an extraction: fields present and missing",
                 params=[P("values", "dict")], local_source="pass",
                 tags=("summarise", "digest", "present", "missing", "coverage")),
        ToolSpec(module="fin", func="load_values", returns="dict",
                 summary="Read a saved L1 extraction back from the workspace",
                 params=[P("path", "str")], local_source="pass",
                 tags=("load", "read", "values", "extraction", "workspace")),
    ]


# --- objectives -----------------------------------------------------------
ALWAYS_TOOLS = ("doc.save_json", "sec.download_filing")


def l1_objective(ticker: str, fiscal_year: int) -> str:
    schema = field_schema()
    return (
        f"Extract the twelve L1 financial fields for {ticker} fiscal year {fiscal_year} "
        f"from its annual report.\n\n"
        f"Fields (use exactly these keys):\n"
        + "\n".join(f"  {f.key} — {f.label} ({f.statement} statement)" for f in L1_FIELDS)
        + "\n\nFor each field emit an object:\n"
        + json.dumps(schema["l1_output_contract"], indent=2)
        + "\n\nValues must be in **whole USD**, not thousands or millions — apply the\n"
        "table's scale. Where the filing does not disclose a field, emit\n"
        '{"value": null, "reason": "not_disclosed"}.\n\n'
        "Write the result to `l1.json` as a single JSON object keyed by field name,\n"
        "then reply DONE with a short summary of coverage. The file is the\n"
        "deliverable; the summary is for the reader.\n\n"
        "Paths are relative to your working directory, which is already the\n"
        "workspace — write `l1.json`, not `workspace/l1.json`."
    )


def l2_objective(ticker: str, fiscal_year: int) -> str:
    schema = field_schema()
    return (
        f"Compute the eight L2 financial metrics for {ticker} fiscal year {fiscal_year}.\n\n"
        f"Metrics: {', '.join(L2_KEYS)}\n\n"
        "First extract the L1 inputs you need from the annual report, then derive each\n"
        "metric **in code** with `fin.ratio` or `fin.all_ratios`. For each metric emit:\n"
        + json.dumps(schema["l2_output_contract"], indent=2)
        + "\n\nA metric whose inputs the filing does not disclose must be null with a\n"
        "reason — banks have no current ratio, financial filers often have no gross\n"
        "margin. Do not derive a missing input by subtraction.\n\n"
        "Write the result to `l2.json` keyed by metric name (a relative path — your\n"
        "working directory is already the workspace), then reply DONE."
    )


def l3_objective(ticker: str, fiscal_year: int, prior_year: int) -> str:
    """The judgement task: what do two years of figures actually show?

    Deliberately does *not* enumerate the signal taxonomy. Handing the agent the
    list of things to look for would turn an analysis task into a checklist, and
    the metric would then measure whether it can fill in a form. The signals are
    scoring machinery, not part of the prompt.
    """
    return (
        f"Compare {ticker} fiscal year {fiscal_year} against fiscal year {prior_year} "
        f"using both annual reports, and report what the figures show.\n\n"
        "Extract the figures you need, compute the changes in code, and then write "
        "the analysis. For each finding, give:\n"
        '  - `observation`: what changed, with both years\' numbers\n'
        '  - `evidence`: the computed change, and the fields it came from\n'
        '  - `question`: what a diligence reader should ask about it\n'
        '  - `severity`: "watch", "concern" or "red_flag"\n\n'
        "Report only what the numbers support. If the two years show nothing "
        "material, say so plainly — an invented concern is worse than a short "
        "report, and a list of every possible worry is not analysis.\n\n"
        "Write the result to `l3.json` (a relative path — your working directory is\n"
        "already the workspace) as "
        '`{"findings": [...], "summary": "<two or three sentences>"}`, '
        "then reply DONE."
    )


def signal_ground_truth(
    current_l1: dict[str, float | None], prior_l1: dict[str, float | None]
) -> tuple[list[str], list[str]]:
    """(present, detectable) signal keys for one year-over-year comparison."""
    from teamclaw.scenarios.dd_finance.signals import detect, detectable

    return [s.key for s in detect(current_l1, prior_l1)], detectable(current_l1, prior_l1)


def build_spec(
    *,
    tools: ToolRegistry,
    window: int = 32_000,
    max_steps: int = 14,
    max_output: int = 2_048,
    prefer_docker: bool = True,
    wall_clock_s: float = 120.0,
) -> AgentSpec:
    return AgentSpec(
        name="dd-analyst",
        scenario="dd_finance",
        persona=AGENT_PERSONA,
        tools=tools,
        skills=build_skills(),
        budget=Budget(window=window, max_output=max_output),
        max_steps=max_steps,
        always_tools=ALWAYS_TOOLS,
        tool_k=8,
        skill_k=2,
        memory_k=5,
        sandbox_limits=SandboxLimits(wall_clock_s=wall_clock_s, memory_mb=2048),
        prefer_docker=prefer_docker,
    )


def parse_output(workspace: Workspace, level: str) -> dict[str, Any]:
    """Read the agent's deliverable file. A missing file scores as no answer.

    Deliberately strict: the objective names the path, and an agent that reports
    numbers in prose but never writes the file has not completed the task. Prose
    is not machine-checkable, and the whole point of the file contract is that
    the result can be verified without a second model reading it.
    """
    name = {"l1": "l1.json", "l2": "l2.json", "l3": "l3.json"}.get(level, "l1.json")
    if not workspace.exists(name):
        return {}
    try:
        data = json.loads(workspace.read_text(name))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    if level == "l3":
        # L3 output is prose plus findings, not a fixed key set, so it is passed
        # through whole; the signal metrics read it structurally and by keyword.
        return data
    valid_keys = set(L1_KEYS if level == "l1" else L2_KEYS)
    return {k: v for k, v in data.items() if k in valid_keys}
