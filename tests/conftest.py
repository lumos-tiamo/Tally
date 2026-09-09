"""Shared fixtures.

Every test here runs with no network and no credentials: the model layer uses
``FakeProvider``/``ScriptedProvider``, and SEC access is either cached or absent.
A test suite that needs the internet is a test suite that gets skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from teamclaw.context.ledger import Budget
from teamclaw.execution.registry import ToolParam, ToolRegistry, ToolSpec
from teamclaw.execution.workspace import Workspace
from teamclaw.models.providers.fake import FakeProvider, ScriptedProvider
from teamclaw.models.router import Registry, Router
from teamclaw.observability.trace import Tracer, new_run_id
from teamclaw.orchestration.agent import AgentSpec


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace.create(tmp_path / "ws", "test-run")


@pytest.fixture
def tracer(tmp_path: Path) -> Tracer:
    t = Tracer(new_run_id(), tmp_path / "runs")
    yield t
    t.close()


def single_provider_registry(provider) -> Registry:  # noqa: ANN001
    """Route every purpose to one provider, for deterministic tests."""
    registry = Registry(providers={provider.name: provider})
    for purpose in list(registry.policy):
        registry.policy[purpose] = (provider.name,)
    return registry


@pytest.fixture
def fake_registry() -> Registry:
    return single_provider_registry(FakeProvider())


@pytest.fixture
def router(fake_registry: Registry, tracer: Tracer) -> Router:
    return Router(fake_registry, tracer=tracer)


@pytest.fixture
def scripted():
    """Factory: build a Router driven by a fixed list of model replies."""

    def _build(responses, tracer=None, **kwargs):  # noqa: ANN001
        provider = ScriptedProvider(responses, **kwargs)
        return Router(single_provider_registry(provider), tracer=tracer), provider

    return _build


@pytest.fixture
def demo_tools() -> ToolRegistry:
    """Two tools: one bridged (host handler), one local (runs in the sandbox)."""

    def fetch(cik: str, form: str = "10-K") -> dict:
        return {"cik": cik, "form": form, "revenue": 391035000000,
                "cost_of_revenue": 210352000000, "page": 31}

    return ToolRegistry().extend([
        ToolSpec(module="sec", func="fetch", summary="Fetch filing figures",
                 params=[ToolParam("cik", "str"), ToolParam("form", "str", "'10-K'")],
                 returns="dict", requires_network=True, handler=fetch,
                 tags=("sec", "filing", "fetch")),
        ToolSpec(module="fin", func="gross_margin", summary="Gross margin with provenance",
                 params=[ToolParam("revenue", "float"), ToolParam("cost_of_revenue", "float")],
                 returns="dict",
                 local_source=("gm = (revenue - cost_of_revenue) / revenue\n"
                               "return {'value': gm, 'numerator': revenue - cost_of_revenue,"
                               " 'denominator': revenue}"),
                 tags=("ratio", "margin", "gross", "compute")),
    ])


@pytest.fixture
def agent_spec(demo_tools: ToolRegistry) -> AgentSpec:
    return AgentSpec(
        name="test-agent", scenario="test",
        persona="You are a test analyst. Cite everything.",
        tools=demo_tools, budget=Budget(window=16_000, max_output=800),
        max_steps=6, prefer_docker=False,
    )


@pytest.fixture
def sample_case():
    """A ground-truth case built from real Apple FY2024 figures."""
    from teamclaw.scenarios.dd_finance.groundtruth import GroundTruthCase

    return GroundTruthCase(
        case_id="AAPL-FY2024", ticker="AAPL", cik="0000320193", fiscal_year=2024,
        fy_end="2024-09-28", sector="tech", difficulty=(), held_out=False,
        accession="0000320193-24-000123", document_url="https://example.invalid/aapl.htm",
        l1={
            "revenue": 391035000000.0, "cost_of_revenue": 210352000000.0,
            "operating_income": 123216000000.0, "net_income": 93736000000.0,
            "total_assets": 364980000000.0, "total_liabilities": 308030000000.0,
            "total_equity": 56950000000.0, "current_assets": 152987000000.0,
            "current_liabilities": 176392000000.0, "accounts_receivable": 33410000000.0,
            "cash_from_operations": 118254000000.0, "capex": 9447000000.0,
        },
        l1_absent=[], l1_absence_trusted=[],
        l2={"gross_margin": 0.46206349815, "net_margin": 0.23971255769},
    )


@pytest.fixture
def filing_text(tmp_path: Path) -> Path:
    """A miniature filing with the structure the doc tools depend on."""
    text = "\n".join([
        "APPLE INC. FORM 10-K",
        "INDEX",
        "Item 1. Business  1",
        "Item 1A. Risk Factors  5",
        "Item 7. Management's Discussion  25",
        "Item 8. Financial Statements  40",
        "",
        "Item 1. Business",
        "The Company designs and markets smartphones.",
        "X" * 2000,
        "Item 1A. Risk Factors",
        "The markets are highly competitive.",
        "X" * 2000,
        "Item 7. Management's Discussion and Analysis",
        "(in millions)",
        "Total net sales $ 391,035 2 % $ 383,285",
        "Percentage of total net sales 8 % 8 %",
        "X" * 1500,
        "Item 8. Financial Statements",
        "CONSOLIDATED STATEMENTS OF OPERATIONS (In millions)",
        "Years ended September 28, 2024 September 30, 2023",
        "Total net sales 391,035 383,285 394,328",
        "Total cost of sales 210,352 214,137 223,546",
        "Operating income 123,216 114,301 119,437",
        "Net income 93,736 96,995 99,803",
        "CONSOLIDATED BALANCE SHEETS (In millions)",
        "Total current assets 152,987 143,566",
        "Total assets 364,980 352,583",
        "Total current liabilities 176,392 145,308",
        "Total liabilities 308,030 290,437",
        "Total shareholders' equity 56,950 62,146",
        "Accounts receivable, net 33,410 29,508",
        "CONSOLIDATED STATEMENTS OF CASH FLOWS (In millions)",
        "Cash generated by operating activities 118,254 110,543",
        "Payments for acquisition of property, plant and equipment (9,447) (10,959)",
        "Item 9. Changes in Accountants",
        "See the information in Item 8 above.",
    ])
    path = tmp_path / "filing.txt"
    path.write_text(text, encoding="utf-8")
    return path
