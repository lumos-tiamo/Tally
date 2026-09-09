"""How the value of tool retrieval scales with the size of the tool registry.

The `minus-tool-retrieval` arm barely moves at 17 tools — about 1% more context
per step — and reporting only that number would understate the mechanism badly.
An agent connected to a handful of MCP servers routinely has 100–300 tools, and
the cost of injecting all of them grows linearly while the cost of injecting the
relevant ones does not.

So this measures the same quantity across registry sizes: what a step pays with
retrieval versus without, at 17 (the real diligence registry) and at synthetic
sizes above it. The synthetic tools are shaped like real ones — a summary, two
documented parameters, a detail paragraph — because a registry of one-line stubs
would make the schema cost look artificially small.

No model calls. Prompt construction is deterministic, so this is measurement
rather than estimation.
"""

from __future__ import annotations

import json
from pathlib import Path

from tally.config import settings
from tally.context.tokenizer import count_tokens
from tally.execution.registry import ToolParam, ToolRegistry, ToolSpec
from tally.execution.workspace import Workspace
from tally.scenarios.dd_finance.sec_client import SecClient
from tally.scenarios.dd_finance.spec import build_tools

STEP = "extract the total revenue figure from the income statement of the filing"
SIZES = (17, 40, 80, 160, 320)
K = 8

# Vocabulary for synthetic tools, so retrieval has something to discriminate on
# rather than 300 identically-worded entries.
DOMAINS = (
    ("crm", "contact", "look up a customer contact record"),
    ("calendar", "event", "read or create a calendar event"),
    ("email", "message", "search or send an email message"),
    ("tickets", "issue", "read or update a tracker issue"),
    ("wiki", "page", "search internal documentation pages"),
    ("hr", "employee", "read an employee directory record"),
    ("cloud", "instance", "inspect a compute instance"),
    ("metrics", "series", "query a monitoring time series"),
    ("payments", "invoice", "look up an invoice or payment"),
    ("inventory", "sku", "check stock for a product sku"),
)


def synthetic(count: int) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    i = 0
    while len(specs) < count:
        module, noun, purpose = DOMAINS[i % len(DOMAINS)]
        n = i // len(DOMAINS)
        specs.append(ToolSpec(
            module=module,
            func=f"{noun}_operation_{n}",
            summary=f"{purpose} (variant {n})",
            params=[
                ToolParam("identifier", "str", doc=f"the {noun} identifier"),
                ToolParam("include_history", "bool", "False",
                          doc="whether to include prior revisions"),
            ],
            returns="dict",
            detail=(f"Operates on {module} {noun} records. Returns the record and its "
                    f"metadata. Variant {n} differs in the projection it applies."),
            tags=(module, noun, "operation"),
        ))
        i += 1
    return specs


def measure(size: int, base: ToolRegistry) -> dict:
    registry = ToolRegistry()
    registry.module_sources = dict(base.module_sources)
    registry.extend(base.tools.values())
    extra = max(0, size - len(base))
    if extra:
        registry.extend(synthetic(extra))

    costs = registry.token_cost_comparison(count_tokens, step_description=STEP, k=K)
    retrieved = costs.get("retrieved_signatures", 0)
    all_sigs = costs["signature_lines"]
    schemas = costs["full_json_schemas"]
    return {
        "tools": len(registry),
        "retrieved_signatures": retrieved,
        "retrieved_tool_count": costs.get("retrieved_tools", 0),
        "all_signatures": all_sigs,
        "full_json_schemas": schemas,
        "saving_vs_all_signatures_pct": round(100.0 * (all_sigs - retrieved) / all_sigs, 1)
        if all_sigs else 0.0,
        "saving_vs_schemas_pct": round(100.0 * (schemas - retrieved) / schemas, 1)
        if schemas else 0.0,
    }


def main() -> int:
    cfg = settings()
    base = build_tools(SecClient(cfg=cfg),
                       Workspace.create(cfg.paths.workspaces, "_scaling"))
    rows = [measure(size, base) for size in SIZES]

    report = {
        "measurement": "tool-representation cost vs registry size",
        "step_description": STEP,
        "retrieval_k": K,
        "model_calls": 0,
        "note": ("Registry sizes above 17 are synthetic tools shaped like real ones. "
                 "Retrieval cost is flat in registry size because k is fixed; both "
                 "alternatives grow linearly."),
        "rows": rows,
    }
    out = Path("results") / "tool_retrieval_scaling.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"{'tools':>7} {'retrieved':>10} {'all sigs':>10} {'schemas':>9} "
          f"{'vs all sigs':>12} {'vs schemas':>11}")
    print("-" * 64)
    for row in rows:
        print(f"{row['tools']:>7} {row['retrieved_signatures']:>10} "
              f"{row['all_signatures']:>10} {row['full_json_schemas']:>9} "
              f"{row['saving_vs_all_signatures_pct']:>11.1f}% "
              f"{row['saving_vs_schemas_pct']:>10.1f}%")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
