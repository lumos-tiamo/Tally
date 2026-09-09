"""Command line entry point.

Grouped by what you are trying to find out:

    teamclaw doctor              what is configured and what is not
    teamclaw scenarios           the platform-abstraction cost table
    teamclaw tools               tool-representation token costs
    teamclaw dataset build|stats the ground-truth corpus
    teamclaw baseline            the zero-model floor
    teamclaw run                 one case, one arm, verbose
    teamclaw eval                one arm over the corpus
    teamclaw ablate              every arm, then the comparison table
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from teamclaw import __version__
from teamclaw.config import settings

app = typer.Typer(add_completion=False, help="Multi-agent digital-employee platform.")
dataset_app = typer.Typer(help="Ground-truth corpus.")
app.add_typer(dataset_app, name="dataset")
console = Console()


def _dataset_path() -> Path:
    return settings().paths.datasets / "dd_finance_groundtruth.jsonl"


# --- doctor ---------------------------------------------------------------
@app.command()
def doctor() -> None:
    """Report what is configured, what is reachable, and what is missing."""
    from teamclaw.execution.sandbox import DockerSandbox, build_sandbox
    from teamclaw.models.registry_default import build_registry, describe_registry

    cfg = settings()
    console.print(f"[bold]teamclaw {__version__}[/bold]")

    table = Table("provider", "model", "paid", "available", title="Model providers")
    for row in describe_registry(build_registry(cfg, include_fake=True)):
        table.add_row(row["name"], row["model"], "yes" if row["paid"] else "no",
                      "[green]yes[/green]" if row["available"] else "[red]no[/red]")
    console.print(table)

    docker = DockerSandbox(workspace=cfg.paths.workspaces)
    sandbox = build_sandbox(workspace=cfg.paths.workspaces, prefer_docker=True)
    env = Table("check", "status", title="Environment")
    env.add_row("docker daemon", "yes" if docker.available() else "[yellow]not running[/yellow]")
    env.add_row("sandbox image", "present" if docker.image_present()
                else "[yellow]absent (build docker/Dockerfile.sandbox)[/yellow]")
    env.add_row("active sandbox", f"{sandbox.backend} / {sandbox.isolation_level()}")
    env.add_row("SEC user agent", cfg.sec_user_agent or "[red]unset[/red]")
    env.add_row("paid calls", "enabled" if cfg.allow_paid else "blocked")
    path = _dataset_path()
    env.add_row("ground truth", str(path) if path.exists() else "[yellow]not built[/yellow]")
    console.print(env)

    if sandbox.backend != "docker":
        console.print(
            "\n[yellow]Note[/yellow] the local sandbox shares the host kernel and "
            "filesystem namespace. Results record their isolation level, but do not "
            "run untrusted code under it."
        )


# --- scenarios / tools ----------------------------------------------------
@app.command()
def scenarios() -> None:
    """Cost of each scenario. The platform-abstraction claim, as a table."""
    root = Path(__file__).parent / "scenarios"
    table = Table("scenario", "AgentSpec lines", "tool lines", "total",
                  "runtime changes", title="Cost of adding a scenario")
    totals: list[int] = []
    for package in ("dd_finance", "bi_analyst", "deep_research", "code_engineer"):
        spec_lines = tool_lines = 0
        for path in (root / package).rglob("*.py"):
            count = sum(1 for line in path.read_text(encoding="utf-8").splitlines()
                        if line.strip() and not line.strip().startswith("#"))
            if "sandbox_tools" in path.parts:
                tool_lines += count
            else:
                spec_lines += count
        total = spec_lines + tool_lines
        if package != "dd_finance":
            totals.append(total)
        table.add_row(package, str(spec_lines), str(tool_lines), str(total), "0")
    console.print(table)
    if totals:
        console.print(f"Light scenarios: mean [bold]{sum(totals) // len(totals)}[/bold] lines, "
                      f"range {min(totals)}–{max(totals)}, zero runtime changes.")


@app.command()
def tools(scenario: str = typer.Option("dd_finance", help="Scenario to inspect.")) -> None:
    """Token cost of the three tool representations."""
    from teamclaw.context.tokenizer import count_tokens
    from teamclaw.execution.workspace import Workspace
    from teamclaw.scenarios.dd_finance.sec_client import SecClient

    cfg = settings()
    if scenario == "dd_finance":
        from teamclaw.scenarios.dd_finance.spec import build_tools

        registry = build_tools(SecClient(cfg=cfg),
                               Workspace.create(cfg.paths.workspaces, "_inspect"))
    elif scenario == "bi_analyst":
        from teamclaw.scenarios.bi_analyst.spec import build_tools

        registry = build_tools(cfg.paths.data / "example.db")
    elif scenario == "code_engineer":
        from teamclaw.scenarios.code_engineer.spec import build_tools

        registry = build_tools()
    else:
        raise typer.BadParameter(f"unknown or web-dependent scenario: {scenario}")

    step = "extract the total revenue figure from the income statement of the filing"
    costs = registry.token_cost_comparison(count_tokens, step_description=step, k=8)
    table = Table("representation", "tokens", "what it is",
                  title=f"{scenario}: {costs['tools']} tools")
    table.add_row("retrieved signatures",
                  str(costs.get("retrieved_signatures", "-")),
                  f"what one step actually pays ({costs.get('retrieved_tools', 0)} tools)")
    table.add_row("all signatures", str(costs["signature_lines"]),
                  "every tool, no retrieval")
    table.add_row("full JSON schemas", str(costs["full_json_schemas"]),
                  "what publishing MCP schemas would cost")
    table.add_row("stub sources", str(costs["stub_sources"]),
                  "on disk in the sandbox, read on demand — never in a prompt")
    console.print(table)
    per_step = costs.get("retrieved_signatures")
    if per_step and costs["full_json_schemas"]:
        saved = costs["full_json_schemas"] - per_step
        console.print(
            f"Per step: [bold]{per_step}[/bold] tokens instead of "
            f"[bold]{costs['full_json_schemas']}[/bold] — "
            f"{saved / costs['full_json_schemas']:.0%} lower than injecting schemas."
        )


# --- dataset --------------------------------------------------------------
@dataset_app.command("build")
def dataset_build(
    years: int = typer.Option(3, help="Fiscal years per company."),
    limit: Optional[int] = typer.Option(None, help="Only the first N companies."),
) -> None:
    """Build the ground-truth corpus from SEC XBRL."""
    from teamclaw.scenarios.dd_finance.corpus import CORPUS
    from teamclaw.scenarios.dd_finance.groundtruth import build_dataset, save_dataset
    from teamclaw.scenarios.dd_finance.sec_client import SecClient

    cfg = settings()
    cfg.require_sec_user_agent()
    entries = list(CORPUS)[:limit] if limit else list(CORPUS)
    report = build_dataset(SecClient(cfg=cfg), entries, years=years, progress=True)
    path = save_dataset(report, _dataset_path())
    summary = report.summary()
    console.print(f"\n[green]built[/green] {summary['usable']} usable cases "
                  f"({summary['quarantined']} quarantined) -> {path}")


@dataset_app.command("stats")
def dataset_stats() -> None:
    """Summarise the built corpus."""
    from teamclaw.scenarios.dd_finance.groundtruth import load_dataset

    path = _dataset_path()
    summary_path = path.parent / f"{path.stem}.summary.json"
    if not summary_path.exists():
        console.print("[red]no dataset built[/red] — run `teamclaw dataset build`")
        raise typer.Exit(1)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cases = load_dataset(path)

    table = Table("metric", "value", title="Ground-truth corpus")
    for key in ("cases_built", "usable", "quarantined", "held_out", "errors",
                "mean_l1_coverage"):
        table.add_row(key, str(summary.get(key)))
    table.add_row("primary (trainable)", str(sum(1 for c in cases if not c.held_out)))
    console.print(table)

    sectors = Table("sector", "cases", title="By sector")
    for sector, count in summary.get("by_sector", {}).items():
        sectors.add_row(sector, str(count))
    console.print(sectors)

    absences = Table("field", "cases where the filer omits it",
                     title="Abstention targets")
    for field, count in summary.get("abstention_targets", {}).items():
        absences.add_row(field, str(count))
    console.print(absences)


# --- baseline -------------------------------------------------------------
@app.command()
def baseline(limit: Optional[int] = typer.Option(None, help="Only the first N cases.")) -> None:
    """Zero-model floor: the document tools driven by a fixed caption table."""
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts" / "deterministic_baseline.py"
    argv = [sys.executable, str(script)] + ([str(limit)] if limit else [])
    raise typer.Exit(subprocess.call(argv))


# --- eval -----------------------------------------------------------------
def _load_cases(level: str, limit: int | None, held_out: bool):  # noqa: ANN202
    from teamclaw.scenarios.dd_finance.groundtruth import load_dataset

    cases = [c for c in load_dataset(_dataset_path()) if c.held_out == held_out]
    if level == "l2":
        cases = [c for c in cases if any(v is not None for v in c.l2.values())]
    return cases[:limit] if limit else cases


@app.command()
def eval(  # noqa: A001 - the command really is called eval
    arm: str = typer.Option("full", help="Ablation arm name."),
    level: str = typer.Option("l1", help="Task level: l1 or l2."),
    limit: Optional[int] = typer.Option(None, help="Only the first N cases."),
    held_out: bool = typer.Option(False, help="Use the held-out split."),
    echo: bool = typer.Option(False, help="Stream trace spans."),
) -> None:
    """Run one ablation arm over the corpus and save its scores."""
    from teamclaw.evaluation.harness import Harness
    from teamclaw.evaluation.runner import ARMS_BY_NAME, RunnerContext, make_case_runner
    from teamclaw.models.registry_default import build_registry, describe_registry
    from teamclaw.scenarios.dd_finance.sec_client import SecClient

    cfg = settings()
    if arm not in ARMS_BY_NAME:
        raise typer.BadParameter(f"unknown arm {arm!r}; known: {sorted(ARMS_BY_NAME)}")
    config = ARMS_BY_NAME[arm]
    cases = _load_cases(level, limit, held_out)
    if not cases:
        console.print("[red]no cases[/red] — run `teamclaw dataset build` first")
        raise typer.Exit(1)

    registry = build_registry(cfg, include_paid=config.allow_paid)
    context = RunnerContext(
        registry=registry, client=SecClient(cfg=cfg),
        runs_root=cfg.paths.runs / arm, workspaces_root=cfg.paths.workspaces / arm,
        arm=config, echo=echo,
    )
    harness = Harness(
        cases, arm=arm, level=level,
        conditions={"arm": config.to_json(), "providers": describe_registry(registry),
                    "held_out": held_out},
    )
    result = harness.run(make_case_runner(context), progress=True)
    path = result.save(cfg.paths.runs / f"eval_{arm}_{level}.json")

    headline = result.headline()
    table = Table("metric", "value", title=f"arm={arm} level={level}")
    for key, value in headline.items():
        if key != "metrics":
            table.add_row(key, str(value))
    for name, rate in headline["metrics"].items():
        table.add_row(f"  {name}", f"{rate:.4f}")
    console.print(table)
    if not headline["valid_measurement"]:
        console.print(f"[red]{headline['validity_note']}[/red]")
    console.print(f"saved -> {path}")


@app.command()
def ablate(
    level: str = typer.Option("l1", help="Task level: l1 or l2."),
    limit: Optional[int] = typer.Option(30, help="Cases per arm."),
    arms: Optional[str] = typer.Option(None, help="Comma-separated arm names."),
) -> None:
    """Run every arm and print the comparison table."""
    from teamclaw.evaluation.harness import Harness
    from teamclaw.evaluation.runner import ARMS, ARMS_BY_NAME, RunnerContext, make_case_runner
    from teamclaw.models.registry_default import build_registry
    from teamclaw.scenarios.dd_finance.sec_client import SecClient

    cfg = settings()
    selected = ([ARMS_BY_NAME[a.strip()] for a in arms.split(",")] if arms else list(ARMS))
    cases = _load_cases(level, limit, held_out=False)
    if not cases:
        console.print("[red]no cases[/red] — run `teamclaw dataset build` first")
        raise typer.Exit(1)

    rows: list[dict] = []
    for config in selected:
        console.print(f"\n[bold]arm: {config.name}[/bold] — {config.description}")
        registry = build_registry(cfg, include_paid=config.allow_paid)
        context = RunnerContext(
            registry=registry, client=SecClient(cfg=cfg),
            runs_root=cfg.paths.runs / config.name,
            workspaces_root=cfg.paths.workspaces / config.name, arm=config,
        )
        result = Harness(cases, arm=config.name, level=level,
                         conditions={"arm": config.to_json()}).run(
            make_case_runner(context), progress=True)
        result.save(cfg.paths.runs / f"eval_{config.name}_{level}.json")
        rows.append(result.headline())

    table = Table("arm", "headline", "citations", "consistency", "abstention",
                  "tokens", "cost", "steps", "valid", title=f"Ablation — level {level}")
    for row in rows:
        metrics = row["metrics"]
        table.add_row(
            row["arm"],
            f"{row['headline_rate']:.4f}" if row["headline_rate"] is not None else "-",
            f"{metrics.get('citation_verifiability', 0):.4f}",
            f"{metrics.get('calculation_consistency', 0):.4f}",
            f"{metrics.get('abstention_accuracy', 0):.4f}",
            f"{row['tokens_total']:,}", f"${row['cost_usd']:.4f}",
            str(row["mean_steps"]),
            "yes" if row["valid_measurement"] else "[red]no[/red]",
        )
    console.print(table)
    path = cfg.paths.runs / f"ablation_{level}.json"
    path.write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    console.print(f"saved -> {path}")


@app.command()
def run(
    ticker: str = typer.Argument(..., help="e.g. AAPL"),
    fiscal_year: int = typer.Argument(..., help="e.g. 2024"),
    level: str = typer.Option("l1", help="Task level: l1 or l2."),
    arm: str = typer.Option("full", help="Ablation arm name."),
) -> None:
    """Run a single case with a live trace, for debugging."""
    from teamclaw.evaluation.harness import score_case
    from teamclaw.evaluation.runner import ARMS_BY_NAME, RunnerContext, make_case_runner
    from teamclaw.models.registry_default import build_registry
    from teamclaw.scenarios.dd_finance.groundtruth import load_dataset
    from teamclaw.scenarios.dd_finance.sec_client import SecClient

    cfg = settings()
    case_id = f"{ticker.upper()}-FY{fiscal_year}"
    case = next((c for c in load_dataset(_dataset_path()) if c.case_id == case_id), None)
    if case is None:
        console.print(f"[red]{case_id} is not in the corpus[/red]")
        raise typer.Exit(1)

    config = ARMS_BY_NAME[arm]
    context = RunnerContext(
        registry=build_registry(cfg, include_paid=config.allow_paid),
        client=SecClient(cfg=cfg), runs_root=cfg.paths.runs / "single",
        workspaces_root=cfg.paths.workspaces / "single", arm=config, echo=True,
    )
    result = make_case_runner(context)(case, level)
    metrics = score_case(case, result.predicted, level=level,
                         source_text=result.source_text)

    table = Table("metric", "rate", "correct", "total", title=case_id)
    for name, metric in metrics.items():
        table.add_row(name, f"{metric.rate:.4f}", str(metric.correct), str(metric.total))
    console.print(table)
    console.print(result.run.to_json())
    if result.error:
        console.print(f"[red]{result.error}[/red]")


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
