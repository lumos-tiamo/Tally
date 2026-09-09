"""Eval harness: run cases, score them, refuse to publish nonsense.

The harness owns three responsibilities that are easy to get wrong.

**It refuses to report scores that are not measurements.** Two ways a run can
fail to be one, and both were found by running it. ``FakeProvider`` and
``ScriptedProvider`` exist so the platform can be exercised offline, and a
plumbing check that looks like a 100% accuracy result is worse than no result.
Less obviously, a run where *no model was reachable at all* produces a table of
zeros that reads exactly like a real score of zero — the first end-to-end
invocation of ``teamclaw eval`` with no credentials configured reported
``valid_measurement: True`` alongside zero tokens and zero steps. So
:meth:`EvalRun.publishable` requires both no test doubles *and* evidence that
model calls actually happened.

**It records the conditions, not just the numbers.** Sandbox isolation level,
provider mix, arm configuration, dataset digest. A score without its conditions
cannot be compared to another score, and ablation arms differ precisely in
conditions.

**It scores abstention against *trusted* absences only.** A field our concept
mapper failed to find is not a field the filer omitted, and grading the agent on
our own gaps would punish it for being right.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from teamclaw.evaluation.metrics import (
    MetricResult,
    RunMetrics,
    abstention_accuracy,
    aggregate,
    calculation_consistency,
    citation_verifiability,
    numeric_accuracy,
    signal_findings,
)
from teamclaw.observability.trace import SpanKind, Tracer
from teamclaw.orchestration.agent import RunResult
from teamclaw.scenarios.dd_finance.fields import (
    L1_HEADLINE_TOLERANCE,
    L2_HEADLINE_TOLERANCE,
)
from teamclaw.scenarios.dd_finance.groundtruth import GroundTruthCase

FAKE_PROVIDERS = frozenset({"fake", "scripted"})


@dataclass
class CaseOutcome:
    case_id: str
    level: str
    finished: bool
    predicted: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    run: RunMetrics = field(default_factory=RunMetrics)
    error: str = ""
    providers_used: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "level": self.level,
            "finished": self.finished,
            "error": self.error[:300],
            "providers_used": list(self.providers_used),
            "metrics": {k: v.to_json() for k, v in self.metrics.items()},
            "run": self.run.to_json(),
        }


@dataclass
class EvalRun:
    arm: str
    level: str
    started_at: float = field(default_factory=time.time)
    outcomes: list[CaseOutcome] = field(default_factory=list)
    conditions: dict[str, Any] = field(default_factory=dict)
    dataset_digest: str = ""
    notes: list[str] = field(default_factory=list)

    # -- validity ----------------------------------------------------------
    @property
    def providers_used(self) -> set[str]:
        used: set[str] = set()
        for o in self.outcomes:
            used.update(o.providers_used)
        return used

    @property
    def used_fake_provider(self) -> bool:
        return bool(self.providers_used & FAKE_PROVIDERS)

    @property
    def model_calls(self) -> int:
        return sum(o.run.steps for o in self.outcomes)

    @property
    def tokens_total(self) -> int:
        return sum(o.run.tokens_total for o in self.outcomes)

    @property
    def publishable(self) -> bool:
        """A run is a measurement only if a real model actually did the work."""
        return (
            bool(self.outcomes)
            and not self.used_fake_provider
            and self.tokens_total > 0
        )

    def invalidity_reason(self) -> str:
        if not self.outcomes:
            return "no cases were run"
        if self.used_fake_provider:
            return (
                "a fake/scripted provider served at least one call. These figures "
                "verify plumbing only and must never be reported as eval results."
            )
        if self.tokens_total == 0:
            errors = sorted({o.error.split(":")[0] for o in self.outcomes if o.error})
            detail = f" ({', '.join(errors)})" if errors else ""
            return (
                "no model tokens were consumed, so nothing was actually evaluated"
                f"{detail}. A table of zeros here is an unconfigured run, not a "
                "score of zero."
            )
        return ""

    # -- aggregation -------------------------------------------------------
    def pooled(self, metric: str) -> MetricResult:
        return aggregate([o.metrics[metric] for o in self.outcomes if metric in o.metrics])

    def metric_names(self) -> list[str]:
        names: list[str] = []
        for o in self.outcomes:
            for k in o.metrics:
                if k not in names:
                    names.append(k)
        return names

    def headline(self) -> dict[str, Any]:
        if self.level == "l3":
            key = "signal_recall"
        else:
            tolerance = (L1_HEADLINE_TOLERANCE if self.level == "l1"
                         else L2_HEADLINE_TOLERANCE)
            key = f"numeric_accuracy@{tolerance}"
        pooled = {name: self.pooled(name) for name in self.metric_names()}
        return {
            "arm": self.arm,
            "level": self.level,
            "cases": len(self.outcomes),
            "completed": sum(1 for o in self.outcomes if o.finished),
            "headline_metric": key,
            "headline_rate": pooled[key].rate if key in pooled else None,
            "metrics": {name: r.rate for name, r in pooled.items()},
            "tokens_total": sum(o.run.tokens_total for o in self.outcomes),
            "cost_usd": round(sum(o.run.cost_usd for o in self.outcomes), 6),
            "mean_steps": round(
                sum(o.run.steps for o in self.outcomes) / len(self.outcomes), 2
            ) if self.outcomes else 0.0,
            "recovery_rate": _pooled_recovery(self.outcomes),
            "valid_measurement": self.publishable,
            "validity_note": (
                "" if self.publishable
                else f"NOT A MEASUREMENT: {self.invalidity_reason()}"
            ),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "level": self.level,
            "started_at": self.started_at,
            "dataset_digest": self.dataset_digest,
            "conditions": self.conditions,
            "headline": self.headline(),
            "pooled_metrics": {n: self.pooled(n).to_json() for n in self.metric_names()},
            "cases": [o.to_json() for o in self.outcomes],
            "notes": self.notes,
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
        )
        return path


def _pooled_recovery(outcomes: Sequence[CaseOutcome]) -> float:
    recovered = sum(o.run.recovered_steps for o in outcomes)
    failed = sum(o.run.failed_steps for o in outcomes)
    total = recovered + failed
    return round(recovered / total, 4) if total else 0.0


def dataset_digest(cases: Sequence[GroundTruthCase]) -> str:
    """Content hash of the truth set, so a score is pinned to its corpus."""
    h = hashlib.sha256()
    for c in sorted(cases, key=lambda x: x.case_id):
        h.update(c.case_id.encode())
        h.update(json.dumps(c.l1, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def score_case(
    case: GroundTruthCase,
    predicted: dict[str, Any],
    *,
    level: str,
    source_text: str = "",
    signals_present: Sequence[str] = (),
    signals_detectable: Sequence[str] = (),
) -> dict[str, MetricResult]:
    """Apply the metrics appropriate to the task level."""
    metrics: dict[str, MetricResult] = {}

    if level == "l1":
        for band, res in numeric_accuracy(predicted, case.l1).items():
            metrics[f"numeric_accuracy@{band}"] = res
        # Only trusted absences are graded: an untrusted absence may be our gap.
        metrics["abstention_accuracy"] = abstention_accuracy(
            predicted, case.l1, absent_keys=case.l1_absence_trusted
        )
        if source_text:
            metrics["citation_verifiability"] = citation_verifiability(predicted, source_text)

    elif level == "l2":
        truth = {k: v for k, v in case.l2.items() if v is not None}
        for band, res in numeric_accuracy(predicted, truth, label="numeric_accuracy").items():
            metrics[f"numeric_accuracy@{band}"] = res
        metrics["calculation_consistency"] = calculation_consistency(predicted)
        undefined = [k for k, v in case.l2.items() if v is None]
        metrics["abstention_accuracy"] = abstention_accuracy(
            predicted, case.l2, absent_keys=undefined
        )

    elif level == "l3":
        # The objective backbone. The judge grades the prose separately; these
        # two say whether the agent noticed what the numbers actually show.
        recall, precision = signal_findings(
            predicted, signals_present, signals_detectable
        )
        metrics["signal_recall"] = recall
        metrics["signal_precision"] = precision

    return metrics


def run_metrics_from(run: RunResult, tracer: Tracer, accountant) -> RunMetrics:  # noqa: ANN001
    """Fold a run's trace and accounting into the reported run metrics."""
    totals = accountant.total()
    sandbox_spans = tracer.of_kind(SpanKind.SANDBOX_EXEC)
    tool_spans = tracer.of_kind(SpanKind.TOOL_CALL)
    return RunMetrics(
        steps=run.step_count,
        finished=run.finished,
        tokens_in=totals.tokens_in,
        tokens_out=totals.tokens_out,
        cost_usd=totals.cost_usd,
        duration_s=run.duration_s,
        sandbox_execs=len(sandbox_spans),
        sandbox_failures=sum(1 for s in sandbox_spans if s.attrs.get("exit_code", 0) != 0),
        recovered_steps=run.recovered_steps,
        failed_steps=run.failed_steps,
        compactions=len(tracer.of_kind(SpanKind.COMPACTION)),
        tool_calls=len(tool_spans),
        tool_failures=sum(1 for s in tool_spans if not s.attrs.get("ok", True)),
        peak_utilisation=run.peak_utilisation,
    )


CaseRunner = Callable[[GroundTruthCase, str], "CaseRunResult"]


@dataclass
class CaseRunResult:
    """What a runner hands back: the parsed prediction plus run telemetry."""

    predicted: dict[str, Any]
    run: RunMetrics
    finished: bool
    providers_used: tuple[str, ...] = ()
    source_text: str = ""
    error: str = ""
    # L3 only: the cross-year signals present in the data, and those the data
    # could support at all. The runner computes them because it holds the prior
    # year's case.
    signals_present: tuple[str, ...] = ()
    signals_detectable: tuple[str, ...] = ()


class Harness:
    def __init__(
        self,
        cases: Sequence[GroundTruthCase],
        *,
        arm: str,
        level: str,
        conditions: dict[str, Any] | None = None,
    ) -> None:
        self.cases = list(cases)
        self.arm = arm
        self.level = level
        self.conditions = dict(conditions or {})

    def run(self, runner: CaseRunner, *, limit: int | None = None, progress: bool = False) -> EvalRun:
        selected = self.cases[:limit] if limit else self.cases
        out = EvalRun(
            arm=self.arm, level=self.level,
            conditions=self.conditions, dataset_digest=dataset_digest(selected),
        )
        for n, case in enumerate(selected, 1):
            if progress:
                print(f"[{n:>3}/{len(selected)}] {case.case_id}", flush=True)
            try:
                result = runner(case, self.level)
            except Exception as exc:  # noqa: BLE001 - one bad case must not stop the arm
                out.outcomes.append(
                    CaseOutcome(
                        case_id=case.case_id, level=self.level, finished=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            metrics = score_case(
                case, result.predicted, level=self.level,
                source_text=result.source_text,
                signals_present=result.signals_present,
                signals_detectable=result.signals_detectable,
            )
            out.outcomes.append(
                CaseOutcome(
                    case_id=case.case_id, level=self.level, finished=result.finished,
                    predicted=result.predicted, metrics=metrics, run=result.run,
                    error=result.error, providers_used=result.providers_used,
                )
            )
        return out
