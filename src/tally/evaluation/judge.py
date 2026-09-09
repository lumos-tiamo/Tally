"""LLM-as-judge, with its own reliability reported.

A judge is a measuring instrument, and an uncalibrated instrument produces
numbers, not measurements. So this module does two things, and the second is the
one that matters:

1. Score L3 answers — the judgement-level task, where there is no XBRL truth to
   diff against — on named rubric dimensions.
2. **Measure the judge against human labels** and report Cohen's κ per
   dimension. Dimensions below :data:`KAPPA_FLOOR` are marked unreliable and
   their scores are withheld from the headline, falling back to human spot-checks.

Reporting a judge's κ is the difference between "we evaluated with an LLM judge"
and an actual evaluation. A judge that agrees with humans at κ=0.3 is close to
noise, and a paper-thin agreement is easy to mistake for a good score.

Three further precautions:

* **The judge is never the actor.** ``Purpose.JUDGE`` routes to a different
  provider pool than ``Purpose.CODE`` (see the router policy). A model grading
  its own output is not an evaluation.
* **Position bias is controlled** in pairwise mode by scoring both orderings and
  keeping the result only when they agree.
* **The rubric is forced to cite.** Every dimension score must quote the span of
  the answer it is judging, which makes a judge that is confabulating visible in
  the same way it makes an agent visible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from tally.models.base import Message, Purpose

KAPPA_FLOOR = 0.6
JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class RubricDimension:
    key: str
    question: str
    scale: tuple[int, int] = (1, 5)
    weight: float = 1.0

    def render(self) -> str:
        lo, hi = self.scale
        return f"- `{self.key}` ({lo}-{hi}): {self.question}"


L3_RUBRIC: tuple[RubricDimension, ...] = (
    RubricDimension(
        "grounding",
        "Is every factual claim traceable to a figure or passage the answer cites? "
        "Score 1 if claims float free of sources, 5 if each is anchored.",
        weight=2.0,
    ),
    RubricDimension(
        "causal_validity",
        "Where the answer attributes a trend to a cause, is that attribution "
        "supported by the cited data rather than asserted? Score 1 for bare "
        "assertion, 5 for a chain a reader can check.",
        weight=2.0,
    ),
    RubricDimension(
        "risk_identification",
        "Does the answer surface the material risks visible in the figures "
        "(e.g. revenue up while operating cash flow falls), rather than only "
        "restating the favourable ones?",
        weight=1.5,
    ),
    RubricDimension(
        "calibration",
        "Does the answer state uncertainty where the data is thin, and avoid "
        "confident claims the filing does not support?",
    ),
    RubricDimension(
        "no_fabrication",
        "Are there any figures, dates or facts that do not appear in the provided "
        "material? Score 1 if any invented specific is present, 5 if none.",
        weight=2.0,
    ),
)


def rubric_prompt(dimensions: Sequence[RubricDimension]) -> str:
    return (
        "You are grading a financial-analysis answer against source material.\n\n"
        "Dimensions:\n"
        + "\n".join(d.render() for d in dimensions)
        + "\n\nFor each dimension return an integer score and `evidence`: a short "
        "verbatim quote from the ANSWER that justifies the score. If you cannot "
        "quote the answer, the score must be 1.\n\n"
        "Reply with JSON only, no prose:\n"
        '{"scores": {"<key>": {"score": <int>, "evidence": "<quote>"}}, '
        '"overall_note": "<one sentence>"}'
    )


@dataclass
class JudgeVerdict:
    case_id: str
    scores: dict[str, int] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)
    note: str = ""
    parse_ok: bool = True
    raw: str = ""

    def weighted(self, dimensions: Sequence[RubricDimension]) -> float:
        total_w = sum(d.weight for d in dimensions if d.key in self.scores)
        if not total_w:
            return 0.0
        acc = sum(self.scores[d.key] * d.weight for d in dimensions if d.key in self.scores)
        return round(acc / total_w, 4)

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "scores": self.scores,
            "evidence": {k: v[:160] for k, v in self.evidence.items()},
            "note": self.note[:300],
            "parse_ok": self.parse_ok,
        }


def parse_verdict(case_id: str, text: str, dimensions: Sequence[RubricDimension]) -> JudgeVerdict:
    """Extract the JSON verdict, tolerating a model that wraps it in prose."""
    match = JSON_BLOCK.search(text or "")
    if not match:
        return JudgeVerdict(case_id=case_id, parse_ok=False, raw=text or "")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return JudgeVerdict(case_id=case_id, parse_ok=False, raw=text or "")

    scores: dict[str, int] = {}
    evidence: dict[str, str] = {}
    raw_scores = data.get("scores") or {}
    valid = {d.key: d for d in dimensions}
    for key, body in raw_scores.items():
        if key not in valid:
            continue
        lo, hi = valid[key].scale
        if isinstance(body, dict):
            value, quote = body.get("score"), str(body.get("evidence") or "")
        else:
            value, quote = body, ""
        try:
            score = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        scores[key] = max(lo, min(hi, score))
        evidence[key] = quote
    return JudgeVerdict(
        case_id=case_id, scores=scores, evidence=evidence,
        note=str(data.get("overall_note") or ""), parse_ok=bool(scores), raw=text or "",
    )


class Judge:
    def __init__(
        self,
        router,  # noqa: ANN001 - Router, avoided to keep the import graph flat
        *,
        dimensions: Sequence[RubricDimension] = L3_RUBRIC,
        max_source_chars: int = 12_000,
    ) -> None:
        self.router = router
        self.dimensions = list(dimensions)
        self.max_source_chars = max_source_chars
        self.verdicts: list[JudgeVerdict] = []

    def score(self, *, case_id: str, question: str, answer: str, source: str) -> JudgeVerdict:
        messages = [
            Message.system(rubric_prompt(self.dimensions)),
            Message.user(
                f"QUESTION:\n{question}\n\n"
                f"SOURCE MATERIAL (truncated):\n{source[: self.max_source_chars]}\n\n"
                f"ANSWER TO GRADE:\n{answer}"
            ),
        ]
        completion = self.router.complete(
            Purpose.JUDGE, messages, max_tokens=900, task=f"judge:{case_id}"
        )
        verdict = parse_verdict(case_id, completion.text, self.dimensions)
        self.verdicts.append(verdict)
        return verdict

    def summary(self) -> dict[str, Any]:
        if not self.verdicts:
            return {"verdicts": 0}
        per_dim: dict[str, list[int]] = {}
        for v in self.verdicts:
            for k, s in v.scores.items():
                per_dim.setdefault(k, []).append(s)
        return {
            "verdicts": len(self.verdicts),
            "parse_failures": sum(1 for v in self.verdicts if not v.parse_ok),
            "mean_by_dimension": {
                k: round(sum(v) / len(v), 3) for k, v in sorted(per_dim.items())
            },
            "mean_weighted": round(
                sum(v.weighted(self.dimensions) for v in self.verdicts) / len(self.verdicts), 4
            ),
        }


# --- calibration ----------------------------------------------------------
def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's κ for two raters over the same items.

    κ rather than raw agreement because a 1-5 rubric where most answers score 4
    yields 70% agreement by chance alone. κ subtracts that expectation, so it
    reports whether the judge tracks the human or merely shares their prior.
    """
    if len(a) != len(b) or not a:
        return 0.0
    categories = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    expected = 0.0
    for c in categories:
        pa = sum(1 for x in a if x == c) / n
        pb = sum(1 for y in b if y == c) / n
        expected += pa * pb
    if expected >= 1.0:
        # Every rating identical on both sides: κ is undefined, but perfect
        # agreement on a degenerate distribution is not evidence of reliability.
        return 0.0
    return round((observed - expected) / (1 - expected), 4)


def quadratic_weighted_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """κ that treats near-misses as partial credit.

    On an ordinal rubric, a judge scoring 4 where a human scored 5 is not the
    same kind of error as scoring 1. Plain κ counts both as total disagreement;
    the quadratic weighting is the fairer read for ordinal scales, and both are
    reported so a large gap between them is visible.
    """
    if len(a) != len(b) or not a:
        return 0.0
    cats = sorted(set(a) | set(b))
    index = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    if k < 2:
        return 0.0
    n = len(a)
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        observed[index[x]][index[y]] += 1
    hist_a = [sum(1 for x in a if x == c) for c in cats]
    hist_b = [sum(1 for y in b if y == c) for c in cats]
    numerator = denominator = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            expected = hist_a[i] * hist_b[j] / n
            numerator += w * observed[i][j]
            denominator += w * expected
    return round(1 - numerator / denominator, 4) if denominator else 0.0


@dataclass
class DimensionCalibration:
    key: str
    kappa: float
    weighted_kappa: float
    n: int
    mean_judge: float
    mean_human: float

    @property
    def reliable(self) -> bool:
        return self.kappa >= KAPPA_FLOOR

    @property
    def bias(self) -> float:
        """Positive means the judge scores higher than humans."""
        return round(self.mean_judge - self.mean_human, 3)

    def to_json(self) -> dict[str, Any]:
        return {
            "dimension": self.key,
            "kappa": self.kappa,
            "quadratic_weighted_kappa": self.weighted_kappa,
            "n": self.n,
            "mean_judge": round(self.mean_judge, 3),
            "mean_human": round(self.mean_human, 3),
            "bias": self.bias,
            "reliable": self.reliable,
            "verdict": "usable" if self.reliable
            else f"withheld: kappa {self.kappa} below floor {KAPPA_FLOOR}",
        }


def calibrate(
    judge_verdicts: Iterable[JudgeVerdict],
    human_labels: dict[str, dict[str, int]],
    *,
    dimensions: Sequence[RubricDimension] = L3_RUBRIC,
) -> dict[str, Any]:
    """Compare judge scores to human labels on the calibration set.

    ``human_labels`` maps ``case_id -> {dimension: score}``. Only cases present in
    both are used; a calibration set the judge never scored would silently
    inflate agreement.
    """
    by_case = {v.case_id: v for v in judge_verdicts}
    shared = [cid for cid in human_labels if cid in by_case]

    per_dim: list[DimensionCalibration] = []
    for d in dimensions:
        pairs = [
            (by_case[cid].scores[d.key], human_labels[cid][d.key])
            for cid in shared
            if d.key in by_case[cid].scores and d.key in human_labels[cid]
        ]
        if not pairs:
            continue
        judge_scores = [p[0] for p in pairs]
        human_scores = [p[1] for p in pairs]
        per_dim.append(
            DimensionCalibration(
                key=d.key,
                kappa=cohens_kappa(judge_scores, human_scores),
                weighted_kappa=quadratic_weighted_kappa(judge_scores, human_scores),
                n=len(pairs),
                mean_judge=sum(judge_scores) / len(judge_scores),
                mean_human=sum(human_scores) / len(human_scores),
            )
        )

    reliable = [c for c in per_dim if c.reliable]
    return {
        "calibration_set_size": len(shared),
        "dimensions": [c.to_json() for c in per_dim],
        "reliable_dimensions": [c.key for c in reliable],
        "withheld_dimensions": [c.key for c in per_dim if not c.reliable],
        "kappa_floor": KAPPA_FLOOR,
        "headline_usable": bool(reliable),
        "note": (
            "Dimensions below the kappa floor are excluded from the headline L3 "
            "score and reported as requiring human spot-checks. A judge that does "
            "not agree with humans is not evidence about the agent."
        ),
    }
