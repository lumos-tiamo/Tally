"""The four automatic metrics, plus trace-level run metrics.

Numeric accuracy alone is not enough, and the reason is specific: a model can be
*right for the wrong reason* and *wrong in ways accuracy cannot see*. Each of the
other three exists to catch one of those.

``numeric_accuracy``
    Extracted values against XBRL truth, in three tolerance bands. The baseline
    metric, reported per band because a single band hides whether errors are
    rounding or nonsense.

``citation_verifiability``
    Does the quoted snippet actually appear in the source document, and does the
    number appear near it? This is the hallucination detector. A figure that is
    numerically correct but whose citation cannot be located was guessed — often
    from pretraining knowledge of the company — and that distinction is invisible
    to accuracy.

``calculation_consistency``
    Does ``value`` equal ``numerator / denominator`` as the model reported them?
    Catches two opposite failures accuracy conflates: right inputs with broken
    arithmetic, and correct-looking arithmetic over invented inputs.

``abstention_accuracy``
    For fields the filer genuinely does not disclose, did the model abstain? A
    bank has no current ratio. Scoring only on disclosed fields rewards a model
    that answers everything, which is precisely the behaviour that makes these
    systems unusable in diligence.

All four return counts alongside rates, because a rate over three cases is not a
measurement and the reader deserves to see the denominator.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from tally.scenarios.dd_finance.fields import (
    L1_BY_KEY,
    L2_BY_KEY,
    TOLERANCES,
    Tolerance,
)

# How close the number must appear to its quoted snippet, in characters.
CITATION_PROXIMITY_CHARS = 400
# Minimum quote length worth verifying; shorter snippets match by accident.
MIN_QUOTE_CHARS = 12


# --- normalisation --------------------------------------------------------
def normalise_text(text: str) -> str:
    """Collapse whitespace and unify unicode so quotes match filing HTML.

    Filings are full of non-breaking spaces, soft hyphens and curly quotes; a
    model that retypes a snippet will normalise them, and a naive substring test
    then reports a hallucination that did not happen.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("­", "").replace("​", "")
    text = re.sub(r"[‘’‚‛]", "'", text)
    text = re.sub(r"[“”„‟]", '"', text)
    text = re.sub(r"[‐-―]", "-", text)
    return re.sub(r"\s+", " ", text).strip().lower()


# Comma grouping only — never whitespace. A financial statement row reads
# "391,035  383,285  394,328" (three fiscal years side by side), and allowing
# spaces inside a number makes the regex swallow the whole row as the single
# value 391035383285394328. Every citation check against a table row then fails,
# which reads as a hallucination that never happened.
# The comma-grouped branch requires at least one group. With ``*`` it consumes
# the leading three digits of a bare four-digit number — every "2024" in a filing
# becomes (202, 4) — which silently breaks citation verification against any line
# containing a year.
_NUM = re.compile(
    r"-?\(?\$?\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?"
    r"|-?\(?\$?\s?\d+(?:\.\d+)?\)?"
)


def parse_number(raw: Any) -> float | None:
    """Parse a filing-style number: ``$1,234``, ``(567)`` as negative, ``1 234``."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = re.sub(r"[^\d.\-]", "", text)
    if cleaned in {"", "-", ".", "-."}:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative and value > 0 else value


def numbers_in(text: str) -> list[float]:
    out: list[float] = []
    for match in _NUM.finditer(text or ""):
        value = parse_number(match.group(0))
        if value is not None:
            out.append(value)
    return out


def scale_variants(value: float) -> list[float]:
    """A filing may state 391,035 for $391.035bn when the table is 'in millions'.

    So a reported figure is accepted if it matches truth at any of the usual
    presentation scales. Without this, every correctly-read value from a
    thousands-scaled table is counted wrong.
    """
    return [value, value * 1e3, value * 1e6, value * 1e9, value / 1e3, value / 1e6, value / 1e9]


# --- result containers ----------------------------------------------------
@dataclass
class MetricResult:
    name: str
    correct: int = 0
    total: int = 0
    detail: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def applicable(self) -> bool:
        """False when there was nothing to grade.

        Worth surfacing rather than hiding behind a rate of 0.0. A case where the
        figures show no material signal and the agent correctly reports none has
        an *undefined* recall, not a recall of zero — and reading 0.00 as failure
        would invert the interpretation of the best possible outcome. Pooling is
        unaffected either way: 0/0 contributes to neither numerator nor
        denominator under micro-averaging.
        """
        return self.total > 0

    @property
    def rate(self) -> float:
        return round(self.correct / self.total, 4) if self.total else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "metric": self.name,
            "rate": self.rate if self.applicable else None,
            "applicable": self.applicable,
            "correct": self.correct,
            "total": self.total,
            **self.detail,
            "failure_samples": self.failures[:8],
        }


# --- 1. numeric accuracy --------------------------------------------------
def numeric_accuracy(
    predicted: dict[str, Any],
    truth: dict[str, float | None],
    *,
    tolerances: Sequence[Tolerance] = TOLERANCES,
    allow_scale_variants: bool = True,
    label: str = "numeric_accuracy",
) -> dict[str, MetricResult]:
    """Score per tolerance band. Fields absent from truth are excluded here.

    Abstention is *not* graded in this metric — a field the filer did not
    disclose has no correct number, so including it would let a model raise its
    accuracy by abstaining. That case belongs to :func:`abstention_accuracy`.
    """
    results = {t.name: MetricResult(f"{label}@{t.name}") for t in tolerances}

    for key, expected in truth.items():
        if expected is None:
            continue
        got = _extract_value(predicted.get(key))
        for t in tolerances:
            res = results[t.name]
            res.total += 1
            if got is None:
                res.failures.append({"field": key, "expected": expected, "got": None,
                                     "why": "no value produced"})
                continue
            candidates = scale_variants(got) if allow_scale_variants else [got]
            if any(t.accepts(c, float(expected)) for c in candidates):
                res.correct += 1
            else:
                res.failures.append({
                    "field": key, "expected": expected, "got": got,
                    "relative_error": round(abs(got - float(expected)) / abs(float(expected)), 6)
                    if expected else None,
                })
    return results


def _extract_value(entry: Any) -> float | None:
    """Accept either a bare number or the ``{value, page, quote}`` contract."""
    if entry is None:
        return None
    if isinstance(entry, dict):
        return parse_number(entry.get("value"))
    return parse_number(entry)


# --- 2. citation verifiability -------------------------------------------
def citation_verifiability(
    predicted: dict[str, Any],
    source_text: str,
    *,
    proximity_chars: int = CITATION_PROXIMITY_CHARS,
) -> MetricResult:
    """Is the quote findable in the source, with the number near it?

    Two failure modes are separated in ``detail``: a quote that does not appear
    at all (fabricated citation) and a quote that appears but nowhere near the
    number (misattributed citation). The second is the subtler and more common
    one — the model finds real text and hangs the wrong figure on it.
    """
    result = MetricResult("citation_verifiability")
    haystack = normalise_text(source_text)
    counters = {"missing_quote": 0, "quote_not_found": 0, "number_far_from_quote": 0,
                "too_short": 0, "verified": 0}

    for key, entry in predicted.items():
        if not isinstance(entry, dict):
            continue
        value = parse_number(entry.get("value"))
        if value is None:
            continue  # an abstention carries no citation obligation
        result.total += 1

        quote = normalise_text(str(entry.get("quote") or ""))
        if not quote:
            counters["missing_quote"] += 1
            result.failures.append({"field": key, "why": "no quote supplied"})
            continue
        if len(quote) < MIN_QUOTE_CHARS:
            counters["too_short"] += 1
            result.failures.append({"field": key, "why": "quote too short to verify",
                                    "quote": quote})
            continue

        position = haystack.find(quote)
        if position < 0:
            counters["quote_not_found"] += 1
            result.failures.append({"field": key, "why": "quote not present in source",
                                    "quote": quote[:120]})
            continue

        window = haystack[
            max(0, position - proximity_chars) : position + len(quote) + proximity_chars
        ]
        wanted = scale_variants(value)
        found = any(
            any(math.isclose(n, w, rel_tol=0.005) for w in wanted) for n in numbers_in(window)
        )
        if found:
            counters["verified"] += 1
            result.correct += 1
        else:
            counters["number_far_from_quote"] += 1
            result.failures.append({
                "field": key, "why": "quote found but value not near it",
                "value": value, "quote": quote[:120],
            })

    result.detail = counters
    return result


# --- 3. calculation consistency ------------------------------------------
def calculation_consistency(
    predicted_ratios: dict[str, Any], *, rel_tol: float = 0.01
) -> MetricResult:
    """Does the reported value follow from the reported numerator/denominator?

    This is independent of whether the inputs were *right*: it checks internal
    coherence. Combined with numeric accuracy over L1, it separates "read the
    wrong number" from "did the wrong arithmetic".
    """
    result = MetricResult("calculation_consistency")
    counters = {"consistent": 0, "arithmetic_mismatch": 0, "missing_provenance": 0,
                "unknown_ratio": 0}

    for key, entry in predicted_ratios.items():
        if not isinstance(entry, dict):
            continue
        value = parse_number(entry.get("value"))
        if value is None:
            continue  # abstention: nothing to re-derive
        result.total += 1

        num = parse_number(entry.get("numerator"))
        den = parse_number(entry.get("denominator"))
        if num is None or den is None:
            counters["missing_provenance"] += 1
            result.failures.append({"ratio": key, "why": "numerator/denominator not reported"})
            continue

        rdef = L2_BY_KEY.get(key)
        if rdef is None:
            counters["unknown_ratio"] += 1
            result.failures.append({"ratio": key, "why": "not a defined L2 ratio"})
            continue

        if rdef.op.value == "ratio":
            if den == 0:
                counters["arithmetic_mismatch"] += 1
                result.failures.append({"ratio": key, "why": "zero denominator with a value"})
                continue
            expected = num / den
        else:
            expected = num - den

        if math.isclose(value, expected, rel_tol=rel_tol, abs_tol=1e-9):
            counters["consistent"] += 1
            result.correct += 1
        else:
            counters["arithmetic_mismatch"] += 1
            result.failures.append({
                "ratio": key, "reported": value, "recomputed": expected,
                "numerator": num, "denominator": den,
            })

    result.detail = counters
    return result


# --- 4. abstention accuracy ----------------------------------------------
def abstention_accuracy(
    predicted: dict[str, Any],
    truth: dict[str, float | None],
    *,
    absent_keys: Iterable[str] | None = None,
) -> MetricResult:
    """On fields the filer does not disclose, did the model decline to answer?

    Only fields whose absence is *trusted* should be passed in ``absent_keys``.
    Grading against absences that are really mapping failures would punish a
    model for finding something our resolver missed.
    """
    result = MetricResult("abstention_accuracy")
    keys = list(absent_keys) if absent_keys is not None else [
        k for k, v in truth.items() if v is None
    ]
    counters = {"correct_abstention": 0, "fabricated": 0}

    for key in keys:
        result.total += 1
        entry = predicted.get(key)
        value = _extract_value(entry)
        if value is None:
            counters["correct_abstention"] += 1
            result.correct += 1
            # A bare None passes, but an explicit reason is better practice and
            # worth surfacing separately rather than silently accepting.
            if isinstance(entry, dict) and not entry.get("reason"):
                result.detail.setdefault("abstained_without_reason", 0)
                result.detail["abstained_without_reason"] = (
                    result.detail.get("abstained_without_reason", 0) + 1
                )
        else:
            counters["fabricated"] += 1
            result.failures.append({
                "field": key, "fabricated_value": value,
                "why": "filer does not disclose this field",
                "label": L1_BY_KEY[key].label if key in L1_BY_KEY else key,
            })

    result.detail.update(counters)
    return result


# --- 5. signal recall (the L3 objective backbone) ------------------------
def signal_findings(
    predicted: dict[str, Any],
    present_keys: Sequence[str],
    detectable_keys: Sequence[str],
) -> tuple[MetricResult, MetricResult]:
    """Score L3 findings against the cross-year signals actually in the data.

    Returns ``(recall, precision)`` as separate results, because they answer
    different questions and an agent can be pathological in either direction:
    one that lists every possible concern scores perfect recall, and one that
    reports nothing scores perfect precision.

    ``detectable_keys`` is what makes this fair. A signal whose inputs the filer
    never disclosed could not have been found, so it is excluded from the recall
    denominator — otherwise a bank case penalises the agent for not reporting a
    current ratio it has no way to compute. And a *false* signal is only counted
    against precision when it was detectable and absent; asserting something
    uncomputable is scored as a fabrication either way.
    """
    reported = _reported_signal_keys(predicted)
    present = set(present_keys)
    detectable = set(detectable_keys)

    recall = MetricResult("signal_recall")
    for key in sorted(present & detectable):
        recall.total += 1
        if key in reported:
            recall.correct += 1
        else:
            recall.failures.append({"signal": key, "why": "present in the data but not reported"})

    precision = MetricResult("signal_precision")
    for key in sorted(reported):
        precision.total += 1
        if key in present:
            precision.correct += 1
        else:
            precision.failures.append({
                "signal": key,
                "why": "reported but not supported by the figures",
                "detectable": key in detectable,
            })

    detail = {"present": len(present & detectable), "reported": len(reported),
              "detectable": len(detectable)}
    # A case with nothing material to find, where the agent found nothing, is the
    # best possible outcome and has no rate. Recording it explicitly means the
    # pooled report can say how often the agent stayed correctly silent instead
    # of that fact vanishing into an empty denominator.
    detail["correctly_silent"] = int(not (present & detectable) and not reported)
    detail["fabricated_on_quiet_case"] = int(not (present & detectable) and bool(reported))
    recall.detail = dict(detail)
    precision.detail = dict(detail)
    return recall, precision


def _reported_signal_keys(predicted: dict[str, Any]) -> set[str]:
    """Extract signal keys from an L3 answer, by key or by prose keyword.

    Accepts the structured form first — ``{"findings": [{"signal": "...", ...}]}``
    — and falls back to keyword matching over free text. The fallback exists
    because a weak model often writes the right observation without using the
    key, and scoring that as a miss would measure format compliance rather than
    analysis.
    """
    from tally.scenarios.dd_finance.signals import SIGNALS, SIGNALS_BY_KEY

    keys: set[str] = set()

    findings = predicted.get("findings")
    if isinstance(findings, list):
        for entry in findings:
            if isinstance(entry, dict):
                key = str(entry.get("signal") or entry.get("key") or "")
                if key in SIGNALS_BY_KEY:
                    keys.add(key)

    blob = normalise_text(json.dumps(predicted, ensure_ascii=False, default=str))
    for definition in SIGNALS:
        if definition.key in keys:
            continue
        if normalise_text(definition.key.replace("_", " ")) in blob:
            keys.add(definition.key)
            continue
        # Two distinct keywords, so a passing mention of "cash flow" in an
        # unrelated sentence does not count as having found the divergence.
        hits = sum(1 for kw in definition.keywords if normalise_text(kw) in blob)
        if hits >= 2:
            keys.add(definition.key)
    return keys


# --- trace-level metrics --------------------------------------------------
@dataclass
class RunMetrics:
    steps: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    sandbox_execs: int = 0
    sandbox_failures: int = 0
    recovered_steps: int = 0
    failed_steps: int = 0
    compactions: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    peak_utilisation: float = 0.0
    finished: bool = False

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def sandbox_failure_rate(self) -> float:
        return round(self.sandbox_failures / self.sandbox_execs, 4) if self.sandbox_execs else 0.0

    @property
    def tool_failure_rate(self) -> float:
        return round(self.tool_failures / self.tool_calls, 4) if self.tool_calls else 0.0

    @property
    def recovery_rate(self) -> float:
        """Of the steps that hit an error, how many went on to succeed."""
        eligible = self.recovered_steps + self.failed_steps
        return round(self.recovered_steps / eligible, 4) if eligible else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "finished": self.finished,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_total": self.tokens_total,
            "cost_usd": round(self.cost_usd, 6),
            "duration_s": round(self.duration_s, 2),
            "sandbox_execs": self.sandbox_execs,
            "sandbox_failure_rate": self.sandbox_failure_rate,
            "recovery_rate": self.recovery_rate,
            "compactions": self.compactions,
            "tool_calls": self.tool_calls,
            "tool_failure_rate": self.tool_failure_rate,
            "peak_utilisation": round(self.peak_utilisation, 4),
        }


def aggregate(results: Sequence[MetricResult]) -> MetricResult:
    """Pool several per-case MetricResults into one. Micro-average.

    Micro rather than macro: a case with twelve gradable fields should weigh more
    than one with three, and macro-averaging over cases with wildly different
    field counts flatters small cases.
    """
    if not results:
        return MetricResult("empty")
    out = MetricResult(results[0].name)
    merged_detail: dict[str, Any] = {}
    for r in results:
        out.correct += r.correct
        out.total += r.total
        out.failures.extend(r.failures)
        for k, v in r.detail.items():
            if isinstance(v, (int, float)):
                merged_detail[k] = merged_detail.get(k, 0) + v
    out.detail = merged_detail
    out.detail["cases"] = len(results)
    return out
