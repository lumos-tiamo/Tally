"""Read-only endpoints over what the project measured.

These exist so the dashboard can show the eval work rather than describing it:
the corpus, the zero-model floor, the tool-cost curve, the ablation arms and the
gap decomposition. Everything is read from the committed artefacts in
``results/`` and ``data/datasets/`` — the dashboard never recomputes a score,
because a number shown next to a claim should be the same number the repository
can be diffed against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from teamclaw.api.deps import AppState, app_state

router = APIRouter(prefix="/api/insight", tags=["insight"])


def _read(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


@router.get("/overview")
def overview(state: AppState = Depends(app_state)) -> dict[str, Any]:
    root = state.cfg.paths.root
    corpus = _read(state.cfg.paths.datasets / "dd_finance_groundtruth.summary.json")
    signals = _read(state.cfg.paths.datasets / "dd_finance_l3_signals.summary.json")
    floor = _read(root / "results" / "deterministic_baseline_l1.json")
    return {
        "totals": state.store.totals(),
        "corpus": {
            "cases": (corpus or {}).get("usable"),
            "held_out": (corpus or {}).get("held_out"),
            "quarantined": (corpus or {}).get("quarantined"),
            "mean_l1_coverage": (corpus or {}).get("mean_l1_coverage"),
            "by_sector": (corpus or {}).get("by_sector", {}),
            "abstention_targets": (corpus or {}).get("abstention_targets", {}),
        } if corpus else None,
        "l3_signals": {
            "pairs": (signals or {}).get("pairs"),
            "signals_total": (signals or {}).get("signals_total"),
            "quiet_pairs": (signals or {}).get("quiet_pairs"),
            "quiet_share": (signals or {}).get("quiet_share"),
            "by_signal": (signals or {}).get("by_signal", {}),
        } if signals else None,
        "zero_model_floor": {
            "numeric_strict": (floor or {}).get("numeric_accuracy_strict", {}).get("rate"),
            "citation": (floor or {}).get("citation_verifiability", {}).get("rate"),
            "abstention": (floor or {}).get("abstention_accuracy", {}).get("rate"),
            "cases": (floor or {}).get("cases_scored"),
        } if floor else None,
    }


@router.get("/tool-cost")
def tool_cost(state: AppState = Depends(app_state)) -> dict[str, Any]:
    """Retrieval cost is flat in registry size; both alternatives are linear."""
    return _read(state.cfg.paths.root / "results" / "tool_retrieval_scaling.json") or {
        "rows": [], "note": "not measured yet — run scripts/tool_retrieval_scaling.py"
    }


@router.get("/arms")
def arms(state: AppState = Depends(app_state)) -> dict[str, Any]:
    """Ablation arms: their configuration, and any results already on disk."""
    from teamclaw.evaluation.runner import ARMS

    results_dir = state.cfg.paths.root / "results"
    return {
        "arms": [
            {
                **arm.to_json(),
                "result": (_read(results_dir / f"{arm.name.replace('-', '_')}_l1.json")
                           or {}).get("headline"),
            }
            for arm in ARMS
        ],
        "context_cost": _read(results_dir / "arm_context_cost.json"),
        "gap_decomposition": _read(results_dir / "workflow_c_gap_decomposition.json"),
    }


@router.get("/health")
def health(state: AppState = Depends(app_state)) -> dict[str, Any]:
    return {**state.health(), "config_warnings": state.cfg.api_security_warnings()}
