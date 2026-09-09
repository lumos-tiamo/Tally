"""Derive the L3 signal ground truth from the L1 corpus and commit it.

The signal set for a company-year is a deterministic function of two years of L1
values, so it does not need to be stored — but it is, for two reasons. It makes
the L3 truth reviewable without running any code, and it pins the L3 scores to a
signal definition at a point in time: changing a threshold changes the truth, and
a committed artefact makes that visible in a diff instead of silently moving every
score.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tally.config import settings
from tally.scenarios.dd_finance.groundtruth import load_dataset
from tally.scenarios.dd_finance.signals import SIGNALS, detect, detectable


def main() -> int:
    cfg = settings()
    cases = load_dataset(cfg.paths.datasets / "dd_finance_groundtruth.jsonl")
    if not cases:
        print("no L1 corpus; run `tally dataset build` first")
        return 1

    by_ticker: dict[str, list] = {}
    for case in cases:
        by_ticker.setdefault(case.ticker, []).append(case)

    rows: list[dict] = []
    fired: Counter[str] = Counter()
    possible: Counter[str] = Counter()
    for group in by_ticker.values():
        group.sort(key=lambda c: c.fiscal_year)
        for previous, current in zip(group, group[1:]):
            found = detect(current.l1, previous.l1)
            able = detectable(current.l1, previous.l1)
            fired.update(s.key for s in found)
            possible.update(able)
            rows.append({
                "case_id": current.case_id,
                "ticker": current.ticker,
                "sector": current.sector,
                "held_out": current.held_out,
                "fiscal_year": current.fiscal_year,
                "prior_fiscal_year": previous.fiscal_year,
                "signals_present": [s.to_json() for s in found],
                "signals_detectable": able,
            })

    quiet = sum(1 for r in rows if not r["signals_present"])
    report = {
        "pairs": len(rows),
        "companies": len(by_ticker),
        "signals_total": int(sum(fired.values())),
        "mean_signals_per_pair": round(sum(fired.values()) / len(rows), 3) if rows else 0.0,
        "quiet_pairs": quiet,
        "quiet_share": round(quiet / len(rows), 3) if rows else 0.0,
        "by_signal": {
            definition.key: {
                "severity": definition.severity.value,
                "fired": fired[definition.key],
                "detectable": possible[definition.key],
                "fire_rate": round(fired[definition.key] / possible[definition.key], 4)
                if possible[definition.key] else 0.0,
            }
            for definition in SIGNALS
        },
        "note": ("Quiet pairs are not filler: they are the cases that test whether "
                 "the agent avoids inventing findings, and they carry no recall "
                 "denominator by design."),
    }

    out = cfg.paths.datasets / "dd_finance_l3_signals.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out.parent / f"{out.stem}.summary.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )

    print(f"pairs {report['pairs']}  signals {report['signals_total']}  "
          f"mean/pair {report['mean_signals_per_pair']}  quiet {quiet} "
          f"({report['quiet_share']:.0%})")
    print(f"\n{'signal':<32} {'sev':<9} {'fired':>6} {'detectable':>11} {'rate':>7}")
    for key, body in report["by_signal"].items():
        print(f"{key:<32} {body['severity']:<9} {body['fired']:>6} "
              f"{body['detectable']:>11} {body['fire_rate']:>7.2%}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
