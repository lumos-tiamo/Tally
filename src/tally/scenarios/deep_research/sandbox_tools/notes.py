"""In-sandbox research note-keeping with source provenance.

The research scenario's distinctive problem is not retrieval, it is *keeping
claims attached to sources* across many parallel sub-agents. So the note store
is the tool: a claim cannot be recorded without a source id, and the report
builder refuses to emit any claim whose source is unknown.

That inversion — the tool enforces provenance rather than the prompt asking for
it — is what makes citation discipline survive a weak model.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

STORE = "research/notes.jsonl"
SOURCES = "research/sources.json"

# Crude credibility tiers. Deliberately explicit rather than learned: a research
# agent that cannot say why it trusted something has not done research.
TIERS = {
    "primary": 1.0,     # filings, official statistics, standards bodies
    "reported": 0.7,    # established outlets reporting primary material
    "secondary": 0.5,   # analysis, blogs with named authors
    "unattributed": 0.2,  # forums, anonymous posts, aggregators
}


def _path(root: str, rel: str) -> Path:
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def add_source(root: str, url: str, title: str, tier: str = "secondary") -> dict:
    """Register a source and get back the id that claims must cite."""
    if tier not in TIERS:
        return {"error": f"unknown tier {tier!r}", "known": sorted(TIERS)}
    source_id = "s_" + hashlib.sha256(url.encode()).hexdigest()[:8]
    path = _path(root, SOURCES)
    sources = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    sources[source_id] = {"url": url, "title": title, "tier": tier,
                          "weight": TIERS[tier]}
    path.write_text(json.dumps(sources, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"source_id": source_id, "tier": tier, "weight": TIERS[tier],
            "total_sources": len(sources)}


def add_claim(root: str, claim: str, source_id: str, quote: str = "") -> dict:
    """Record a claim. Refused if the source is not registered.

    The refusal is the feature: an unsourced claim cannot enter the store, so it
    cannot reach the report, so the report cannot contain one.
    """
    sources_path = _path(root, SOURCES)
    sources = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() else {}
    if source_id not in sources:
        return {"error": f"unknown source_id {source_id!r}; call add_source first",
                "known": sorted(sources)}
    if len(quote.strip()) < 8:
        return {"error": "a quote of at least 8 characters is required so the "
                         "claim can be checked against its source"}
    record = {"claim": claim.strip(), "source_id": source_id,
              "quote": quote.strip()[:400], "tier": sources[source_id]["tier"]}
    with _path(root, STORE).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"recorded": True, "claim_chars": len(record["claim"]), "tier": record["tier"]}


def load(root: str) -> dict:
    """All claims and sources. Returns counts plus the claims, not raw pages."""
    store, sources_path = _path(root, STORE), _path(root, SOURCES)
    claims = []
    if store.exists():
        for line in store.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    claims.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    sources = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() else {}
    by_tier: dict[str, int] = {}
    for c in claims:
        by_tier[c["tier"]] = by_tier.get(c["tier"], 0) + 1
    return {"claims": claims, "sources": sources,
            "claim_count": len(claims), "source_count": len(sources),
            "claims_by_tier": by_tier}


def corroboration(root: str) -> dict:
    """Which claims are supported by more than one independent source.

    Near-duplicate claims are matched on a normalised token set rather than exact
    text, because two sources never phrase a finding identically and exact
    matching would report every claim as uncorroborated.
    """
    data = load(root)
    groups: list[dict] = []
    for claim in data["claims"]:
        tokens = set(re.findall(r"[a-z0-9]+", claim["claim"].lower()))
        placed = False
        for group in groups:
            overlap = len(tokens & group["tokens"]) / max(1, len(tokens | group["tokens"]))
            if overlap >= 0.5:
                group["sources"].add(claim["source_id"])
                group["members"].append(claim["claim"])
                group["tokens"] |= tokens
                placed = True
                break
        if not placed:
            groups.append({"tokens": tokens, "sources": {claim["source_id"]},
                           "members": [claim["claim"]]})
    return {
        "groups": [
            {"claim": g["members"][0], "independent_sources": len(g["sources"]),
             "variants": len(g["members"])}
            for g in groups
        ],
        "corroborated": sum(1 for g in groups if len(g["sources"]) > 1),
        "single_source": sum(1 for g in groups if len(g["sources"]) == 1),
    }


def build_report(root: str, title: str, out_path: str = "report.md") -> dict:
    """Assemble a markdown report. Every claim carries its citation.

    Claims whose source vanished are dropped and counted, not silently emitted:
    a report is only as trustworthy as its weakest citation.
    """
    data = load(root)
    sources = data["sources"]
    lines = [f"# {title}", ""]
    dropped = 0
    tier_order = ["primary", "reported", "secondary", "unattributed"]
    for tier in tier_order:
        tier_claims = [c for c in data["claims"] if c["tier"] == tier]
        if not tier_claims:
            continue
        lines += [f"## Findings from {tier} sources", ""]
        for c in tier_claims:
            src = sources.get(c["source_id"])
            if src is None:
                dropped += 1
                continue
            lines.append(f"- {c['claim']} [{c['source_id']}]")
            lines.append(f"  > {c['quote'][:200]}")
        lines.append("")
    lines += ["## Sources", ""]
    for sid, src in sorted(sources.items()):
        lines.append(f"- `{sid}` ({src['tier']}) [{src['title']}]({src['url']})")

    corr = corroboration(root)
    lines += ["", "## Corroboration", "",
              f"- {corr['corroborated']} findings confirmed by more than one source",
              f"- {corr['single_source']} findings rest on a single source"]

    target = _path(root, out_path)
    target.write_text("\n".join(lines), encoding="utf-8")
    return {"path": str(target), "bytes": target.stat().st_size,
            "claims_included": data["claim_count"] - dropped,
            "claims_dropped_missing_source": dropped,
            "corroborated": corr["corroborated"]}
