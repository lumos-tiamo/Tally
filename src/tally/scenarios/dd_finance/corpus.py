"""Corpus selection for the diligence eval.

Selection is by *accounting difficulty*, not by market cap. A set of large-cap
tech filers would score well and prove nothing: they all use the same tags, all
have classified balance sheets, and all disclose every field. The mapping layer
and the agent are both only tested by filers that break those assumptions.

Difficulty axes, and what each one breaks:

``unclassified_balance_sheet``
    Banks and insurers present no current/non-current split, so
    ``current_assets`` and ``current_liabilities`` are legitimately absent. These
    cases are the entire test of *abstention* — a model that invents a current
    ratio for a bank is hallucinating, and without banks in the corpus that never
    shows up.
``no_cost_of_revenue``
    Financials and some REITs report no cost-of-revenue line, so gross margin is
    undefined rather than zero.
``revenue_tag_variance``
    Pre/post ASC 606 filers and insurers use different revenue concepts.
``restatement_risk``
    Filers with known restatements or frequent amendments, to exercise the
    "latest filed wins" path.
``heavy_ma``
    Acquisition-heavy years, where goodwill and NCI make ``NetIncomeLoss`` vs
    ``ProfitLoss`` actually differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class CorpusEntry:
    ticker: str
    name: str
    sector: str
    difficulty: tuple[str, ...] = ()
    held_out: bool = False

    def to_json(self) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "sector": self.sector,
            "difficulty": list(self.difficulty),
            "held_out": self.held_out,
        }


# 46 primary + 14 held-out. Fiscal years are supplied at build time (3 per
# company by default), giving ~138 primary cases and ~42 held out.
CORPUS: tuple[CorpusEntry, ...] = (
    # --- large-cap tech: the easy baseline, deliberately a minority ---------
    CorpusEntry("AAPL", "Apple", "tech"),
    CorpusEntry("MSFT", "Microsoft", "tech"),
    CorpusEntry("GOOGL", "Alphabet", "tech", ("heavy_ma",)),
    CorpusEntry("NVDA", "NVIDIA", "semis"),
    CorpusEntry("AMZN", "Amazon", "retail", ("heavy_ma",)),
    CorpusEntry("META", "Meta Platforms", "tech"),
    # --- financials: unclassified balance sheets, no cost of revenue -------
    CorpusEntry("JPM", "JPMorgan Chase", "banks",
                ("unclassified_balance_sheet", "no_cost_of_revenue", "revenue_tag_variance")),
    CorpusEntry("BAC", "Bank of America", "banks",
                ("unclassified_balance_sheet", "no_cost_of_revenue")),
    CorpusEntry("WFC", "Wells Fargo", "banks",
                ("unclassified_balance_sheet", "no_cost_of_revenue", "restatement_risk")),
    CorpusEntry("GS", "Goldman Sachs", "banks",
                ("unclassified_balance_sheet", "no_cost_of_revenue")),
    CorpusEntry("BRK-B", "Berkshire Hathaway", "insurance",
                ("unclassified_balance_sheet", "revenue_tag_variance", "heavy_ma")),
    CorpusEntry("PGR", "Progressive", "insurance",
                ("unclassified_balance_sheet", "no_cost_of_revenue")),
    CorpusEntry("AIG", "AIG", "insurance",
                ("unclassified_balance_sheet", "restatement_risk")),
    # --- healthcare / pharma: R&D heavy, milestone revenue -----------------
    CorpusEntry("JNJ", "Johnson & Johnson", "pharma", ("heavy_ma",)),
    CorpusEntry("PFE", "Pfizer", "pharma", ("heavy_ma", "revenue_tag_variance")),
    CorpusEntry("UNH", "UnitedHealth", "healthcare",
                ("unclassified_balance_sheet", "revenue_tag_variance")),
    CorpusEntry("CVS", "CVS Health", "healthcare", ("heavy_ma",)),
    CorpusEntry("ABBV", "AbbVie", "pharma", ("heavy_ma",)),
    # --- energy / utilities: regulated accounting, impairments -------------
    CorpusEntry("XOM", "Exxon Mobil", "energy", ("heavy_ma",)),
    CorpusEntry("CVX", "Chevron", "energy"),
    CorpusEntry("NEE", "NextEra Energy", "utilities", ("unclassified_balance_sheet",)),
    CorpusEntry("DUK", "Duke Energy", "utilities"),
    # --- industrials / transport -------------------------------------------
    CorpusEntry("BA", "Boeing", "industrials", ("restatement_risk",)),
    CorpusEntry("CAT", "Caterpillar", "industrials"),
    CorpusEntry("GE", "GE Aerospace", "industrials", ("restatement_risk", "heavy_ma")),
    CorpusEntry("UPS", "UPS", "transport"),
    CorpusEntry("DAL", "Delta Air Lines", "transport", ("unclassified_balance_sheet",)),
    # --- consumer -----------------------------------------------------------
    CorpusEntry("WMT", "Walmart", "retail"),
    CorpusEntry("COST", "Costco", "retail"),
    CorpusEntry("TGT", "Target", "retail"),
    CorpusEntry("KO", "Coca-Cola", "staples"),
    CorpusEntry("PG", "Procter & Gamble", "staples"),
    CorpusEntry("NKE", "Nike", "consumer"),
    CorpusEntry("SBUX", "Starbucks", "consumer"),
    CorpusEntry("MCD", "McDonald's", "consumer"),
    # --- real estate: REIT accounting ---------------------------------------
    CorpusEntry("AMT", "American Tower", "reit",
                ("unclassified_balance_sheet", "no_cost_of_revenue")),
    CorpusEntry("PLD", "Prologis", "reit", ("unclassified_balance_sheet", "heavy_ma")),
    CorpusEntry("SPG", "Simon Property", "reit", ("unclassified_balance_sheet",)),
    # --- telecom / media ----------------------------------------------------
    CorpusEntry("T", "AT&T", "telecom", ("restatement_risk", "heavy_ma")),
    CorpusEntry("VZ", "Verizon", "telecom"),
    CorpusEntry("DIS", "Disney", "media", ("heavy_ma",)),
    CorpusEntry("NFLX", "Netflix", "media"),
    # --- smaller / loss-making: negative equity, losses ---------------------
    CorpusEntry("UBER", "Uber", "tech", ("heavy_ma",)),
    CorpusEntry("SNAP", "Snap", "tech"),
    CorpusEntry("RIVN", "Rivian", "auto"),
    CorpusEntry("PLUG", "Plug Power", "industrials", ("restatement_risk",)),

    # --- held out: never used for tuning -----------------------------------
    CorpusEntry("ORCL", "Oracle", "tech", ("heavy_ma",), held_out=True),
    CorpusEntry("CRM", "Salesforce", "tech", ("heavy_ma",), held_out=True),
    CorpusEntry("INTC", "Intel", "semis", held_out=True),
    CorpusEntry("MS", "Morgan Stanley", "banks",
                ("unclassified_balance_sheet", "no_cost_of_revenue"), held_out=True),
    CorpusEntry("MET", "MetLife", "insurance",
                ("unclassified_balance_sheet",), held_out=True),
    CorpusEntry("MRK", "Merck", "pharma", ("heavy_ma",), held_out=True),
    CorpusEntry("COP", "ConocoPhillips", "energy", held_out=True),
    CorpusEntry("SO", "Southern Company", "utilities", held_out=True),
    CorpusEntry("LMT", "Lockheed Martin", "industrials", held_out=True),
    CorpusEntry("HD", "Home Depot", "retail", held_out=True),
    CorpusEntry("PEP", "PepsiCo", "staples", held_out=True),
    CorpusEntry("EQIX", "Equinix", "reit",
                ("unclassified_balance_sheet",), held_out=True),
    CorpusEntry("TMUS", "T-Mobile", "telecom", ("heavy_ma",), held_out=True),
    CorpusEntry("LYFT", "Lyft", "tech", held_out=True),
)

PRIMARY = tuple(e for e in CORPUS if not e.held_out)
HELD_OUT = tuple(e for e in CORPUS if e.held_out)
BY_TICKER: dict[str, CorpusEntry] = {e.ticker: e for e in CORPUS}


def select(
    *,
    held_out: bool = False,
    sectors: Sequence[str] | None = None,
    difficulty: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[CorpusEntry]:
    pool = list(HELD_OUT if held_out else PRIMARY)
    if sectors:
        wanted = {s.lower() for s in sectors}
        pool = [e for e in pool if e.sector.lower() in wanted]
    if difficulty:
        wanted_d = set(difficulty)
        pool = [e for e in pool if wanted_d & set(e.difficulty)]
    return pool[:limit] if limit else pool


def difficulty_summary() -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in CORPUS:
        for d in e.difficulty:
            counts[d] = counts.get(d, 0) + 1
        if not e.difficulty:
            counts["plain"] = counts.get("plain", 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def sector_summary() -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in CORPUS:
        counts[e.sector] = counts.get(e.sector, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
