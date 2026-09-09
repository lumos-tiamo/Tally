"""SEC EDGAR client: host-side, cached, rate-limited.

Three constraints shape this module.

**Fair access.** SEC requires a self-identifying User-Agent and throttles above
roughly 10 requests/second. Both are enforced here rather than trusted to
callers, because a scenario that gets a repo IP-blocked mid-eval is unrecoverable
in a way a slow eval is not. :class:`RateLimiter` is deliberately conservative.

**Caching is not an optimisation, it is a correctness property.** A ground-truth
set built from live fetches is not reproducible: filings get amended, and the
same eval re-run next week would score against different truth. Every response
is written to disk keyed by URL, and the cache is consulted first, so an eval run
is pinned to the corpus it was built from.

**The fiscal calendar comes from the filing index, not from the facts.** As
:mod:`.concepts` explains, ``fy`` on a fact is the filing's year. The authoritative
fiscal-year end for a company-year is the ``reportDate`` of that year's 10-K,
which lives in the submissions document.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from teamclaw.config import Settings, settings as load_settings

DATA_BASE = "https://data.sec.gov"
WWW_BASE = "https://www.sec.gov"
TICKERS_URL = f"{WWW_BASE}/files/company_tickers.json"


class RateLimiter:
    """Token-free minimum-interval limiter, thread-safe."""

    def __init__(self, min_interval_s: float = 0.15) -> None:
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < self.min_interval_s:
                time.sleep(self.min_interval_s - gap)
            self._last = time.monotonic()


@dataclass
class Filing:
    accession: str
    form: str
    filing_date: dt.date
    report_date: dt.date | None
    primary_document: str
    cik: str
    fiscal_year: int | None = None

    @property
    def accession_nodash(self) -> str:
        return self.accession.replace("-", "")

    @property
    def index_url(self) -> str:
        return (
            f"{WWW_BASE}/Archives/edgar/data/{int(self.cik)}/"
            f"{self.accession_nodash}/{self.accession}-index.htm"
        )

    @property
    def document_url(self) -> str:
        return (
            f"{WWW_BASE}/Archives/edgar/data/{int(self.cik)}/"
            f"{self.accession_nodash}/{self.primary_document}"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "accession": self.accession,
            "form": self.form,
            "filing_date": self.filing_date.isoformat(),
            "report_date": self.report_date.isoformat() if self.report_date else None,
            "fiscal_year": self.fiscal_year,
            "primary_document": self.primary_document,
            "document_url": self.document_url,
            "cik": self.cik,
        }


def _date(raw: str | None) -> dt.date | None:
    if not raw:
        return None
    try:
        return dt.date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


@dataclass
class SecClient:
    cfg: Settings = field(default_factory=load_settings)
    cache_dir: Path | None = None
    limiter: RateLimiter = field(default_factory=RateLimiter)
    timeout_s: float = 45.0
    offline: bool = False          # cache-only; raises on a miss
    max_retries: int = 3
    fetches: int = 0
    cache_hits: int = 0

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir or self.cfg.paths.cache / "sec")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- transport ---------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        suffix = ".json" if url.endswith(".json") else ".bin"
        return self.cache_dir / digest[:2] / f"{digest}{suffix}"

    def _get(self, url: str, *, binary: bool = False) -> bytes:
        path = self._cache_path(url)
        if path.exists():
            self.cache_hits += 1
            return path.read_bytes()
        if self.offline:
            raise FileNotFoundError(f"offline and not cached: {url}")

        headers = {
            "User-Agent": self.cfg.require_sec_user_agent(),
            "Accept-Encoding": "gzip, deflate",
            "Host": url.split("/")[2],
        }
        last: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                with httpx.Client(timeout=self.timeout_s, follow_redirects=True) as client:
                    resp = client.get(url, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
                time.sleep(min(2.0**attempt, 8.0))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                last = RuntimeError(f"SEC {resp.status_code} for {url}")
                # Back off harder on throttling: the fair-access limit is per-IP
                # and being impatient here gets the whole run blocked.
                time.sleep(min(5.0 * (attempt + 1), 20.0))
                continue
            if resp.status_code == 404:
                raise FileNotFoundError(f"SEC 404: {url}")
            resp.raise_for_status()

            self.fetches += 1
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(resp.content)
            return resp.content
        raise RuntimeError(f"SEC fetch failed after {self.max_retries} attempts: {url}") from last

    def _get_json(self, url: str) -> dict[str, Any]:
        return json.loads(self._get(url).decode("utf-8"))

    # -- endpoints ---------------------------------------------------------
    @staticmethod
    def pad_cik(cik: str | int) -> str:
        digits = re.sub(r"\D", "", str(cik))
        return digits.zfill(10)

    def company_tickers(self) -> dict[str, dict[str, Any]]:
        """ticker -> {cik, title}."""
        raw = self._get_json(TICKERS_URL)
        out: dict[str, dict[str, Any]] = {}
        for row in raw.values():
            ticker = str(row.get("ticker", "")).upper()
            if ticker:
                out[ticker] = {
                    "cik": self.pad_cik(row.get("cik_str", "")),
                    "title": row.get("title", ""),
                }
        return out

    def cik_for_ticker(self, ticker: str) -> str:
        table = self.company_tickers()
        key = ticker.upper().strip()
        if key not in table:
            raise KeyError(f"unknown ticker {ticker!r}")
        return table[key]["cik"]

    def submissions(self, cik: str | int) -> dict[str, Any]:
        return self._get_json(f"{DATA_BASE}/submissions/CIK{self.pad_cik(cik)}.json")

    def company_facts(self, cik: str | int) -> dict[str, Any]:
        return self._get_json(
            f"{DATA_BASE}/api/xbrl/companyfacts/CIK{self.pad_cik(cik)}.json"
        )

    # -- filings -----------------------------------------------------------
    _FILING_COLS = (
        "accessionNumber", "form", "filingDate", "reportDate", "primaryDocument"
    )

    def _filings_from_block(
        self, block: dict[str, Any], padded_cik: str, forms: tuple[str, ...]
    ) -> list[Filing]:
        series = {c: block.get(c) or [] for c in self._FILING_COLS}
        n = min((len(v) for v in series.values()), default=0)
        out: list[Filing] = []
        for i in range(n):
            form = str(series["form"][i])
            if forms and form not in forms:
                continue
            filed = _date(series["filingDate"][i])
            if filed is None:
                continue
            report = _date(series["reportDate"][i])
            out.append(
                Filing(
                    accession=str(series["accessionNumber"][i]),
                    form=form,
                    filing_date=filed,
                    report_date=report,
                    primary_document=str(series["primaryDocument"][i]),
                    cik=padded_cik,
                    fiscal_year=fiscal_year_of(report) if report else None,
                )
            )
        return out

    def annual_filings(
        self, cik: str | int, *, forms: tuple[str, ...] = ("10-K",), limit: int = 12
    ) -> list[Filing]:
        """Annual reports, newest first, with their fiscal-year end dates.

        ``reportDate`` is the fiscal period end; the fiscal-year *label* is
        derived from it rather than from any ``fy`` field, for the reason given in
        :mod:`.concepts`. A December-31 year end maps to its own calendar year; an
        early-January end (52/53-week filers) maps back one year.

        **Pagination matters here.** ``filings.recent`` holds only the most recent
        ~1000 submissions. A prolific filer — a large bank publishing thousands of
        8-Ks, 424Bs and FWPs a year — pushes its own 10-K out of that window
        within months, so reading only ``recent`` silently returns no annual
        reports for exactly the companies whose accounting is hardest. The
        overflow lives in ``filings.files[]`` as additional JSON documents, which
        are fetched here on demand until ``limit`` annual reports are found.
        """
        data = self.submissions(cik)
        padded = self.pad_cik(cik)
        filings_block = data.get("filings") or {}

        out = self._filings_from_block(filings_block.get("recent") or {}, padded, forms)

        # Overflow pages carry ``filingFrom``/``filingTo``. Sorting by ``filingTo``
        # descending and stopping early matters: a large bank has a dozen or more
        # overflow pages of several MB each, and walking them in file order to find
        # last year's 10-K downloads the entire filing history first.
        if len(out) < limit:
            pages = [
                (str(e.get("filingTo") or ""), str(e.get("name") or ""))
                for e in (filings_block.get("files") or [])
                if e.get("name")
            ]
            for _, name in sorted(pages, reverse=True):
                try:
                    block = self._get_json(f"{DATA_BASE}/submissions/{name}")
                except (FileNotFoundError, RuntimeError):
                    continue
                out.extend(self._filings_from_block(block, padded, forms))
                if len(out) >= limit:
                    break

        out.sort(key=lambda f: f.filing_date, reverse=True)
        return out[:limit]

    def fiscal_year_end(self, cik: str | int, fiscal_year: int) -> dt.date | None:
        for f in self.annual_filings(cik, limit=20):
            if f.fiscal_year == fiscal_year and f.report_date is not None:
                return f.report_date
        return None

    def filing_document(self, filing: Filing) -> str:
        return self._get(filing.document_url, binary=True).decode("utf-8", errors="replace")

    def stats(self) -> dict[str, Any]:
        return {
            "network_fetches": self.fetches,
            "cache_hits": self.cache_hits,
            "cache_dir": str(self.cache_dir),
            "offline": self.offline,
        }


def fiscal_year_of(report_date: dt.date) -> int:
    """Fiscal-year label for a period-end date.

    A year end in January (or the first days of it) belongs to the prior fiscal
    year: 52/53-week retailers routinely close FY2024 on 2025-02-01. Anything
    from February onward takes its own calendar year.
    """
    return report_date.year - 1 if report_date.month == 1 else report_date.year
