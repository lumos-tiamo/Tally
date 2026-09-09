"""In-sandbox filing reader: locate, quote, and tabulate. Runs with no network.

A 10-K is 100–300 pages of HTML. The agent must never load it into context, so
every function here is built to return *locations and digests*, never bulk text.

Two problems this solves that a naive text search does not:

**Filing HTML has no page numbers.** Printed page numbers exist only as
page-break markers in the markup. :func:`locate` reports a page *estimate*
derived from those markers when present, and a characters-per-page fallback when
not, alongside the far more useful ``item`` (Item 1A, Item 7, Item 8 …) and a
character offset. The eval verifies quotes, not page numbers, precisely because
the page number is the least reliable coordinate in the document.

**Numbers live in tables whose scale is stated elsewhere.** "391,035" means
$391 billion only because a heading three rows up says "in millions".
:func:`table_scale` looks for that heading, and :func:`find_number` reports the
scale it found so the caller can convert deliberately instead of guessing.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

# Item headings that mark the top-level structure of a 10-K.
ITEM_PATTERN = re.compile(
    r"item\s+(1|1a|1b|1c|2|3|4|5|6|7|7a|8|9|9a|9b|10|11|12|13|14|15|16)\b[\.\:\s\-—]",
    re.IGNORECASE,
)
PAGE_BREAK = re.compile(
    r"page-break-(?:before|after)\s*:\s*always|<hr[^>]*>", re.IGNORECASE
)
SCALE_PATTERN = re.compile(
    r"\(?\s*in\s+(thousands|millions|billions)\s*(?:,\s*except[^)]*)?\)?", re.IGNORECASE
)
# The comma-grouped branch requires at least ONE group (``+``, not ``*``).
# With ``*`` it matches the first three digits of a bare "2024" and leaves "4"
# behind as a separate number — so every year in the document arrives as the
# pair (202, 4), and a statement line's real figures get lost among them.
NUMBER = re.compile(
    r"-?\(?\$?\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?"
    r"|-?\(?\$?\s?\d+(?:\.\d+)?\)?"
)
CHARS_PER_PAGE = 3200

SCALE_FACTOR = {"thousands": 1e3, "millions": 1e6, "billions": 1e9}


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def normalise(text: str) -> str:
    """NFKC, unify dashes/quotes, collapse whitespace, lowercase."""
    return normalise_with_map(text)[0]


def normalise_with_map(text: str) -> tuple[str, list[int]]:
    """Normalise and return an exact normalised-index -> raw-index map.

    The map is the point. Collapsing runs of whitespace shortens the text
    non-uniformly, so a normalised offset cannot be scaled back to a raw offset
    by ratio: in a filing converted from HTML, whitespace runs are dense in the
    tables and sparse in the prose, and a proportional estimate lands tens of
    thousands of characters away. That is the difference between quoting the
    income statement and quoting the competition section.

    Built character by character rather than with regex substitution, because
    only a per-character walk can record where each surviving character came from.
    """
    raw = text or ""
    out: list[str] = []
    index: list[int] = []
    pending_space = False
    started = False

    for i, ch in enumerate(raw):
        # Drop the invisibles outright.
        if ch in "\u00ad\u200b":
            continue
        if ch.isspace():
            pending_space = started
            continue

        folded = unicodedata.normalize("NFKC", ch)
        if ch in "\u2018\u2019\u201a\u201b":
            folded = "'"
        elif ch in "\u201c\u201d\u201e\u201f":
            folded = '"'
        elif "\u2010" <= ch <= "\u2015":
            folded = "-"
        folded = folded.lower()

        if pending_space:
            out.append(" ")
            index.append(i)
            pending_space = False
        for c in folded:
            out.append(c)
            index.append(i)
        started = True

    return "".join(out), index


def to_text(path: str, out_path: str | None = None) -> dict:
    """Strip a filing's HTML to plain text, writing it beside the source.

    Returns a digest, never the text. The plain-text file is what every other
    function here operates on, so this is normally the agent's first call.
    """
    raw = _read(path)
    breaks = [m.start() for m in PAGE_BREAK.finditer(raw)]

    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(raw, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text("\n")
    except Exception:
        # No parser available: fall back to tag stripping rather than failing.
        text = re.sub(r"<[^>]+>", "\n", raw)

    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    target = Path(out_path or (str(Path(path).with_suffix("")) + ".txt"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")

    scale = SCALE_PATTERN.search(text[:200_000])
    return {
        "path": str(target),
        "chars": len(text),
        "lines": text.count("\n") + 1,
        "page_break_markers": len(breaks),
        "estimated_pages": len(breaks) or max(1, len(text) // CHARS_PER_PAGE),
        "document_scale_hint": scale.group(1).lower() if scale else None,
        "head": text[:300],
    }


# Canonical 10-K item order. Used to reconstruct the real section sequence.
ITEM_ORDER: tuple[str, ...] = (
    "1", "1a", "1b", "1c", "2", "3", "4", "5", "6", "7", "7a", "8",
    "9", "9a", "9b", "10", "11", "12", "13", "14", "15", "16",
)


def outline(path: str) -> dict:
    """Item-level table of contents with character offsets.

    Neither "first occurrence" nor "last occurrence" works. Every item is named
    at least twice — once in the front index, once at the section itself — and
    filings also cross-reference each other ("the information required is included
    in Item 8"), which is why a last-occurrence scan puts Item 8 *after* Item 9A.

    So the sequence is reconstructed, and specifically **backwards**: walk the
    canonical item order in reverse and take, for each item, the last occurrence
    that comes before the item already chosen. Forward-greedy fails in a way that
    is easy to miss — the front index lists all 22 items in canonical order within
    a few hundred characters, so a forward walk satisfies the whole ordering
    inside the index and returns an outline where every offset points at the table
    of contents. Walking from the end locks onto the body headings, and a
    cross-reference like "included in Item 8" inside Item 9A cannot win because it
    sits after Item 9's own heading.
    """
    text = _read(path)
    lowered = text.lower()

    occurrences: dict[str, list[int]] = {}
    for m in ITEM_PATTERN.finditer(lowered):
        occurrences.setdefault(m.group(1).lower(), []).append(m.start())

    chosen: dict[str, int] = {}
    cursor = len(text)
    for key in reversed(ITEM_ORDER):
        for offset in reversed(occurrences.get(key, [])):
            if offset <= cursor:
                chosen[key] = offset
                cursor = offset
                break

    ordered = [(k, chosen[k]) for k in ITEM_ORDER if k in chosen]
    items = [{"item": f"Item {k.upper()}", "offset": v} for k, v in ordered]
    return {"path": path, "items": items, "count": len(items), "chars": len(text)}


def _item_at(path: str, offset: int) -> str:
    data = outline(path)
    current = ""
    for entry in data["items"]:
        if entry["offset"] <= offset:
            current = entry["item"]
        else:
            break
    return current


def locate(path: str, needle: str, *, limit: int = 5, window: int = 220) -> dict:
    """Find a phrase and report where it is, with a quotable context window.

    Matching is on normalised text, so a needle retyped with straight quotes
    still finds text that used curly ones — the difference that otherwise makes a
    correct citation look fabricated.
    """
    text = _read(path)
    hay, index = normalise_with_map(text)
    probe = normalise(needle)
    if not probe:
        return {"needle": needle, "hits": [], "count": 0}

    hits = []
    start = 0
    total_pages = max(1, len(text) // CHARS_PER_PAGE)
    while len(hits) < limit:
        pos = hay.find(probe, start)
        if pos < 0:
            break
        raw_pos = index[pos] if pos < len(index) else len(text) - 1
        lo = max(0, raw_pos - window)
        hits.append({
            "offset": raw_pos,
            "page_estimate": min(total_pages, 1 + raw_pos // CHARS_PER_PAGE),
            "item": _item_at(path, raw_pos),
            "context": re.sub(r"\s+", " ", text[lo : raw_pos + window]).strip(),
        })
        start = pos + max(1, len(probe))
    return {"needle": needle, "hits": hits, "count": len(hits)}


def table_scale(path: str, offset: int, *, look_back: int = 4000) -> dict:
    """Find the 'in millions/thousands' heading governing a position.

    Searches backwards, because the scale is stated above the table it governs.
    Returns ``None`` when no heading is found rather than assuming units — a
    guessed scale is a 1000x error.
    """
    text = _read(path)
    lo = max(0, offset - look_back)
    window = text[lo:offset]
    matches = list(SCALE_PATTERN.finditer(window))
    if not matches:
        return {"scale": None, "factor": 1.0, "source": None,
                "note": "no scale heading found within look_back; do not assume units"}
    last = matches[-1]
    word = last.group(1).lower()
    return {
        "scale": word,
        "factor": SCALE_FACTOR[word],
        "source": re.sub(r"\s+", " ", window[max(0, last.start() - 80) : last.end() + 40]).strip(),
        "distance_chars": offset - (lo + last.start()),
    }


def _clean_number(raw: str) -> str:
    """Trim the ``$`` and stray newline a currency-prefixed match picks up."""
    return re.sub(r"\s+", "", str(raw)).lstrip("$").strip()


def parse_value(raw: str) -> float | None:
    """Parse a statement figure, honouring accounting parentheses as negative.

    Cash-flow statements report outflows as ``(9,447)``. Returning the string
    only would leave the model to strip the parenthesis, and a model that strips
    it without noticing the sign turns an outflow into an inflow.
    """
    text = re.sub(r"\s+", "", str(raw or ""))
    negative = text.startswith("(") and text.endswith(")")
    cleaned = re.sub(r"[^\d.\-]", "", text)
    if cleaned in {"", "-", ".", "-."}:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative and value > 0 else value


def _statement_row_score(
    line: str, numbers: list[str], scale: str | None, *, label: str = ""
) -> float:
    """How much a hit looks like a financial-statement row rather than prose.

    Ranking matters: a caption such as "total net sales" appears in the MD&A
    narrative pages before it appears as a line in the income statement, so
    document order puts prose first. A statement row is short, carries several
    comma-grouped figures, and sits under a scale heading.

    The label's *position* in the line carries as much signal as the line's
    shape. "Percentage of total net sales" contains "total net sales" as a
    substring and otherwise looks exactly like a statement row — short, scaled,
    numeric — so shape alone ties it with the real income-statement line and the
    sort order decides. A statement caption begins its line; a caption embedded
    mid-phrase is describing something else.
    """
    score = 0.0
    grouped = [n for n in numbers if "," in n]
    score += 2.0 * min(len(grouped), 4)
    if scale:
        score += 3.0
    words = len(line.split())
    if words <= 8:
        score += 3.0
    elif words <= 16:
        score += 1.0
    else:
        score -= 2.0

    if label:
        norm_line = normalise(line)
        norm_label = normalise(label)
        if norm_line.startswith(norm_label):
            score += 5.0
        elif norm_label in norm_line:
            # Embedded in a longer caption: penalise in proportion to how much
            # text precedes it, so "percentage of X" loses to "X".
            prefix_words = len(norm_line[: norm_line.find(norm_label)].split())
            score -= 2.0 + min(prefix_words, 4)
    return score


def find_number(path: str, label: str, *, limit: int = 5) -> dict:
    """Locate a labelled figure: the numbers on the same line as a caption.

    Candidates are returned ranked by how much each looks like a statement row
    (see :func:`_statement_row_score`), not in document order, because the
    narrative sections mention the same captions earlier than the statements do.
    Several candidates come back on purpose: a statement line carries one figure
    per fiscal year, and picking the right year is the agent's decision.
    """
    text = _read(path)
    # Search wide, then rank, then trim. A caption like "total net sales" can
    # occur a dozen times in the narrative before the income statement uses it,
    # so a candidate pool sized to `limit` never contains the statement row at
    # all and ranking has nothing to fix.
    found = locate(path, label, limit=max(24, limit * 6))
    out = []
    for hit in found["hits"]:
        line_start = text.rfind("\n", 0, hit["offset"]) + 1
        line_end = text.find("\n", hit["offset"])
        line = text[line_start : line_end if line_end > 0 else len(text)]
        # A caption and its figures are sometimes split across sibling lines in
        # converted HTML, so extend a little past the caption's own line.
        extended = text[line_start : (line_end if line_end > 0 else len(text)) + 200]
        numbers = [_clean_number(n.group(0)) for n in NUMBER.finditer(extended)]
        numbers = [n for n in numbers if n]
        scale = table_scale(path, hit["offset"])
        clean_line = re.sub(r"\s+", " ", line).strip()
        out.append({
            "offset": hit["offset"],
            "page_estimate": hit["page_estimate"],
            "item": hit["item"],
            "line": clean_line[:300],
            "numbers": numbers[:8],
            "scale": scale["scale"],
            "scale_factor": scale["factor"],
            "values": [parse_value(n) for n in numbers[:8]],
            "looks_like_statement_row": round(
                _statement_row_score(clean_line, numbers, scale["scale"], label=label), 2
            ),
            "quote": re.sub(r"\s+", " ", extended).strip()[:200],
        })
    out.sort(key=lambda c: -c["looks_like_statement_row"])
    return {"label": label, "candidates": out[:limit], "count": len(out[:limit])}


def grep(path: str, pattern: str, *, limit: int = 20, ignore_case: bool = True) -> dict:
    """Regex search returning matched lines with offsets. For exploration."""
    text = _read(path)
    flags = re.IGNORECASE if ignore_case else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as exc:
        return {"pattern": pattern, "error": f"bad regex: {exc}", "matches": []}
    matches = []
    for m in rx.finditer(text):
        if len(matches) >= limit:
            break
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.start())
        matches.append({
            "offset": m.start(),
            "page_estimate": 1 + m.start() // CHARS_PER_PAGE,
            "line": re.sub(r"\s+", " ", text[line_start : line_end if line_end > 0 else len(text)]).strip()[:300],
        })
    return {"pattern": pattern, "matches": matches, "count": len(matches)}


def section(path: str, item: str, *, max_chars: int = 4000) -> dict:
    """Return the head of one Item section. Capped, and it says so."""
    data = outline(path)
    key = item.strip().lower().replace("item", "").strip()
    entries = data["items"]
    for i, entry in enumerate(entries):
        if entry["item"].lower().replace("item", "").strip() == key:
            start = entry["offset"]
            end = entries[i + 1]["offset"] if i + 1 < len(entries) else data["chars"]
            text = _read(path)[start:end]
            return {
                "item": entry["item"], "offset": start, "chars": end - start,
                "truncated": (end - start) > max_chars,
                "text": text[:max_chars],
            }
    return {"item": item, "error": f"item not found; available: "
            f"{[e['item'] for e in entries][:20]}"}


def save_json(obj: object, path: str) -> dict:
    """Persist a result to the workspace and return only its digest."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return {"path": str(target), "bytes": target.stat().st_size,
            "keys": list(obj.keys()) if isinstance(obj, dict) else None}
