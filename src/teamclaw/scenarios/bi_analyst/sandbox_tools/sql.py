"""In-sandbox SQL over a read-only SQLite database.

Two things this scenario stresses that the diligence one does not:

**Schema context.** The agent cannot see the database; it has to ask. ``schema``
returns table and column names with row counts — the shape, not the data — which
is exactly the ledger's workspace-slot discipline applied to a database.

**Execution safety.** A generated query is untrusted code. Connections are
opened in SQLite's URI read-only mode, statements are checked against a
write/DDL blocklist before execution, every query gets a ``LIMIT``, and there is
a statement timeout via a progress handler. Read-only mode alone is not enough:
a runaway cross join is a denial of service even with no write access, and on a
free-tier eval a hung case costs the whole run.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|"
    r"pragma|vacuum|reindex|begin|commit|rollback)\b",
    re.IGNORECASE,
)
MULTI_STATEMENT = re.compile(r";\s*\S")
DEFAULT_LIMIT = 200
TIMEOUT_S = 10.0


def _connect(db_path: str) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    deadline = time.monotonic() + TIMEOUT_S
    # A progress handler is the only portable way to interrupt a long query:
    # read-only prevents damage, not a cross join that never finishes.
    conn.set_progress_handler(
        lambda: 1 if time.monotonic() > deadline else 0, 10_000
    )
    return conn


def schema(db_path: str) -> dict:
    """Tables, columns and row counts. Shape only — never the rows."""
    with _connect(db_path) as conn:
        tables = [
            r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        out = {}
        for table in tables:
            cols = [
                {"name": r["name"], "type": r["type"], "pk": bool(r["pk"])}
                for r in conn.execute(f'PRAGMA table_info("{table}")')
            ]
            count = conn.execute(f'SELECT COUNT(*) AS n FROM "{table}"').fetchone()["n"]
            out[table] = {"columns": cols, "rows": count}
    return {"database": db_path, "tables": out, "table_count": len(out)}


def check(query: str) -> dict:
    """Static safety check. Call this before `run` if you are unsure."""
    text = (query or "").strip().rstrip(";")
    if not text:
        return {"safe": False, "reason": "empty query"}
    if MULTI_STATEMENT.search(query or ""):
        return {"safe": False, "reason": "multiple statements are not permitted"}
    if not re.match(r"^\s*(select|with)\b", text, re.IGNORECASE):
        return {"safe": False, "reason": "only SELECT/WITH queries are permitted"}
    if (bad := FORBIDDEN.search(text)):
        return {"safe": False, "reason": f"forbidden keyword: {bad.group(0)}"}
    return {"safe": True, "reason": ""}


def run(db_path: str, query: str, limit: int = DEFAULT_LIMIT) -> dict:
    """Execute a read-only query and return at most `limit` rows.

    Returns a digest plus the rows. Print the digest and the head, not the whole
    result set — a 200-row answer in context is the same mistake as pasting a
    filing into it.
    """
    verdict = check(query)
    if not verdict["safe"]:
        return {"error": verdict["reason"], "query": query, "rows": []}

    wrapped = query.strip().rstrip(";")
    if not re.search(r"\blimit\s+\d+\s*$", wrapped, re.IGNORECASE):
        wrapped = f"SELECT * FROM ({wrapped}) LIMIT {int(limit)}"

    started = time.time()
    try:
        with _connect(db_path) as conn:
            cursor = conn.execute(wrapped)
            rows = [dict(r) for r in cursor.fetchall()]
            columns = [d[0] for d in (cursor.description or [])]
    except sqlite3.OperationalError as exc:
        return {"error": f"OperationalError: {exc}", "query": wrapped, "rows": []}
    except sqlite3.DatabaseError as exc:
        return {"error": f"DatabaseError: {exc}", "query": wrapped, "rows": []}

    return {
        "query": wrapped,
        "columns": columns,
        "row_count": len(rows),
        "truncated": len(rows) >= int(limit),
        "duration_s": round(time.time() - started, 4),
        "rows": rows,
    }


def sample(db_path: str, table: str, n: int = 5) -> dict:
    """A few rows from one table, to learn its value shapes."""
    return run(db_path, f'SELECT * FROM "{table}"', limit=int(n))


def save_json(obj: object, path: str) -> dict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str),
                      encoding="utf-8")
    return {"path": str(target), "bytes": target.stat().st_size}
