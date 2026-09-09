"""In-sandbox repository tools: read, patch, and run the tests.

The engineering scenario's distinctive property is that it has a *ground-truth
oracle the agent can call itself*: the test suite. That changes the loop shape —
instead of extract-then-report, it is change-then-verify, and the agent can tell
whether it succeeded before it says so.

Two safeguards make that usable rather than dangerous:

**Patches are exact-match replacements, not line numbers.** ``replace`` requires
the old text to appear exactly once. A line-oriented patch tool silently corrupts
a file whenever the agent's mental model of it has drifted, and drift is the norm
after a few edits.

**Every edit is backed up before it lands**, so a bad change can be reverted in
one call rather than reconstructed from memory.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BACKUP_DIR = ".repo_backups"
MAX_READ_CHARS = 20_000


def _safe(root: str, rel: str) -> Path:
    base = Path(root).resolve()
    target = (base / rel).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"path escapes the repository: {rel!r}")
    return target


def tree(root: str, pattern: str = "*.py", limit: int = 200) -> dict:
    """List matching files with sizes and line counts. Shape, not content."""
    base = Path(root)
    files = []
    for path in sorted(base.rglob(pattern)):
        if BACKUP_DIR in path.parts or "__pycache__" in path.parts:
            continue
        if not path.is_file():
            continue
        try:
            lines = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))
        except OSError:
            continue
        files.append({"path": str(path.relative_to(base)), "bytes": path.stat().st_size,
                      "lines": lines})
        if len(files) >= limit:
            break
    return {"root": root, "files": files, "count": len(files)}


def read(root: str, rel: str, start: int = 1, end: int = 0) -> dict:
    """Read a line range. Capped, and it reports the cap.

    Line-ranged by default because whole-file reads are how a repo scenario
    exhausts a context window in three steps.
    """
    path = _safe(root, rel)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hi = end if end else len(lines)
    lo = max(1, start)
    chunk = "\n".join(f"{n:>5} | {lines[n - 1]}" for n in range(lo, min(hi, len(lines)) + 1))
    return {"path": rel, "total_lines": len(lines), "from": lo, "to": min(hi, len(lines)),
            "truncated": len(chunk) > MAX_READ_CHARS, "text": chunk[:MAX_READ_CHARS]}


def grep(root: str, pattern: str, glob: str = "*.py", limit: int = 40) -> dict:
    """Search the repository, returning file:line matches."""
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return {"error": f"bad regex: {exc}", "matches": []}
    base = Path(root)
    matches = []
    for path in sorted(base.rglob(glob)):
        if BACKUP_DIR in path.parts or "__pycache__" in path.parts or not path.is_file():
            continue
        try:
            for n, line in enumerate(path.open("r", encoding="utf-8", errors="replace"), 1):
                if rx.search(line):
                    matches.append({"path": str(path.relative_to(base)), "line": n,
                                    "text": line.rstrip()[:200]})
                    if len(matches) >= limit:
                        return {"pattern": pattern, "matches": matches,
                                "count": len(matches), "truncated": True}
        except OSError:
            continue
    return {"pattern": pattern, "matches": matches, "count": len(matches), "truncated": False}


def _backup(root: str, rel: str) -> str:
    path = _safe(root, rel)
    target = Path(root) / BACKUP_DIR / f"{rel.replace('/', '__')}.{int(time.time())}.bak"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    return str(target.relative_to(Path(root)))


def replace(root: str, rel: str, old: str, new: str) -> dict:
    """Replace an exact snippet that must occur exactly once.

    Uniqueness is required, not convenient. Two occurrences means the agent's
    anchor is ambiguous and the edit it intends is not the edit it would get; a
    tool that picks the first match turns that ambiguity into a silent bug.
    """
    path = _safe(root, rel)
    text = path.read_text(encoding="utf-8")
    occurrences = text.count(old)
    if occurrences == 0:
        return {"applied": False, "reason": "old text not found; re-read the file — "
                                            "it may have changed since you last saw it"}
    if occurrences > 1:
        return {"applied": False, "occurrences": occurrences,
                "reason": "old text appears more than once; include more surrounding "
                          "context so the anchor is unique"}
    backup = _backup(root, rel)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return {"applied": True, "path": rel, "backup": backup,
            "delta_chars": len(new) - len(old)}


def write_file(root: str, rel: str, content: str) -> dict:
    """Create or overwrite a file, backing up any existing version."""
    path = _safe(root, rel)
    backup = None
    if path.exists():
        backup = _backup(root, rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"written": True, "path": rel, "bytes": path.stat().st_size, "backup": backup}


def revert(root: str, backup_rel: str) -> dict:
    """Restore a file from a backup produced by an earlier edit."""
    backup = _safe(root, backup_rel)
    if not backup.exists():
        return {"reverted": False, "reason": f"no such backup: {backup_rel}"}
    original = backup.name.split(".")[0].replace("__", "/")
    target = _safe(root, original)
    shutil.copy2(backup, target)
    return {"reverted": True, "path": original, "from": backup_rel}


def run_tests(root: str, target: str = "", timeout_s: int = 120) -> dict:
    """Run pytest and return a digest of the outcome, not the whole log.

    The tail of the output is what carries the failure; the head is setup noise.
    So the digest keeps counts plus the last lines, which is what the next edit
    actually needs.
    """
    argv = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"]
    if target:
        argv.append(target)
    started = time.time()
    try:
        proc = subprocess.run(argv, cwd=root, capture_output=True, text=True,
                              timeout=timeout_s)
        out = (proc.stdout or "") + (proc.stderr or "")
        code = proc.returncode
    except subprocess.TimeoutExpired:
        return {"ran": False, "timed_out": True, "duration_s": timeout_s,
                "reason": f"test run exceeded {timeout_s}s"}
    except FileNotFoundError:
        return {"ran": False, "reason": "pytest is not installed in the sandbox"}

    passed = failed = errors = 0
    if (m := re.search(r"(\d+) passed", out)):
        passed = int(m.group(1))
    if (m := re.search(r"(\d+) failed", out)):
        failed = int(m.group(1))
    if (m := re.search(r"(\d+) error", out)):
        errors = int(m.group(1))
    tail = "\n".join(out.strip().splitlines()[-25:])
    return {"ran": True, "exit_code": code, "green": code == 0,
            "passed": passed, "failed": failed, "errors": errors,
            "duration_s": round(time.time() - started, 2), "tail": tail}


def save_json(obj: object, path: str) -> dict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str),
                      encoding="utf-8")
    return {"path": str(target), "bytes": target.stat().st_size}
