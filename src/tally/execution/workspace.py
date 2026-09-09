"""The agent's working directory — and, by construction, its checkpoint.

The central bet of this platform is that an agent's state belongs on disk rather
than in its conversation. That single decision buys three properties at once:

* **Context stays small.** A 300-page filing, a 40MB parquet of extracted tables
  and a half-written report all live here; only their *digests* enter the prompt.
* **Long-horizon work becomes possible.** Step 40 can read what step 3 wrote,
  without those 37 steps still occupying the window.
* **Resume is nearly free.** Restarting a crashed run means re-reading the
  workspace, not replaying the conversation.

``digest()`` is the interface the context ledger consumes: a file tree plus a
one-line summary per artefact, capped so that a workspace with 500 files does
not itself become the context problem it was meant to solve.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Extensions we can summarise textually. Anything else is reported by size only.
TEXTLIKE = {".txt", ".md", ".json", ".jsonl", ".csv", ".tsv", ".py", ".html", ".xml", ".yaml", ".yml", ".log"}
MAX_DIGEST_BYTES = 400


@dataclass
class Artifact:
    path: Path
    rel: str
    size: int
    mtime: float
    sha256: str = ""
    note: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "path": self.rel,
            "size": self.size,
            "mtime": self.mtime,
            "sha256": self.sha256[:12],
            "note": self.note,
        }


@dataclass
class Workspace:
    """A rooted, mostly-append-only directory scoped to one run or sub-agent."""

    root: Path
    notes: dict[str, str] = field(default_factory=dict)
    max_digest_files: int = 40

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- construction ------------------------------------------------------
    @staticmethod
    def create(base: Path, run_id: str) -> "Workspace":
        return Workspace(root=Path(base) / run_id)

    def child(self, name: str) -> "Workspace":
        """A nested workspace for a sub-agent.

        Sub-agents get their own subtree so their scratch files cannot collide
        with the parent's, while the parent can still read the results — which is
        how file handoff between agents works without a message bus.
        """
        return Workspace(root=self.root / "subagents" / name)

    # -- path safety -------------------------------------------------------
    def resolve(self, rel: str | Path) -> Path:
        """Resolve a relative path, refusing anything that escapes the root.

        Sandboxed code writes here, so traversal must be blocked at the API
        boundary as well as by the container mount.
        """
        candidate = (self.root / Path(rel)).resolve()
        root = self.root.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"path escapes workspace: {rel!r}")
        return candidate

    # -- writes ------------------------------------------------------------
    def write_text(self, rel: str, text: str, *, note: str = "") -> Artifact:
        path = self.resolve(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        if note:
            self.notes[self._rel(path)] = note
        return self.artifact(path)

    def write_json(self, rel: str, data: object, *, note: str = "") -> Artifact:
        return self.write_text(
            rel, json.dumps(data, ensure_ascii=False, indent=1), note=note
        )

    def write_bytes(self, rel: str, blob: bytes, *, note: str = "") -> Artifact:
        path = self.resolve(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        if note:
            self.notes[self._rel(path)] = note
        return self.artifact(path)

    def annotate(self, rel: str, note: str) -> None:
        self.notes[str(Path(rel))] = note

    # -- reads -------------------------------------------------------------
    def read_text(self, rel: str, *, limit: int | None = None) -> str:
        path = self.resolve(rel)
        text = path.read_text(encoding="utf-8", errors="replace")
        return text if limit is None else text[:limit]

    def exists(self, rel: str) -> bool:
        try:
            return self.resolve(rel).exists()
        except ValueError:
            return False

    def _rel(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root.resolve()))

    def artifact(self, path: Path) -> Artifact:
        rel = self._rel(path)
        stat = path.stat()
        return Artifact(
            path=path,
            rel=rel,
            size=stat.st_size,
            mtime=stat.st_mtime,
            sha256=self._sha256(path),
            note=self.notes.get(rel, ""),
        )

    @staticmethod
    def _sha256(path: Path, chunk: int = 1 << 16) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            while block := fh.read(chunk):
                h.update(block)
        return h.hexdigest()

    def artifacts(self, *, newest_first: bool = True, include_internal: bool = False) -> list[Artifact]:
        """List artefacts.

        Dot-directories are platform plumbing — the RPC bridge's request/response
        files, compaction archives' scratch — and are hidden by default. They are
        not the agent's work product, and letting them into :meth:`digest` both
        wastes context and invites the agent to reason about its own transport.
        Pass ``include_internal=True`` for checkpoint manifests, which must cover
        everything on disk to detect drift honestly.
        """
        found: list[Artifact] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [
                d for d in dirnames
                if d not in {"__pycache__", ".ipynb_checkpoints"}
                and (include_internal or not d.startswith("."))
            ]
            for fn in filenames:
                if not include_internal and fn.startswith("."):
                    continue
                p = Path(dirpath) / fn
                try:
                    found.append(self.artifact(p))
                except (OSError, ValueError):
                    continue
        found.sort(key=lambda a: -a.mtime if newest_first else a.mtime)
        return found

    # -- the context-facing view -------------------------------------------
    def tree(self, *, max_entries: int = 60) -> str:
        arts = self.artifacts(newest_first=False)
        lines = [f"{a.rel}  ({self._human(a.size)})" for a in arts[:max_entries]]
        if len(arts) > max_entries:
            lines.append(f"... and {len(arts) - max_entries} more files")
        return "\n".join(lines) if lines else "(empty workspace)"

    def digest(self, *, max_files: int | None = None) -> str:
        """File tree plus a short per-artefact summary. Never file bodies.

        This is what the ``WORKSPACE`` ledger slot receives. The hard rule — tree
        and digests, never contents — is what keeps a 40MB parquet from costing
        anything in the prompt.
        """
        cap = max_files or self.max_digest_files
        arts = self.artifacts(newest_first=True)
        head = [f"workspace root: {self.root.name}", f"files: {len(arts)}", ""]
        body: list[str] = []
        for a in arts[:cap]:
            line = f"- {a.rel} ({self._human(a.size)})"
            if a.note:
                line += f" — {a.note}"
            preview = self._preview(a)
            if preview:
                line += f"\n    {preview}"
            body.append(line)
        if len(arts) > cap:
            body.append(f"- ... {len(arts) - cap} more files not listed")
        return "\n".join([*head, *body]) if arts else "(empty workspace)"

    def _preview(self, a: Artifact) -> str:
        if a.path.suffix.lower() not in TEXTLIKE:
            return ""
        try:
            with a.path.open("r", encoding="utf-8", errors="replace") as fh:
                blob = fh.read(MAX_DIGEST_BYTES)
        except OSError:
            return ""
        first = " ".join(blob.split())
        return (first[:180] + "…") if len(first) > 180 else first

    @staticmethod
    def _human(n: float) -> str:
        size = float(n)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.0f}B" if unit == "B" else f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}GB"

    # -- checkpointing -----------------------------------------------------
    def snapshot_manifest(self) -> dict[str, object]:
        """Content-addressed listing, used to detect drift on resume."""
        return {
            "root": str(self.root),
            "created": time.time(),
            # Manifests cover internal files too: drift detection must see
            # everything, even what the agent is not shown.
            "files": [
                a.to_json()
                for a in self.artifacts(newest_first=False, include_internal=True)
            ],
        }

    def clear(self) -> None:
        """Empty the workspace in place.

        Prefer a fresh workspace path over clearing and reusing one when a
        container sandbox is in play. Docker Desktop's shared filesystem caches
        directory entries in the guest, so a file the *host* deletes and the
        *container* then recreates at the same path can fail to open — the guest
        is still holding the stale entry. Distinct paths per run avoid the whole
        class of problem, cost nothing, and leave the previous run's artefacts
        available for inspection.
        """
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.notes.clear()

    def copy_in(self, sources: Iterable[Path], *, subdir: str = "inputs") -> list[Artifact]:
        out: list[Artifact] = []
        for src in sources:
            src = Path(src)
            dest = self.resolve(f"{subdir}/{src.name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            out.append(self.artifact(dest))
        return out
