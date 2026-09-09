"""Run checkpointing and resume.

Because agent state lives in the workspace rather than in the conversation, a
checkpoint is small: the step counter, the history digest, the memory pointer and
a manifest of the workspace. Resuming is then re-reading the workspace, not
replaying the transcript — which is the property that makes long-horizon runs
survivable.

The manifest is content-addressed (sha256 prefixes). On resume we compare it
against the workspace as it exists now and report drift, because silently
resuming onto a workspace someone edited by hand produces results that cannot be
reproduced, and finding that out later is expensive.

Writes are atomic and versioned: ``checkpoint-<n>.json`` plus a ``latest.json``
pointer. Keeping every step's checkpoint rather than overwriting one lets a run
be rewound to any step, which is how a bad step gets re-run in an ablation
without re-running the whole case.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from teamclaw.execution.workspace import Workspace


@dataclass
class Checkpoint:
    run_id: str
    step: int
    objective: str
    history: list[dict[str, str]] = field(default_factory=list)
    scratch: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    finished: bool = False
    final_answer: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step": self.step,
            "objective": self.objective,
            "history": self.history,
            "scratch": self.scratch,
            "manifest": self.manifest,
            "created_at": self.created_at,
            "finished": self.finished,
            "final_answer": self.final_answer,
        }

    @staticmethod
    def from_json(data: dict[str, Any]) -> "Checkpoint":
        return Checkpoint(
            run_id=str(data.get("run_id", "")),
            step=int(data.get("step", 0)),
            objective=str(data.get("objective", "")),
            history=list(data.get("history") or []),
            scratch=dict(data.get("scratch") or {}),
            manifest=dict(data.get("manifest") or {}),
            created_at=float(data.get("created_at", time.time())),
            finished=bool(data.get("finished", False)),
            final_answer=str(data.get("final_answer", "")),
        )


@dataclass
class DriftReport:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.added or self.removed or self.modified)

    def summary(self) -> str:
        if self.clean:
            return "workspace matches the checkpoint manifest"
        bits = []
        if self.added:
            bits.append(f"{len(self.added)} added")
        if self.removed:
            bits.append(f"{len(self.removed)} removed")
        if self.modified:
            bits.append(f"{len(self.modified)} modified")
        return "workspace drifted from manifest: " + ", ".join(bits)

    def to_json(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "added": self.added[:20],
            "removed": self.removed[:20],
            "modified": self.modified[:20],
        }


class CheckpointStore:
    def __init__(self, run_dir: Path) -> None:
        self.dir = Path(run_dir) / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _atomic_write(self, path: Path, payload: dict[str, Any]) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=1, default=str)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def save(self, cp: Checkpoint) -> Path:
        path = self.dir / f"checkpoint-{cp.step:04d}.json"
        self._atomic_write(path, cp.to_json())
        self._atomic_write(self.dir / "latest.json", cp.to_json())
        return path

    def load_latest(self) -> Checkpoint | None:
        latest = self.dir / "latest.json"
        if latest.exists():
            try:
                return Checkpoint.from_json(json.loads(latest.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass  # fall through to the numbered checkpoints
        return self._load_highest()

    def _load_highest(self) -> Checkpoint | None:
        candidates = sorted(self.dir.glob("checkpoint-*.json"), reverse=True)
        for path in candidates:
            try:
                return Checkpoint.from_json(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue  # a torn checkpoint: try the one before it
        return None

    def load_step(self, step: int) -> Checkpoint | None:
        path = self.dir / f"checkpoint-{step:04d}.json"
        if not path.exists():
            return None
        try:
            return Checkpoint.from_json(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return None

    def steps(self) -> list[int]:
        out: list[int] = []
        for p in self.dir.glob("checkpoint-*.json"):
            try:
                out.append(int(p.stem.split("-")[1]))
            except (IndexError, ValueError):
                continue
        return sorted(out)


def check_drift(cp: Checkpoint, workspace: Workspace) -> DriftReport:
    """Compare a checkpoint's manifest against the workspace on disk."""
    before = {f["path"]: f.get("sha256", "") for f in cp.manifest.get("files", [])}
    now = {a.rel: a.sha256[:12] for a in workspace.artifacts()}
    report = DriftReport()
    for path, digest in now.items():
        if path not in before:
            report.added.append(path)
        elif before[path] and digest != before[path]:
            report.modified.append(path)
    for path in before:
        if path not in now:
            report.removed.append(path)
    return report
