"""On-disk completion cache keyed by request fingerprint.

Sharded two levels deep by hash prefix so a few hundred thousand entries do not
land in one directory. Writes are atomic (temp file + rename) so an interrupted
run cannot leave a half-written JSON that poisons later reads.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from tally.models.base import Completion


class CompletionCache:
    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if enabled:
            root.mkdir(parents=True, exist_ok=True)

    def _path(self, fingerprint: str) -> Path:
        return self.root / fingerprint[:2] / fingerprint[2:4] / f"{fingerprint}.json"

    def get(self, fingerprint: str) -> Completion | None:
        if not self.enabled:
            return None
        path = self._path(fingerprint)
        if not path.exists():
            self.misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Corrupt entry: treat as a miss and let it be overwritten.
            self.misses += 1
            return None
        self.hits += 1
        return Completion(
            text=data["text"],
            model=data["model"],
            provider=data["provider"],
            tokens_in=data.get("tokens_in", 0),
            tokens_out=data.get("tokens_out", 0),
            cached=True,
            finish_reason=data.get("finish_reason", "stop"),
        )

    def put(self, fingerprint: str, completion: Completion) -> None:
        if not self.enabled:
            return
        path = self._path(fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "text": completion.text,
            "model": completion.model,
            "provider": completion.provider,
            "tokens_in": completion.tokens_in,
            "tokens_out": completion.tokens_out,
            "finish_reason": completion.finish_reason,
        }
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            os.replace(tmp, path)
            self.writes += 1
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, float | int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_rate": round(self.hit_rate, 4),
        }
