"""Sliding-window quota tracking with proactive avoidance.

Free tiers are limited by requests-per-minute and requests-per-day, not by
dollars. The router must therefore know *before* it calls whether a provider is
close to its limit, because discovering the limit by getting a 429 wastes both
wall-clock and the retry budget.

State persists to disk so a run that starts an hour after the previous one does
not re-learn the day's consumption from scratch.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QuotaLimit:
    per_minute: int | None = None
    per_day: int | None = None
    # Fraction of the limit at which the router starts preferring alternatives.
    soft_ratio: float = 0.85


@dataclass
class _Window:
    stamps: list[float] = field(default_factory=list)

    def prune(self, horizon_s: float, now: float) -> None:
        cutoff = now - horizon_s
        self.stamps = [t for t in self.stamps if t >= cutoff]

    def count(self, horizon_s: float, now: float) -> int:
        self.prune(horizon_s, now)
        return len(self.stamps)


class QuotaTracker:
    MINUTE = 60.0
    DAY = 86_400.0

    def __init__(self, limits: dict[str, QuotaLimit], state_path: Path | None = None) -> None:
        self.limits = limits
        self.state_path = state_path
        self._windows: dict[str, _Window] = {}
        self._cooldown_until: dict[str, float] = {}
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        now = time.time()
        for name, stamps in data.get("windows", {}).items():
            w = _Window([float(t) for t in stamps])
            w.prune(self.DAY, now)
            self._windows[name] = w
        self._cooldown_until = {
            k: float(v) for k, v in data.get("cooldown_until", {}).items() if float(v) > now
        }

    def save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "windows": {k: v.stamps for k, v in self._windows.items()},
            "cooldown_until": self._cooldown_until,
        }
        self.state_path.write_text(json.dumps(payload), encoding="utf-8")

    # -- accounting --------------------------------------------------------
    def _window(self, name: str) -> _Window:
        return self._windows.setdefault(name, _Window())

    def note_call(self, name: str, *, now: float | None = None) -> None:
        self._window(name).stamps.append(now or time.time())

    def note_rate_limited(self, name: str, retry_after_s: float | None = None) -> None:
        """A 429 is authoritative: back off for at least the advertised window."""
        wait = retry_after_s if retry_after_s and retry_after_s > 0 else 60.0
        self._cooldown_until[name] = time.time() + wait

    # -- queries -----------------------------------------------------------
    def in_cooldown(self, name: str, *, now: float | None = None) -> bool:
        return (now or time.time()) < self._cooldown_until.get(name, 0.0)

    def usage(self, name: str, *, now: float | None = None) -> dict[str, int]:
        now = now or time.time()
        w = self._window(name)
        return {"minute": w.count(self.MINUTE, now), "day": w.count(self.DAY, now)}

    def headroom(self, name: str, *, now: float | None = None) -> float:
        """1.0 = untouched, 0.0 = at the limit. Used to rank equal-tier providers."""
        limit = self.limits.get(name)
        if limit is None:
            return 1.0
        used = self.usage(name, now=now)
        ratios: list[float] = []
        if limit.per_minute:
            ratios.append(used["minute"] / limit.per_minute)
        if limit.per_day:
            ratios.append(used["day"] / limit.per_day)
        return max(0.0, 1.0 - max(ratios, default=0.0))

    def is_exhausted(self, name: str, *, now: float | None = None) -> bool:
        limit = self.limits.get(name)
        if limit is None:
            return False
        used = self.usage(name, now=now)
        if limit.per_minute and used["minute"] >= limit.per_minute:
            return True
        if limit.per_day and used["day"] >= limit.per_day:
            return True
        return False

    def is_soft_limited(self, name: str, *, now: float | None = None) -> bool:
        limit = self.limits.get(name)
        if limit is None:
            return False
        return self.headroom(name, now=now) <= (1.0 - limit.soft_ratio)

    def report(self) -> dict[str, dict[str, object]]:
        now = time.time()
        return {
            name: {
                "usage": self.usage(name, now=now),
                "headroom": round(self.headroom(name, now=now), 3),
                "cooldown": self.in_cooldown(name, now=now),
                "exhausted": self.is_exhausted(name, now=now),
            }
            for name in sorted(set(self._windows) | set(self.limits))
        }
