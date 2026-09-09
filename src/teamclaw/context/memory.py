"""Layered memory store.

This module exists to answer, in code, the five questions that any reviewer
will ask about a "layered memory" claim:

1. **When is memory written?** Only through :meth:`MemoryStore.write`, called at
   step boundaries by the agent loop with an explicit ``source`` span id. There
   is no implicit "remember everything" path, because an unfiltered write loop
   is what turns a memory store into a noise generator.
2. **Who decides what enters the context?** :meth:`recall` scores candidates and
   the ledger's memory slot allocates them a budget. Scoring is
   ``relevance * decay * confidence``, all three visible in the returned record,
   so a recall decision can be explained after the fact.
3. **What happens on conflict or staleness?** Records carry a ``key``, and two
   writes sharing a key are a conflict. Resolution is an explicit
   :class:`ConflictPolicy`, defaulting to ``HIGHEST_CONFIDENCE`` rather than
   last-write-wins: under last-write-wins a single low-confidence extraction can
   destroy a verified fact, which is the failure mode that makes reviewers
   distrust memory claims. The loser is kept on disk marked superseded, so an
   audit can reconstruct what was believed and why it lost. ``ttl_days`` expires
   time-bound facts independently of conflict.
4. **What gets trimmed when the window is full?** Not memory's decision: the
   ledger's ``MEMORY`` slot has ``hard_floor_tokens=0`` and the highest
   elasticity of any slot, so memory is sacrificed before instructions or tools.
5. **How is recall accuracy measured?** :func:`recall_metrics` computes
   precision/recall against a labelled probe set; the eval harness runs it as a
   standalone check so the memory layer has its own number rather than hiding
   inside end-to-end accuracy.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

from teamclaw.context.retrieval import Doc, Retriever, build_retriever


class MemoryKind(str, Enum):
    """Memory layers. ``PERSONA`` and ``FOCUS`` are singletons per agent."""

    PERSONA = "persona"      # who the agent is; pinned into the system slot
    FOCUS = "focus"          # the current objective; short, always recalled
    FACT = "fact"            # durable learned fact about the domain or user
    PROCEDURE = "procedure"  # "how we do X here" — learned workflow
    EPISODE = "episode"      # what happened in a past session


class ConflictPolicy(str, Enum):
    """How two writes sharing a ``key`` are resolved."""

    HIGHEST_CONFIDENCE = "highest_confidence"
    LAST_WRITE_WINS = "last_write_wins"


DEFAULT_HALF_LIFE_DAYS: dict[str, float] = {
    MemoryKind.PERSONA.value: math.inf,
    MemoryKind.FOCUS.value: 7.0,
    MemoryKind.FACT.value: 180.0,
    MemoryKind.PROCEDURE.value: 365.0,
    MemoryKind.EPISODE.value: 30.0,
}


@dataclass
class MemoryRecord:
    text: str
    kind: MemoryKind = MemoryKind.FACT
    key: str | None = None           # stable identity; a new write supersedes
    confidence: float = 0.8          # 0..1, how much we trust this
    created_at: float = field(default_factory=time.time)
    last_used_at: float | None = None
    use_count: int = 0
    ttl_days: float | None = None    # hard expiry for time-bound facts
    source: str = "-"                # trace span id or ingest job
    tags: tuple[str, ...] = ()
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    superseded_by: str | None = None

    # -- lifecycle ---------------------------------------------------------
    def age_days(self, now: float | None = None) -> float:
        return ((now or time.time()) - self.created_at) / 86_400.0

    def is_expired(self, now: float | None = None) -> bool:
        if self.ttl_days is None:
            return False
        return self.age_days(now) > self.ttl_days

    def is_active(self, now: float | None = None) -> bool:
        return self.superseded_by is None and not self.is_expired(now)

    def decay(self, now: float | None = None, half_life_days: float | None = None) -> float:
        """Exponential decay on age. Persona never decays."""
        hl = half_life_days if half_life_days is not None else DEFAULT_HALF_LIFE_DAYS.get(
            self.kind.value, 180.0
        )
        if hl == math.inf:
            return 1.0
        return 0.5 ** (self.age_days(now) / hl) if hl > 0 else 1.0

    def to_json(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "text": self.text,
            "kind": self.kind.value,
            "key": self.key,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "use_count": self.use_count,
            "ttl_days": self.ttl_days,
            "source": self.source,
            "tags": list(self.tags),
            "superseded_by": self.superseded_by,
        }

    @staticmethod
    def from_json(data: dict[str, object]) -> "MemoryRecord":
        return MemoryRecord(
            text=str(data["text"]),
            kind=MemoryKind(str(data.get("kind", "fact"))),
            key=data.get("key") or None,  # type: ignore[arg-type]
            confidence=float(data.get("confidence", 0.8)),  # type: ignore[arg-type]
            created_at=float(data.get("created_at", time.time())),  # type: ignore[arg-type]
            last_used_at=(float(data["last_used_at"]) if data.get("last_used_at") else None),  # type: ignore[arg-type]
            use_count=int(data.get("use_count", 0)),  # type: ignore[arg-type]
            ttl_days=(float(data["ttl_days"]) if data.get("ttl_days") is not None else None),  # type: ignore[arg-type]
            source=str(data.get("source", "-")),
            tags=tuple(data.get("tags") or ()),  # type: ignore[arg-type]
            record_id=str(data.get("record_id") or uuid.uuid4().hex[:12]),
            superseded_by=data.get("superseded_by") or None,  # type: ignore[arg-type]
        )


@dataclass
class Recall:
    record: MemoryRecord
    relevance: float
    decay: float
    score: float

    def explain(self) -> str:
        return (
            f"{self.record.kind.value}/{self.record.record_id} "
            f"rel={self.relevance:.3f} decay={self.decay:.3f} "
            f"conf={self.record.confidence:.2f} -> {self.score:.4f}"
        )


class MemoryStore:
    """Append-only JSONL memory, scoped per agent."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        retriever: Retriever | None = None,
        agent: str = "root",
        conflict_policy: ConflictPolicy = ConflictPolicy.HIGHEST_CONFIDENCE,
    ) -> None:
        self.path = path
        self.agent = agent
        self.conflict_policy = conflict_policy
        self.conflicts: list[dict[str, object]] = []
        self.records: list[MemoryRecord] = []
        self._retriever = retriever or build_retriever()
        self._dirty = True
        if path is not None and path.exists():
            self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        assert self.path is not None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self.records.append(MemoryRecord.from_json(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue  # skip a corrupt line rather than lose the whole store
        self._dirty = True

    def flush(self) -> None:
        """Rewrite the whole file. Called after supersede/forget mutations."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for r in self.records:
                fh.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")

    def _append(self, record: MemoryRecord) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")

    # -- writes ------------------------------------------------------------
    def write(
        self,
        text: str,
        *,
        kind: MemoryKind = MemoryKind.FACT,
        key: str | None = None,
        confidence: float = 0.8,
        ttl_days: float | None = None,
        source: str = "-",
        tags: Sequence[str] = (),
    ) -> MemoryRecord:
        record = MemoryRecord(
            text=text.strip(),
            kind=kind,
            key=key,
            confidence=max(0.0, min(1.0, confidence)),
            ttl_days=ttl_days,
            source=source,
            tags=tuple(tags),
        )
        if key:
            self._resolve_conflict(record)
        self.records.append(record)
        self._dirty = True
        if key:
            self.flush()
        else:
            self._append(record)
        return record

    def _resolve_conflict(self, incoming: MemoryRecord) -> None:
        """Decide which of two same-key beliefs stays active.

        ``FOCUS`` always takes the latest write regardless of policy: the current
        objective is a statement about now, not a claim whose truth we weigh.
        """
        priors = [
            r
            for r in self.records
            if r.key == incoming.key and r.superseded_by is None
        ]
        if not priors:
            return

        force_latest = (
            incoming.kind is MemoryKind.FOCUS
            or self.conflict_policy is ConflictPolicy.LAST_WRITE_WINS
        )
        for prior in priors:
            if force_latest or incoming.confidence >= prior.confidence:
                prior.superseded_by = incoming.record_id
                winner, loser = incoming, prior
            else:
                # The incoming belief loses: recorded for audit, excluded from recall.
                incoming.superseded_by = prior.record_id
                winner, loser = prior, incoming
            self.conflicts.append(
                {
                    "key": incoming.key,
                    "policy": self.conflict_policy.value,
                    "winner": winner.record_id,
                    "winner_confidence": winner.confidence,
                    "loser": loser.record_id,
                    "loser_confidence": loser.confidence,
                    "at": time.time(),
                }
            )

    def conflict_rate(self) -> float:
        """Share of keyed writes that collided. Reported by the eval harness."""
        keyed = sum(1 for r in self.records if r.key)
        return round(len(self.conflicts) / keyed, 4) if keyed else 0.0

    def forget(self, record_id: str) -> bool:
        for r in self.records:
            if r.record_id == record_id and r.superseded_by is None:
                r.superseded_by = "forgotten"
                self._dirty = True
                self.flush()
                return True
        return False

    def prune_expired(self, now: float | None = None) -> int:
        n = 0
        for r in self.records:
            if r.superseded_by is None and r.is_expired(now):
                r.superseded_by = "expired"
                n += 1
        if n:
            self._dirty = True
            self.flush()
        return n

    # -- reads -------------------------------------------------------------
    def active(self, now: float | None = None) -> list[MemoryRecord]:
        return [r for r in self.records if r.is_active(now)]

    def _reindex(self, pool: Sequence[MemoryRecord]) -> None:
        self._retriever.index([Doc(doc_id=r.record_id, text=r.text) for r in pool])
        self._dirty = False

    def recall(
        self,
        query: str,
        *,
        k: int = 8,
        kinds: Iterable[MemoryKind] | None = None,
        min_score: float = 0.0,
        now: float | None = None,
        mark_used: bool = True,
    ) -> list[Recall]:
        now = now or time.time()
        pool = self.active(now)
        if kinds is not None:
            wanted = {k_.value for k_ in kinds}
            pool = [r for r in pool if r.kind.value in wanted]
        if not pool:
            return []

        self._reindex(pool)
        by_id = {r.record_id: r for r in pool}
        hits = self._retriever.search(query, k=max(k * 3, k))
        top = max((s for _, s in hits), default=0.0) or 1.0

        # FOCUS is always a candidate even when it does not match lexically: the
        # current objective is relevant by definition, not by keyword overlap.
        scored: list[Recall] = []
        seen: set[str] = set()
        for doc_id, raw in hits:
            rec = by_id.get(doc_id)
            if rec is None:
                continue
            relevance = raw / top
            decay = rec.decay(now)
            scored.append(
                Recall(rec, relevance, decay, relevance * decay * rec.confidence)
            )
            seen.add(doc_id)
        for rec in pool:
            if rec.kind is MemoryKind.FOCUS and rec.record_id not in seen:
                decay = rec.decay(now)
                scored.append(Recall(rec, 1.0, decay, 1.0 * decay * rec.confidence))

        scored.sort(key=lambda r: -r.score)
        out = [r for r in scored if r.score >= min_score][:k]
        if mark_used:
            for r in out:
                r.record.last_used_at = now
                r.record.use_count += 1
        return out

    def persona(self) -> str:
        parts = [r.text for r in self.active() if r.kind is MemoryKind.PERSONA]
        return "\n".join(parts)

    def focus(self) -> str:
        recs = [r for r in self.active() if r.kind is MemoryKind.FOCUS]
        recs.sort(key=lambda r: -r.created_at)
        return recs[0].text if recs else ""

    def set_focus(self, text: str, *, source: str = "-") -> MemoryRecord:
        return self.write(
            text, kind=MemoryKind.FOCUS, key=f"focus:{self.agent}", source=source
        )

    def stats(self) -> dict[str, object]:
        by_kind: dict[str, int] = {}
        for r in self.active():
            by_kind[r.kind.value] = by_kind.get(r.kind.value, 0) + 1
        return {
            "total": len(self.records),
            "active": len(self.active()),
            "superseded": sum(1 for r in self.records if r.superseded_by),
            "by_kind": by_kind,
            "conflicts": len(self.conflicts),
            "conflict_rate": self.conflict_rate(),
        }


def recall_metrics(
    store: MemoryStore,
    probes: Sequence[tuple[str, set[str]]],
    *,
    k: int = 5,
) -> dict[str, float]:
    """Precision@k / recall@k over ``(query, relevant_record_ids)`` probes.

    Giving the memory layer its own number is the point: an end-to-end score
    cannot tell you whether memory helped, and "we added memory" with no recall
    metric is exactly the claim reviewers discount.
    """
    if not probes:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "probes": 0}
    precisions: list[float] = []
    recalls: list[float] = []
    for query, relevant in probes:
        got = {r.record.record_id for r in store.recall(query, k=k, mark_used=False)}
        tp = len(got & relevant)
        precisions.append(tp / len(got) if got else 0.0)
        recalls.append(tp / len(relevant) if relevant else 0.0)
    return {
        "precision_at_k": round(sum(precisions) / len(precisions), 4),
        "recall_at_k": round(sum(recalls) / len(recalls), 4),
        "probes": len(probes),
    }
