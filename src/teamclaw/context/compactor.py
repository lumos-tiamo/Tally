"""History compaction, triggered only at step boundaries.

Two decisions here are load-bearing and both are stated in the spec:

**When.** Compaction runs when ledger utilisation crosses a threshold *and* the
agent is between steps — never mid-reasoning. Compressing while a tool result is
still being interpreted destroys the reasoning chain the model is in the middle
of, which shows up as the agent "forgetting" what it was doing. Checking at step
boundaries costs at most one oversized prompt and keeps the chain intact.

**Where the output goes.** The summary is *written to the workspace* and the
replaced turns are archived alongside it, not discarded. So compaction is
reversible: a later step can re-read the archive, and an auditor can see exactly
what was dropped. Losing the original is what makes summarisation feel lossy;
persisting it makes it a cache eviction instead.

The middle of the history is compacted, not the head or the tail: the head holds
the task framing and the tail holds what just happened, while the middle is where
redundancy accumulates.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from teamclaw.context.slots import Item
from teamclaw.models.base import Message, Purpose, Role
from teamclaw.observability.trace import SpanKind, Tracer

SUMMARY_SYSTEM = (
    "You compress an agent's working history. Preserve, in this order of priority:\n"
    "1. Concrete facts and figures already established, with their sources.\n"
    "2. Artefact paths written to the workspace and what each contains.\n"
    "3. Decisions taken and the reason for each.\n"
    "4. Failed approaches and why they failed, so they are not retried.\n"
    "Drop: pleasantries, restated instructions, and verbatim tool output that is "
    "already saved to an artefact.\n"
    "Write terse bullet points. Never invent a fact that is not in the input."
)


@dataclass
class CompactionResult:
    summary: str
    replaced: int
    tokens_before: int
    tokens_after: int
    archive_path: Path | None = None
    used_llm: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def saved_tokens(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def compression_ratio(self) -> float:
        return (self.tokens_after / self.tokens_before) if self.tokens_before else 1.0

    def to_json(self) -> dict[str, object]:
        return {
            "replaced_turns": self.replaced,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "saved_tokens": self.saved_tokens,
            "compression_ratio": round(self.compression_ratio, 4),
            "archive": str(self.archive_path) if self.archive_path else None,
            "used_llm": self.used_llm,
            "notes": self.notes,
        }


class Compactor:
    def __init__(
        self,
        *,
        router: object | None = None,
        workspace_dir: Path | None = None,
        tracer: Tracer | None = None,
        keep_head: int = 2,
        keep_tail: int = 6,
        max_summary_tokens: int = 700,
    ) -> None:
        self.router = router
        self.workspace_dir = workspace_dir
        self.tracer = tracer
        self.keep_head = keep_head
        self.keep_tail = keep_tail
        self.max_summary_tokens = max_summary_tokens
        self.history: list[CompactionResult] = []

    # -- eligibility -------------------------------------------------------
    def should_compact(
        self,
        *,
        utilisation: float,
        at_step_boundary: bool,
        turns: int,
        threshold: float = 0.70,
    ) -> bool:
        """Both conditions are required; neither alone is sufficient."""
        if not at_step_boundary:
            return False
        if utilisation < threshold:
            return False
        # Nothing in the middle to compact.
        return turns > self.keep_head + self.keep_tail + 1

    # -- compaction --------------------------------------------------------
    def compact(
        self,
        items: Sequence[Item],
        *,
        counter,
        task: str = "-",
    ) -> tuple[list[Item], CompactionResult]:
        head = list(items[: self.keep_head])
        tail = list(items[-self.keep_tail :]) if self.keep_tail else []
        middle = list(items[self.keep_head : len(items) - len(tail)])

        if not middle:
            result = CompactionResult(
                summary="",
                replaced=0,
                tokens_before=sum(i.tokens for i in items),
                tokens_after=sum(i.tokens for i in items),
                used_llm=False,
                notes=["nothing in the middle to compact"],
            )
            return list(items), result

        before = sum(i.tokens for i in middle)
        transcript = "\n\n".join(
            f"[{i.kind or 'turn'}] {i.text}" for i in middle
        )

        summary, used_llm, notes = self._summarise(transcript, task=task)
        summary_tokens = counter(summary)

        # A compaction that does not shrink the context is worse than none: it
        # spends a summarisation call, loses detail, and leaves the window just as
        # full. The extractive fallback in particular can grow the text when every
        # line looks informative. Abort and keep the originals.
        if summary_tokens >= before:
            result = CompactionResult(
                summary=summary,
                replaced=0,
                tokens_before=before,
                tokens_after=before,
                used_llm=used_llm,
                notes=[
                    *notes,
                    f"aborted: summary ({summary_tokens} tok) would not shrink "
                    f"the {before} tok it replaces",
                ],
            )
            self.history.append(result)
            if self.tracer is not None:
                with self.tracer.span(SpanKind.COMPACTION, "history:aborted") as sp:
                    sp.set(**result.to_json())
            return list(items), result

        archive = self._archive(middle, summary)

        stand_in = Item(
            text=f"[compacted {len(middle)} earlier turns]\n{summary}",
            tokens=summary_tokens,
            score=float("inf"),  # a summary is never the next thing evicted
            pinned=True,
            label="compaction-summary",
            kind="user",
        )
        after = stand_in.tokens
        result = CompactionResult(
            summary=summary,
            replaced=len(middle),
            tokens_before=before,
            tokens_after=after,
            archive_path=archive,
            used_llm=used_llm,
            notes=notes,
        )
        self.history.append(result)

        if self.tracer is not None:
            with self.tracer.span(SpanKind.COMPACTION, "history") as sp:
                sp.set(**result.to_json())

        return [*head, stand_in, *tail], result

    # -- summarisation -----------------------------------------------------
    def _summarise(self, transcript: str, *, task: str) -> tuple[str, bool, list[str]]:
        if self.router is None:
            return self._extractive(transcript), False, ["no router: extractive fallback"]
        try:
            completion = self.router.complete(  # type: ignore[attr-defined]
                Purpose.SUMMARIZE,
                [
                    Message(Role.SYSTEM, SUMMARY_SYSTEM),
                    Message.user(f"Task: {task}\n\nHistory to compress:\n\n{transcript}"),
                ],
                max_tokens=self.max_summary_tokens,
                task=task,
            )
            text = completion.text.strip()
            if not text:
                return self._extractive(transcript), False, ["empty summary: extractive fallback"]
            return text, True, []
        except Exception as exc:  # noqa: BLE001
            # A compaction failure must not kill the run: degrade to extractive.
            # Losing summary quality is recoverable; losing the run is not.
            return (
                self._extractive(transcript),
                False,
                [f"summarisation failed ({type(exc).__name__}): extractive fallback"],
            )

    @staticmethod
    def _extractive(transcript: str) -> str:
        """Dependency-free fallback: keep lines that look like facts or paths.

        Deliberately crude. Its job is to keep a run alive when every provider is
        rate-limited, not to match an LLM summary.
        """
        keep: list[str] = []
        for line in transcript.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            informative = (
                any(ch.isdigit() for ch in stripped)
                or "workspace/" in stripped
                or stripped.startswith(("-", "*", "#"))
                or "error" in stripped.lower()
                or "failed" in stripped.lower()
            )
            if informative:
                keep.append(stripped[:300])
        if not keep:
            keep = [transcript[:600]]
        return "\n".join(f"- {line}" for line in keep[:40])

    # -- archival ----------------------------------------------------------
    def _archive(self, middle: Sequence[Item], summary: str) -> Path | None:
        if self.workspace_dir is None:
            return None
        target = self.workspace_dir / "compaction"
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"compaction_{int(time.time())}_{len(middle)}turns.json"
        path.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "replaced": [
                        {"kind": i.kind, "label": i.label, "tokens": i.tokens, "text": i.text}
                        for i in middle
                    ],
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        return path

    def stats(self) -> dict[str, object]:
        if not self.history:
            return {"compactions": 0, "tokens_saved": 0}
        return {
            "compactions": len(self.history),
            "tokens_saved": sum(r.saved_tokens for r in self.history),
            "turns_replaced": sum(r.replaced for r in self.history),
            "llm_summaries": sum(1 for r in self.history if r.used_llm),
            "extractive_fallbacks": sum(1 for r in self.history if not r.used_llm),
            "aborted": sum(1 for r in self.history if r.replaced == 0),
            "mean_compression_ratio": round(
                sum(r.compression_ratio for r in self.history) / len(self.history), 4
            ),
        }
