"""Deterministic task graph.

This is the *other* way to organise multi-step work: instead of letting the model
decide the next step, fix the steps and let it decide only inside them. The
platform supports both because the interesting question is not which is better
but where the crossover is — and answering that needs both arms implemented
against the same tools and the same eval. This module is the ``Workflow-C``
ablation arm.

A node is a callable plus its dependencies. The graph topologically sorts, runs,
records each node's duration and output digest, and short-circuits descendants of
a failed node rather than running them on missing inputs.

There is no parallel executor here for the same reason
:meth:`SubAgentFactory.fan_out` is sequential: on a free tier the limit is
requests per minute, so concurrency buys 429s.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from tally.observability.trace import SpanKind, Tracer

NodeFn = Callable[[dict[str, Any]], Any]


class CyclicGraph(ValueError):
    pass


class UnknownDependency(ValueError):
    pass


@dataclass
class Node:
    name: str
    fn: NodeFn
    depends_on: tuple[str, ...] = ()
    description: str = ""
    optional: bool = False   # a failure here does not block descendants


@dataclass
class NodeResult:
    name: str
    ok: bool
    duration_s: float
    output: Any = None
    error: str = ""
    skipped: bool = False
    skip_reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "node": self.name,
            "ok": self.ok,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "duration_s": round(self.duration_s, 3),
            "error": self.error[:300],
            "output_type": type(self.output).__name__ if self.output is not None else None,
        }


@dataclass
class GraphResult:
    results: dict[str, NodeResult] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok or r.skipped for r in self.results.values())

    @property
    def outputs(self) -> dict[str, Any]:
        return {k: r.output for k, r in self.results.items() if r.ok}

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "order": self.order,
            "nodes": [self.results[n].to_json() for n in self.order if n in self.results],
            "failed": [n for n, r in self.results.items() if not r.ok and not r.skipped],
            "skipped": [n for n, r in self.results.items() if r.skipped],
        }


@dataclass
class TaskGraph:
    nodes: dict[str, Node] = field(default_factory=dict)

    def add(
        self,
        name: str,
        fn: NodeFn,
        *,
        depends_on: Sequence[str] = (),
        description: str = "",
        optional: bool = False,
    ) -> "TaskGraph":
        self.nodes[name] = Node(
            name=name, fn=fn, depends_on=tuple(depends_on),
            description=description, optional=optional,
        )
        return self

    # -- ordering ----------------------------------------------------------
    def topological_order(self) -> list[str]:
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep not in self.nodes:
                    raise UnknownDependency(f"{node.name!r} depends on unknown node {dep!r}")

        # Kahn's algorithm, with names sorted at each level so a graph always
        # runs in the same order — an ablation arm that reorders between runs is
        # not reproducible.
        indegree = {n: len(node.depends_on) for n, node in self.nodes.items()}
        ready = sorted(n for n, d in indegree.items() if d == 0)
        order: list[str] = []
        while ready:
            name = ready.pop(0)
            order.append(name)
            for other, node in sorted(self.nodes.items()):
                if name in node.depends_on:
                    indegree[other] -= 1
                    if indegree[other] == 0:
                        ready.append(other)
            ready.sort()
        if len(order) != len(self.nodes):
            remaining = sorted(set(self.nodes) - set(order))
            raise CyclicGraph(f"cycle among nodes: {remaining}")
        return order

    # -- execution ---------------------------------------------------------
    def run(
        self,
        *,
        context: dict[str, Any] | None = None,
        tracer: Tracer | None = None,
    ) -> GraphResult:
        order = self.topological_order()
        state: dict[str, Any] = dict(context or {})
        out = GraphResult(order=order)

        for name in order:
            node = self.nodes[name]
            blocked = [
                dep for dep in node.depends_on
                if dep in out.results and not out.results[dep].ok
                and not self.nodes[dep].optional
            ]
            if blocked:
                out.results[name] = NodeResult(
                    name=name, ok=False, duration_s=0.0, skipped=True,
                    skip_reason=f"upstream failed: {', '.join(blocked)}",
                )
                continue

            started = time.time()
            span_cm = (
                tracer.span(SpanKind.STEP, f"node:{name}")
                if tracer is not None
                else _NullSpan()
            )
            with span_cm as span:
                try:
                    value = node.fn(state)
                    state[name] = value
                    result = NodeResult(name=name, ok=True,
                                        duration_s=time.time() - started, output=value)
                except Exception as exc:  # noqa: BLE001 - recorded, then reported
                    result = NodeResult(
                        name=name, ok=False, duration_s=time.time() - started,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                if span is not None:
                    span.set(**result.to_json())
            out.results[name] = result

        return out

    def describe(self) -> str:
        lines = []
        for name in self.topological_order():
            node = self.nodes[name]
            deps = f" <- {', '.join(node.depends_on)}" if node.depends_on else ""
            flag = " (optional)" if node.optional else ""
            lines.append(f"{name}{deps}{flag}: {node.description or '-'}")
        return "\n".join(lines)


class _NullSpan:
    """Context manager that yields None when no tracer is supplied."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None
