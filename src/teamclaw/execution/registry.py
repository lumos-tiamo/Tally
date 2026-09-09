"""Tool Module Registry.

The registry holds tool *specifications*, and it is the thing that lets the
context ledger keep the tools slot cheap. Three representations of the same tool
exist, at very different token costs:

===================  ============================================  ==========
representation       content                                        cost
===================  ============================================  ==========
``signature_line``   ``sec.search_filings(cik, form, years) -> …``  ~15 tok
``stub_source``      def + full docstring, importable in sandbox    ~120 tok
full JSON schema     what an MCP server actually publishes          ~300 tok
===================  ============================================  ==========

The prompt only ever receives the first form, for the handful of tools that
retrieval selected for this step. The second form lives on disk in the sandbox's
``PYTHONPATH``, where the agent reads it with ``help()`` if it needs detail. The
third form never enters a prompt at all — publishing 40 tools' JSON schemas is
~12k tokens, which is the token black hole the spec warns about.

Two execution modes per tool:

``local``
    Pure computation, runs inside the sandbox. Financial ratios, table
    reshaping, text search.
``bridged``
    Needs network or host credentials. Runs host-side via
    :mod:`teamclaw.execution.bridge`, so the sandbox stays on ``--network none``
    and every egress is an audited file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from teamclaw.context.retrieval import Doc, Retriever, build_retriever


@dataclass(frozen=True)
class ToolParam:
    name: str
    annotation: str = "str"
    default: str | None = None      # rendered source, e.g. "None" or "'10-K'"
    doc: str = ""

    def render(self) -> str:
        base = f"{self.name}: {self.annotation}"
        return f"{base} = {self.default}" if self.default is not None else base


@dataclass
class ToolSpec:
    module: str                      # "sec"
    func: str                        # "search_filings"
    summary: str                     # one line, used in the prompt
    params: Sequence[ToolParam] = ()
    returns: str = "dict"
    detail: str = ""                 # long docstring, sandbox-side only
    requires_network: bool = False
    handler: Callable[..., object] | None = None   # host-side impl for bridged tools
    local_source: str | None = None                # in-sandbox impl for local tools
    tags: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return f"{self.module}.{self.func}"

    @property
    def bridged(self) -> bool:
        return self.local_source is None

    def signature(self) -> str:
        return f"{self.func}({', '.join(p.render() for p in self.params)}) -> {self.returns}"

    def signature_line(self) -> str:
        """The only representation that reaches a prompt."""
        return f"- {self.module}.{self.signature()}  # {self.summary}"

    def docstring(self) -> str:
        lines = [self.summary, ""]
        if self.params:
            lines.append("Args:")
            lines += [
                f"    {p.name} ({p.annotation}): {p.doc or '-'}" for p in self.params
            ]
            lines.append("")
        if self.detail:
            lines += [self.detail.strip(), ""]
        if self.examples:
            lines.append("Examples:")
            lines += [f"    {e}" for e in self.examples]
        return "\n".join(lines).strip()

    def search_text(self) -> str:
        return " ".join(
            [self.name, self.func.replace("_", " "), self.summary, self.detail,
             " ".join(p.name.replace("_", " ") for p in self.params), " ".join(self.tags)]
        )


@dataclass
class ToolRegistry:
    tools: dict[str, ToolSpec] = field(default_factory=dict)
    # module -> verbatim source. A scenario whose local tools are more than a few
    # lines supplies a real module here instead of stuffing bodies into
    # ``ToolSpec.local_source`` strings: real files are testable, importable on
    # the host, and readable in review. The registry still needs a ToolSpec per
    # function, because that is what retrieval indexes and what the prompt shows.
    module_sources: dict[str, str] = field(default_factory=dict)
    _retriever: Retriever | None = field(default=None, repr=False)

    # -- registration ------------------------------------------------------
    def add(self, spec: ToolSpec) -> "ToolRegistry":
        self.tools[spec.name] = spec
        self._retriever = None
        return self

    def extend(self, specs: Iterable[ToolSpec]) -> "ToolRegistry":
        for s in specs:
            self.add(s)
        return self

    def attach_source(self, module: str, source: str) -> "ToolRegistry":
        """Use ``source`` verbatim as the sandbox module instead of generating it."""
        self.module_sources[module] = source
        return self

    def attach_source_file(self, module: str, path) -> "ToolRegistry":  # noqa: ANN001
        from pathlib import Path as _Path

        return self.attach_source(module, _Path(path).read_text(encoding="utf-8"))

    def get(self, name: str) -> ToolSpec | None:
        return self.tools.get(name)

    def modules(self) -> list[str]:
        return sorted({s.module for s in self.tools.values()})

    def by_module(self, module: str) -> list[ToolSpec]:
        return sorted(
            (s for s in self.tools.values() if s.module == module), key=lambda s: s.func
        )

    def names(self) -> list[str]:
        return sorted(self.tools)

    def __len__(self) -> int:
        return len(self.tools)

    # -- retrieval ---------------------------------------------------------
    def _ensure_index(self) -> Retriever:
        if self._retriever is None:
            self._retriever = build_retriever()
            self._retriever.index(
                [Doc(doc_id=s.name, text=s.search_text()) for s in self.tools.values()]
            )
        return self._retriever

    def select(
        self,
        step_description: str,
        *,
        k: int = 8,
        min_matched_idf: float = 0.8,
        always: Sequence[str] = (),
    ) -> list[ToolSpec]:
        """Pick the tools whose signatures earn prompt space this step.

        ``always`` pins tools that must be visible regardless of the step text —
        the workspace primitives, typically, since the agent needs them in every
        step and a step description rarely mentions "write a file".

        Below the retrieval threshold everything is dropped rather than padded to
        ``k``: a step that needs two tools should not be shown eight, because the
        six irrelevant signatures both cost budget and invite misuse.
        """
        picked: dict[str, ToolSpec] = {}
        for name in always:
            spec = self.tools.get(name)
            if spec is not None:
                picked[name] = spec

        for hit in self._ensure_index().search_detailed(step_description, k=k * 3):
            if len(picked) >= k:
                break
            if hit.doc_id in picked or hit.doc_id not in self.tools:
                continue
            if hit.matched_idf < min_matched_idf:
                continue
            picked[hit.doc_id] = self.tools[hit.doc_id]

        return [picked[n] for n in sorted(picked)]

    # -- prompt rendering --------------------------------------------------
    def signature_block(self, specs: Sequence[ToolSpec] | None = None) -> str:
        chosen = list(specs if specs is not None else self.tools.values())
        if not chosen:
            return "(no tools selected for this step)"
        by_module: dict[str, list[ToolSpec]] = {}
        for s in sorted(chosen, key=lambda s: s.name):
            by_module.setdefault(s.module, []).append(s)
        blocks: list[str] = []
        for module, specs_ in sorted(by_module.items()):
            blocks.append(
                f"from tools import {module}\n"
                + "\n".join(s.signature_line() for s in specs_)
            )
        return "\n\n".join(blocks)

    def token_cost_comparison(
        self, counter, *, step_description: str = "", k: int = 8
    ) -> dict[str, int]:
        """Measured cost of each representation. Used by the ablation report.

        ``retrieved_signatures`` is the number that actually matters: what one
        step pays after retrieval has filtered the registry down to the tools
        that step plausibly needs. Comparing *all* signatures against *all*
        schemas understates the saving, because the retrieval arm never injects
        all of either.
        """
        sig = counter(self.signature_block())
        stubs = counter("\n\n".join(self.stub_source(m) for m in self.modules()))
        schemas = counter(
            "\n".join(
                str(
                    {
                        "name": s.name,
                        "description": s.docstring(),
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                p.name: {"type": p.annotation, "description": p.doc}
                                for p in s.params
                            },
                        },
                    }
                )
                for s in self.tools.values()
            )
        )
        out = {
            "signature_lines": sig,
            "stub_sources": stubs,
            "full_json_schemas": schemas,
            "tools": len(self.tools),
        }
        if step_description:
            selected = self.select(step_description, k=k)
            out["retrieved_signatures"] = counter(self.signature_block(selected))
            out["retrieved_tools"] = len(selected)
        return out

    # -- sandbox-side materialisation --------------------------------------
    def stub_source(self, module: str) -> str:
        from teamclaw.execution.stubgen import render_module  # local import: cycle

        if module in self.module_sources:
            return self.module_sources[module]
        return render_module(module, self.by_module(module))

    def materialise(self, tools_dir) -> list[str]:
        from teamclaw.execution.stubgen import write_package  # local import: cycle

        return write_package(tools_dir, self)
