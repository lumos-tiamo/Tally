"""Skill packages: reusable instruction bundles loaded on demand.

A skill is a directory containing ``SKILL.md`` with YAML-ish frontmatter:

    ---
    name: xbrl-crosscheck
    description: Verify an extracted figure against XBRL company facts
    when_to_use: after extracting any monetary field from filing text
    ---
    <body: the actual instructions>

The point of the two-tier structure is budget: the *index* (name + description +
when_to_use, a few dozen tokens each) is cheap enough to always show, while the
*body* is only loaded for skills that matched the current step. Loading every
body would defeat the ledger — a dozen skills is easily 10k tokens.

Frontmatter is parsed with a tiny key/value reader rather than PyYAML: skills are
authored by us, the schema is four scalar fields, and adding a YAML dependency to
the sandbox image for that is not worth it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from teamclaw.context.retrieval import Doc, Retriever, build_retriever

FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass
class Skill:
    name: str
    description: str
    body: str
    when_to_use: str = ""
    path: Path | None = None
    tags: tuple[str, ...] = ()

    @property
    def index_entry(self) -> str:
        line = f"- **{self.name}**: {self.description}"
        if self.when_to_use:
            line += f" (use when: {self.when_to_use})"
        return line

    def rendered(self) -> str:
        head = f"### Skill: {self.name}\n{self.description}"
        return f"{head}\n\n{self.body.strip()}"

    def search_text(self) -> str:
        return " ".join([self.name, self.description, self.when_to_use, " ".join(self.tags)])


def parse_skill_md(text: str, *, path: Path | None = None) -> Skill:
    meta: dict[str, str] = {}
    body = text
    match = FRONTMATTER.match(text)
    if match:
        body = text[match.end():]
        for raw in match.group(1).splitlines():
            if ":" not in raw or raw.strip().startswith("#"):
                continue
            key, _, value = raw.partition(":")
            meta[key.strip().lower()] = value.strip().strip('"').strip("'")

    name = meta.get("name") or (path.parent.name if path else "unnamed")
    tags = tuple(t.strip() for t in meta.get("tags", "").split(",") if t.strip())
    return Skill(
        name=name,
        description=meta.get("description", ""),
        when_to_use=meta.get("when_to_use", ""),
        body=body,
        path=path,
        tags=tags,
    )


@dataclass
class SkillLibrary:
    """Holds skills, exposes an index, and selects bodies per step."""

    skills: dict[str, Skill] = field(default_factory=dict)
    retriever: Retriever | None = None

    def add(self, skill: Skill) -> "SkillLibrary":
        self.skills[skill.name] = skill
        self.retriever = None  # invalidate
        return self

    def extend(self, skills: Iterable[Skill]) -> "SkillLibrary":
        for s in skills:
            self.add(s)
        return self

    @staticmethod
    def from_dir(root: Path) -> "SkillLibrary":
        """Load every ``*/SKILL.md`` under ``root``. Missing dir yields empty library."""
        lib = SkillLibrary()
        if not root.exists():
            return lib
        for md in sorted(root.glob("**/SKILL.md")):
            try:
                lib.add(parse_skill_md(md.read_text(encoding="utf-8"), path=md))
            except OSError:
                continue
        return lib

    def index_text(self) -> str:
        if not self.skills:
            return ""
        lines = [s.index_entry for s in sorted(self.skills.values(), key=lambda s: s.name)]
        return "Available skills (bodies are loaded on demand):\n" + "\n".join(lines)

    def _ensure_index(self) -> Retriever:
        if self.retriever is None:
            self.retriever = build_retriever()
            self.retriever.index(
                [Doc(doc_id=s.name, text=s.search_text()) for s in self.skills.values()]
            )
        return self.retriever

    def select(
        self,
        step_description: str,
        *,
        k: int = 2,
        min_coverage: float = 0.30,
        min_matched_idf: float = 1.0,
    ) -> list[Skill]:
        """Pick the skills whose bodies earn their tokens for this step.

        Two gates, because either alone misbehaves (see :class:`Hit`):

        ``min_coverage`` on IDF-weighted coverage
            "You matched most of the informative content this corpus could offer."
            Weighting by IDF is what stops ``the``/``for``/``a`` from diluting a
            genuinely good match into rejection.
        ``min_matched_idf`` on absolute matched information
            "And what you matched was actually informative." A query made only of
            stopwords can reach coverage 1.0; this floor rejects it.

        A step matching nothing gets **no** skill body, which is the correct
        outcome — always returning k skills would inject irrelevant instructions
        and spend budget the tools slot needs.
        """
        if not self.skills:
            return []
        selected: list[Skill] = []
        for hit in self._ensure_index().search_detailed(step_description, k=k * 3):
            if hit.doc_id not in self.skills:
                continue
            if hit.coverage < min_coverage or hit.matched_idf < min_matched_idf:
                continue
            selected.append(self.skills[hit.doc_id])
            if len(selected) >= k:
                break
        return selected

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def names(self) -> list[str]:
        return sorted(self.skills)


DEFAULT_SKILLS: tuple[Skill, ...] = (
    Skill(
        name="cite-or-abstain",
        description="Report a figure only with a locatable citation, otherwise abstain",
        when_to_use="whenever a numeric value is about to be reported",
        body=(
            "For every number you report:\n"
            "1. Record the source document id and the page or section it came from.\n"
            "2. Quote the surrounding 10-20 words verbatim as `quote`.\n"
            "3. If you cannot locate the figure in the source, emit\n"
            "   `{\"value\": null, \"reason\": \"not_disclosed\"}` rather than an estimate.\n"
            "An abstention is a correct answer when the document is silent; a plausible\n"
            "invented number is always wrong, and is scored as a hallucination."
        ),
        tags=("citation", "hallucination", "numbers"),
    ),
    Skill(
        name="compute-dont-guess",
        description="Derive every ratio in code from named inputs, never mentally",
        when_to_use="whenever a ratio, growth rate or derived metric is required",
        body=(
            "Never state a derived figure you did not compute in the sandbox.\n"
            "1. Extract the raw inputs first and print them.\n"
            "2. Compute the derived value in Python and print the expression.\n"
            "3. Emit `numerator`, `denominator` and the source field for each,\n"
            "   so the result can be re-derived and checked independently.\n"
            "Arithmetic done in your head is unverifiable and frequently wrong at\n"
            "the magnitudes financial statements use."
        ),
        tags=("computation", "ratios", "sandbox"),
    ),
    Skill(
        name="persist-then-summarise",
        description="Write large intermediates to the workspace and keep only a digest in context",
        when_to_use="after producing any output longer than ~50 lines",
        body=(
            "Large intermediate results belong on disk, not in the conversation.\n"
            "1. Write the full artefact under `workspace/` (parquet, json or md).\n"
            "2. `print` only a digest: shape, column names, a few head rows, and the path.\n"
            "3. Reference the artefact by path in later steps and re-read what you need.\n"
            "This keeps the context budget available for reasoning, and makes the run\n"
            "resumable: the workspace is the checkpoint."
        ),
        tags=("workspace", "context", "budget"),
    ),
)


def default_library() -> SkillLibrary:
    return SkillLibrary().extend(DEFAULT_SKILLS)
