"""Request and response models.

Validation here is not decoration. An agent definition sets a token budget, a
step ceiling and a tool allowance, and every one of those is a resource limit —
a window of 2,000,000 or 500 steps is not a preference, it is a way to exhaust
a quota or a disk. The bounds are stated as field constraints so a bad value is
rejected at the edge with a clear message rather than surfacing as a
``BudgetTooSmall`` twelve frames deep.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

# Floors and ceilings. The lower bound on `window` is the one that matters: the
# ledger's hard floors sum to just under 1,000 tokens, so a smaller window
# cannot satisfy them and the run dies on construction.
MIN_WINDOW, MAX_WINDOW = 4_000, 200_000
MAX_STEPS_CEILING = 60


class AgentCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120,
                      pattern=r"^[a-z0-9][a-z0-9_-]*$")
    scenario: str = Field(min_length=2, max_length=60)
    persona: str = Field(min_length=10, max_length=20_000)
    display_name: str = Field(default="", max_length=200)
    tools: list[str] = Field(default_factory=list, max_length=400)
    skills: list[str] = Field(default_factory=list, max_length=100)
    always_tools: list[str] = Field(default_factory=list, max_length=20)
    window: int = Field(default=32_000, ge=MIN_WINDOW, le=MAX_WINDOW)
    max_output: int = Field(default=2_048, ge=256, le=32_000)
    max_steps: int = Field(default=14, ge=1, le=MAX_STEPS_CEILING)
    tool_k: int = Field(default=8, ge=1, le=64)
    skill_k: int = Field(default=2, ge=0, le=10)
    memory_k: int = Field(default=5, ge=0, le=40)
    compaction_threshold: float = Field(default=0.70, ge=0.1, le=10.0)
    enabled: bool = True

    @field_validator("scenario")
    @classmethod
    def known_scenario(cls, value: str) -> str:
        from tally.scenarios import SCENARIOS

        if value not in SCENARIOS:
            raise ValueError(
                f"unknown scenario {value!r}; available: {sorted(SCENARIOS)}"
            )
        return value

    @field_validator("max_output")
    @classmethod
    def output_fits_window(cls, value: int, info) -> int:  # noqa: ANN001
        window = (info.data or {}).get("window", 32_000)
        # The ledger's budget is window - max_output - safety. Reserving most of
        # the window for output leaves nothing to allocate, and the failure shows
        # up as an unexplained BudgetTooSmall rather than a bad request.
        if value > window // 2:
            raise ValueError(
                f"max_output ({value}) may not exceed half the window ({window}); "
                "the remainder is what the context ledger has to allocate"
            )
        return value


class AgentUpdate(BaseModel):
    display_name: str | None = Field(default=None, max_length=200)
    persona: str | None = Field(default=None, min_length=10, max_length=20_000)
    tools: list[str] | None = Field(default=None, max_length=400)
    skills: list[str] | None = Field(default=None, max_length=100)
    always_tools: list[str] | None = Field(default=None, max_length=20)
    window: int | None = Field(default=None, ge=MIN_WINDOW, le=MAX_WINDOW)
    max_output: int | None = Field(default=None, ge=256, le=32_000)
    max_steps: int | None = Field(default=None, ge=1, le=MAX_STEPS_CEILING)
    tool_k: int | None = Field(default=None, ge=1, le=64)
    skill_k: int | None = Field(default=None, ge=0, le=10)
    memory_k: int | None = Field(default=None, ge=0, le=40)
    compaction_threshold: float | None = Field(default=None, ge=0.1, le=10.0)
    enabled: bool | None = None


class RelationCreate(BaseModel):
    target_id: str = Field(min_length=4, max_length=40)
    tenant: str = Field(default="default", min_length=1, max_length=80)
    may_delegate: bool = True
    may_share_files: bool = False


class RunCreate(BaseModel):
    objective: str = Field(min_length=4, max_length=20_000)
    conversation_id: str | None = None


class ChatMessage(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)


class HitlAnswer(BaseModel):
    decision: str = Field(pattern="^(approve|deny|guidance|abort)$")
    message: str = Field(default="", max_length=2_000)


class DelegationCheck(BaseModel):
    source_id: str
    target_id: str
    need_files: bool = False


class ApiError(BaseModel):
    detail: str
    hint: str = ""


def scenario_catalogue() -> list[dict[str, Any]]:
    """What the UI needs to render the agent editor: tools and skills per scenario.

    Built from the scenario modules rather than the database, because the tools a
    sandbox can import are a property of the code. A stored copy would let the
    editor offer tools that no longer exist.
    """
    from tally.scenarios import SCENARIOS

    out: list[dict[str, Any]] = []
    for name, module in SCENARIOS.items():
        spec = module.spec
        skills = []
        if hasattr(spec, "build_skills"):
            skills = [
                {"name": s.name, "description": s.description,
                 "when_to_use": s.when_to_use}
                for s in spec.build_skills().skills.values()
            ]
        out.append({
            "name": name,
            "display_name": name.replace("_", " ").title(),
            "persona": getattr(module, "PERSONA", ""),
            "always_tools": list(getattr(spec, "ALWAYS_TOOLS", ())),
            "skills": skills,
            "needs_network": name in {"dd_finance", "deep_research"},
            "task_levels": ["l1", "l2", "l3"] if name == "dd_finance" else [],
        })
    return out
