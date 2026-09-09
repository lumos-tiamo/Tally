"""Deep research scenario: parallel sub-agents with enforced provenance."""

from teamclaw.scenarios.deep_research import spec
from teamclaw.scenarios.deep_research.spec import (
    PERSONA,
    WebHandlers,
    build_skills,
    build_spec,
    build_tools,
    objective,
)

__all__ = ["spec", "PERSONA", "WebHandlers", "build_skills", "build_spec",
           "build_tools", "objective"]
