"""Financial diligence scenario: the deep arm, with the full eval apparatus."""

from tally.scenarios.dd_finance import spec
from tally.scenarios.dd_finance.spec import (
    AGENT_PERSONA as PERSONA,
    build_skills,
    build_spec,
    build_tools,
    l1_objective,
    l2_objective,
)

__all__ = ["spec", "PERSONA", "build_skills", "build_spec", "build_tools",
           "l1_objective", "l2_objective"]
