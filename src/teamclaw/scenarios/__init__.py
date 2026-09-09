"""Scenario packages.

Every scenario is the same four declarations — ``PERSONA``, ``build_tools``,
``build_spec``, ``objective`` — plus its sandbox tool modules. No scenario
contains runtime code; ``tests/test_platform_abstraction.py`` asserts it.
"""

from teamclaw.scenarios import bi_analyst, code_engineer, dd_finance, deep_research

SCENARIOS = {
    "dd_finance": dd_finance,
    "bi_analyst": bi_analyst,
    "deep_research": deep_research,
    "code_engineer": code_engineer,
}

__all__ = ["dd_finance", "bi_analyst", "deep_research", "code_engineer", "SCENARIOS"]
