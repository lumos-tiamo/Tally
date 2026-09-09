"""Tally: multi-agent digital-employee platform.

Layer map (see docs/superpowers/specs/2026-09-09-agent-platform-design.md):

    scenarios/      declarative AgentSpec + tool modules   (no runtime code)
    orchestration/  agent loop, task graph, subagents, checkpoints, HITL
    context/        context ledger, compactor, memory, skill index
    execution/      sandbox, workspace, tool registry, stub generation
    models/         provider router, quota, cache, local models
    observability/  trace, token/cost accounting
    evaluation/     eval harness, metrics, judge calibration, ablations
"""

__version__ = "0.1.0"
