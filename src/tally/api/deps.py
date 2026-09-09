"""Application state, built once at startup and handed to routes.

Everything expensive or shared lives on one object rather than in module
globals: the database engine, the session store, the run manager, the provider
registry. That makes the app testable — a test constructs an :class:`AppState`
pointed at a temp directory and a fake provider, with no monkey-patching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fastapi import Request

from tally.config import Settings, settings as load_settings
from tally.context.ledger import Budget
from tally.execution.registry import ToolRegistry
from tally.execution.sandbox import SandboxLimits
from tally.execution.workspace import Workspace
from tally.orchestration.agent import AgentSpec
from tally.session import SessionStore
from tally.store import AgentDef, AgentStore, build_engine, init_schema


def build_spec_for(agent: AgentDef, workspace: Workspace,
                   cfg: Settings | None = None) -> AgentSpec:
    """Turn a stored declaration into a runnable spec.

    The tool registry comes from the scenario module, not from the stored tool
    names: the names are an *allow-list* applied on top, so an agent cannot be
    given a tool the scenario does not implement. A stored name that no longer
    exists is dropped rather than raising, because a code change should not brick
    an agent someone saved last month.
    """
    from tally.scenarios import SCENARIOS

    cfg = cfg or load_settings()
    module = SCENARIOS[agent.scenario].spec

    if agent.scenario == "dd_finance":
        from tally.scenarios.dd_finance.sec_client import SecClient

        registry = module.build_tools(SecClient(cfg=cfg), workspace)
    elif agent.scenario == "bi_analyst":
        registry = module.build_tools(cfg.paths.data / "bi_demo.db")
    elif agent.scenario == "deep_research":
        registry = module.build_tools(_offline_web_handlers(module))
    else:
        registry = module.build_tools()

    allowed = set(agent.tools or [])
    if allowed:
        filtered = ToolRegistry()
        filtered.module_sources = dict(registry.module_sources)
        kept = [spec for name, spec in registry.tools.items() if name in allowed]
        filtered.extend(kept)
        if kept:
            registry = filtered

    spec = module.build_spec(tools=registry, window=agent.window,
                             max_steps=agent.max_steps)
    spec.persona = agent.persona or spec.persona
    spec.budget = Budget(window=agent.window, max_output=agent.max_output)
    spec.tool_k = agent.tool_k
    spec.skill_k = agent.skill_k
    spec.memory_k = agent.memory_k
    spec.compaction_threshold = agent.compaction_threshold
    if agent.always_tools:
        spec.always_tools = tuple(
            t for t in agent.always_tools if t in registry.tools
        )
    spec.sandbox_limits = SandboxLimits(wall_clock_s=180.0, memory_mb=2048)
    return spec


def _offline_web_handlers(module: Any) -> Any:
    """Web handlers that refuse rather than silently fetching nothing.

    The research scenario needs a search backend this repository does not ship.
    Returning empty results would look like "the web had nothing to say", so the
    handlers raise with an explanation the agent can act on and a reader can
    understand.
    """
    def search(query: str, limit: int) -> list[dict[str, Any]]:
        raise LookupError(
            "no web search backend is configured; set one up before running the "
            "research scenario rather than treating an empty result as an answer"
        )

    def fetch(url: str) -> dict[str, Any]:
        raise LookupError("no web fetch backend is configured")

    return module.WebHandlers(search=search, fetch=fetch)


@dataclass
class AppState:
    cfg: Settings
    store: AgentStore
    session: SessionStore
    run_manager: Any = None
    registry_factory: Callable[[], Any] = field(default=lambda: None)
    schema_version: int = 0

    @classmethod
    def build(cls, *, cfg: Settings | None = None, db_path: Path | None = None,
              registry_factory: Callable[[], Any] | None = None,
              max_workers: int = 2) -> "AppState":
        from tally.api.runs import RunManager
        from tally.models.registry_default import build_registry

        cfg = cfg or load_settings()
        engine = build_engine(db_path=db_path or (cfg.paths.data / "tally.db"))
        version = init_schema(engine)
        store = AgentStore(engine)
        session = SessionStore.connect(cfg.redis_url)

        factory = registry_factory or (lambda: build_registry(cfg))
        state = cls(cfg=cfg, store=store, session=session,
                    registry_factory=factory, schema_version=version)
        state.run_manager = RunManager(
            store=store, session=session, registry_factory=factory,
            spec_factory=lambda agent, ws: build_spec_for(agent, ws, cfg),
            runs_root=cfg.paths.runs / "api",
            workspaces_root=cfg.paths.workspaces / "api",
            max_workers=max_workers,
        )
        return state

    def health(self) -> dict[str, Any]:
        from tally.execution.sandbox import DockerSandbox, build_sandbox
        from tally.models.registry_default import describe_registry

        registry = self.registry_factory()
        docker = DockerSandbox(workspace=self.cfg.paths.workspaces)
        sandbox = build_sandbox(workspace=self.cfg.paths.workspaces,
                                prefer_docker=True)
        providers = describe_registry(registry) if registry else []
        usable = [p for p in providers
                  if p["available"] and p["name"] not in {"fake", "scripted"}]
        return {
            "schema_version": self.schema_version,
            "session": self.session.status(),
            "sandbox": {"active": sandbox.backend,
                        "isolation": sandbox.isolation_level(),
                        "docker_available": docker.available(),
                        "image_present": docker.image_present(),
                        "unavailable_reason": docker.unavailable_reason()},
            "providers": providers,
            "usable_providers": [p["name"] for p in usable],
            "runs_live": self.run_manager.live() if self.run_manager else [],
            "warnings": self.session.warnings() + ([
                "No usable model provider. Runs will fail on NoProviderAvailable. "
                "Set a free-tier key in .env or start ollama."
            ] if not usable else []),
        }

    def shutdown(self) -> None:
        if self.run_manager is not None:
            self.run_manager.shutdown()


def app_state(request: Request) -> AppState:
    return request.app.state.tally
