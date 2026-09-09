"""The FastAPI application.

Two decisions worth stating.

**Auth is enforced on mutation, not on reading.** A GET of the dashboard's data
is harmless; a POST starts a container and runs generated code. So the token
gate covers every non-GET plus the WebSocket, and reads are open — which keeps
the dashboard usable on a laptop without a token while the dangerous verbs stay
shut. Channel webhooks are exempt because they carry their own, stronger proof:
a platform signature over the body.

**Startup refuses to be quietly dangerous.** Binding beyond localhost with no
token is the one configuration that makes this remote code execution, so it is
printed as a warning at startup and surfaced in ``/api/insight/health`` rather
than left in a README.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tally import __version__
from tally.api.deps import AppState
from tally.api.routes import agents, channels, insight, runs
from tally.config import Settings, settings as load_settings

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
REPO_WEB_DIR = Path(__file__).resolve().parents[3] / "web"

# Reads are open; anything that can start work is not. Channel webhooks
# authenticate with a platform signature instead of the token.
OPEN_PREFIXES = ("/api/channels/",)


def _web_root() -> Path | None:
    for candidate in (WEB_DIR, REPO_WEB_DIR):
        if (candidate / "index.html").exists():
            return candidate
    return None


def create_app(
    *,
    cfg: Settings | None = None,
    db_path: Path | None = None,
    registry_factory: Any = None,
    max_workers: int = 2,
    seed: bool = False,
) -> FastAPI:
    """Build the app.

    ``seed`` is a constructor argument rather than something a caller bolts on
    with ``@app.on_event("startup")``. That decorator does not fire when a
    ``lifespan`` is supplied, which is how the CLI's --seed silently did nothing
    and a fresh install came up with no agents and no explanation.
    """
    cfg = cfg or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = AppState.build(cfg=cfg, db_path=db_path,
                               registry_factory=registry_factory,
                               max_workers=max_workers)
        # Worker threads reach the loop through this to deliver run events.
        state.run_manager.bind_loop(asyncio.get_running_loop())
        app.state.tally = state

        if seed:
            from tally.store import seed_default_agents

            created = seed_default_agents(state.store)
            if created:
                print(f"[tally] seeded {len(created)} agent(s): "
                      f"{', '.join(a.name for a in created)}", flush=True)

        for warning in cfg.api_security_warnings() + state.session.warnings():
            print(f"[tally] WARNING {warning}", flush=True)
        try:
            yield
        finally:
            state.shutdown()

    app = FastAPI(
        title="Tally",
        version=__version__,
        summary="Multi-agent platform: code-execution substrate, context ledger, eval harness",
        lifespan=lifespan,
    )

    # Same-origin by default. The dashboard is served by this app, so no
    # cross-origin access is needed and allowing it would widen the surface of a
    # server that executes code.
    app.add_middleware(
        CORSMiddleware, allow_origins=[], allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"],
    )

    @app.middleware("http")
    async def no_store_api(request: Request, call_next):  # noqa: ANN001, ANN202
        """API responses are live state and must never be cached.

        Without this the browser happily serves a stale /api/agents from its own
        cache, and the console shows agents that no longer exist while the POST
        that should have created a run never leaves the page. It presents as "the
        UI is broken" and is entirely a missing header.
        """
        response = await call_next(request)
        if request.url.path.startswith("/api") or request.url.path == "/healthz":
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response

    @app.middleware("http")
    async def require_token(request: Request, call_next):  # noqa: ANN001, ANN202
        token = cfg.api_token
        path = request.url.path
        if (token
                and request.method not in {"GET", "HEAD", "OPTIONS"}
                and not path.startswith(OPEN_PREFIXES)):
            presented = (request.headers.get("x-tally-token")
                         or request.headers.get("authorization", "").removeprefix("Bearer ").strip())
            if presented != token:
                return JSONResponse(
                    {"detail": "missing or invalid X-Tally-Token",
                     "hint": "this endpoint can start a run that executes code"},
                    status_code=status.HTTP_401_UNAUTHORIZED,
                )
        return await call_next(request)

    app.include_router(agents.router)
    app.include_router(runs.router)
    app.include_router(insight.router)
    app.include_router(channels.router)

    @app.get("/api/version")
    def version() -> dict[str, str]:
        return {"version": __version__}

    @app.get("/healthz")
    def healthz(request: Request) -> dict[str, Any]:
        state: AppState = request.app.state.tally
        return {"ok": True, "schema_version": state.schema_version,
                "session_backend": state.session.backend.name}

    web_root = _web_root()
    if web_root is not None:
        app.mount("/assets", StaticFiles(directory=str(web_root)), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(str(web_root / "index.html"))
    else:
        @app.get("/")
        def no_ui() -> dict[str, str]:
            return {"detail": "the dashboard is not built into this install",
                    "api": "/docs"}

    return app


def seed_if_empty(app: FastAPI) -> list[str]:
    """Create one agent per scenario on a fresh database. Idempotent."""
    from tally.store import seed_default_agents

    state: AppState = app.state.tally
    return [a.name for a in seed_default_agents(state.store)]


app = None  # populated by the CLI; uvicorn should use the factory
