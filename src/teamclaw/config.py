"""Runtime configuration, loaded from environment with a tiny .env reader.

We avoid python-dotenv so the sandbox image stays minimal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> None:
    """Populate os.environ from a .env file without overriding real env vars."""
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _flag(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Paths:
    root: Path = REPO_ROOT
    data: Path = field(default_factory=lambda: REPO_ROOT / "data")
    cache: Path = field(default_factory=lambda: REPO_ROOT / "data" / "cache")
    filings: Path = field(default_factory=lambda: REPO_ROOT / "data" / "filings")
    datasets: Path = field(default_factory=lambda: REPO_ROOT / "data" / "datasets")
    runs: Path = field(default_factory=lambda: REPO_ROOT / "runs")
    workspaces: Path = field(default_factory=lambda: REPO_ROOT / "workspace")

    def ensure(self) -> "Paths":
        for p in (self.data, self.cache, self.filings, self.datasets, self.runs, self.workspaces):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)

    sec_user_agent: str = ""
    ollama_base_url: str = "http://localhost:11434"
    allow_paid: bool = False

    # -- serving layer -----------------------------------------------------
    redis_url: str = ""
    # When set, every mutating API call and every channel webhook must present
    # it. Left empty the server binds to localhost only and says so at startup —
    # an unauthenticated agent platform that can execute code should not be
    # reachable from a network by accident.
    api_token: str = ""
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_workers: int = 1

    # -- channels ----------------------------------------------------------
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_verification_token: str = ""
    feishu_encrypt_key: str = ""
    slack_signing_secret: str = ""
    slack_bot_token: str = ""

    # provider credentials; empty string means "not configured"
    gemini_key: str = ""
    glm_key: str = ""
    groq_key: str = ""
    cerebras_key: str = ""
    siliconflow_key: str = ""
    strong_key: str = ""
    strong_base_url: str = ""
    strong_model: str = ""

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        return cls(
            paths=Paths().ensure(),
            sec_user_agent=_env("TEAMCLAW_SEC_USER_AGENT"),
            ollama_base_url=_env("TEAMCLAW_OLLAMA_BASE_URL", "http://localhost:11434"),
            allow_paid=_flag("TEAMCLAW_ALLOW_PAID", False),
            redis_url=_env("TEAMCLAW_REDIS_URL"),
            api_token=_env("TEAMCLAW_API_TOKEN"),
            api_host=_env("TEAMCLAW_API_HOST", "127.0.0.1"),
            api_port=int(_env("TEAMCLAW_API_PORT", "8000") or 8000),
            api_workers=int(_env("TEAMCLAW_API_WORKERS", "1") or 1),
            feishu_app_id=_env("TEAMCLAW_FEISHU_APP_ID"),
            feishu_app_secret=_env("TEAMCLAW_FEISHU_APP_SECRET"),
            feishu_verification_token=_env("TEAMCLAW_FEISHU_VERIFICATION_TOKEN"),
            feishu_encrypt_key=_env("TEAMCLAW_FEISHU_ENCRYPT_KEY"),
            slack_signing_secret=_env("TEAMCLAW_SLACK_SIGNING_SECRET"),
            slack_bot_token=_env("TEAMCLAW_SLACK_BOT_TOKEN"),
            gemini_key=_env("TEAMCLAW_GEMINI_API_KEY"),
            glm_key=_env("TEAMCLAW_GLM_API_KEY"),
            groq_key=_env("TEAMCLAW_GROQ_API_KEY"),
            cerebras_key=_env("TEAMCLAW_CEREBRAS_API_KEY"),
            siliconflow_key=_env("TEAMCLAW_SILICONFLOW_API_KEY"),
            strong_key=_env("TEAMCLAW_STRONG_API_KEY"),
            strong_base_url=_env("TEAMCLAW_STRONG_BASE_URL"),
            strong_model=_env("TEAMCLAW_STRONG_MODEL"),
        )

    @property
    def exposed_beyond_localhost(self) -> bool:
        return self.api_host not in {"127.0.0.1", "localhost", "::1"}

    def api_security_warnings(self) -> list[str]:
        """Refuse to be quietly insecure.

        This server can start containers and run generated code. Binding it to a
        routable address without a token is the one configuration that turns that
        into a remote code execution service, so it is called out loudly rather
        than left in a README.
        """
        warnings: list[str] = []
        if self.exposed_beyond_localhost and not self.api_token:
            warnings.append(
                f"API is bound to {self.api_host} with no TEAMCLAW_API_TOKEN set. "
                "This server executes generated code in a sandbox; reachable and "
                "unauthenticated is remote code execution. Set a token or bind to "
                "127.0.0.1."
            )
        if self.api_workers > 1 and not self.redis_url:
            warnings.append(
                f"api_workers={self.api_workers} with no TEAMCLAW_REDIS_URL. "
                "Session state would fall back to per-process memory and silently "
                "not be shared between workers."
            )
        return warnings

    def require_sec_user_agent(self) -> str:
        """SEC blocks unidentified traffic; fail loudly rather than get 403s."""
        if not self.sec_user_agent:
            raise RuntimeError(
                "TEAMCLAW_SEC_USER_AGENT is required. SEC's fair-access policy demands a "
                "self-identifying User-Agent such as 'Name your.email@example.com'. "
                "See https://www.sec.gov/os/webmaster-faq"
            )
        return self.sec_user_agent


_SETTINGS: Settings | None = None


def settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings.load()
    return _SETTINGS


def reset_settings() -> None:
    """Test hook: force a re-read of the environment."""
    global _SETTINGS
    _SETTINGS = None
