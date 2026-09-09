"""Sandboxed Python execution — the agent's only action.

Two backends behind one interface:

``DockerSandbox``
    The real thing: no network by default, read-only dataset mount, read-write
    workspace mount, cpu/memory/pid caps, dropped capabilities, non-root user.

    One environmental constraint is enforced rather than assumed. Docker Desktop
    shares only a configured set of host paths — ``/Users`` typically, ``/var/folders``
    typically not — and a bind mount from an unshared path does not fail. It
    silently becomes an *empty directory*, so the step script vanishes and the
    error surfaces as ``can't find '__main__' module``, which reads like a bug in
    the agent's code. :meth:`DockerSandbox.mount_works` probes the actual
    workspace path with a sentinel before the backend is chosen.
``LocalSandbox``
    A subprocess fallback for machines with no Docker daemon (this project was
    built on one). It is *honestly weaker*: it restricts the interpreter's
    environment, forbids a blocklist of imports by static check, applies
    RLIMIT_AS / RLIMIT_CPU and a wall-clock timeout, and runs in a temp cwd —
    but it shares the host kernel and filesystem namespace. It exists so the
    platform is runnable and testable everywhere, and
    :meth:`Sandbox.isolation_level` reports which one is active so an eval run
    can record it. Never point ``LocalSandbox`` at untrusted code.

Why dependencies are frozen
---------------------------
The agent may not install packages. Two reasons, both practical: an agent that
can ``pip install`` can pull an arbitrary supply chain into the run, and a run
whose dependency set changes between arms is not a valid ablation comparison.
"""

from __future__ import annotations

import ast
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

from teamclaw.observability.trace import SpanKind, Tracer

DEFAULT_IMAGE = "teamclaw-sandbox:latest"

# Mount-probe results, keyed by (image, resolved workspace path). Whether a host
# path is shared with the Docker VM cannot change during a run, and the probe
# costs a container start — so caching it at module level keeps a test suite of
# ten container tests from paying for ten probes.
_MOUNT_PROBE_CACHE: dict[tuple[str, str], bool] = {}

# Modules the local fallback refuses to let agent code import. This is a
# defence-in-depth measure for the weak backend, not a security boundary: a
# determined escape via getattr chains is possible. Docker is the boundary.
LOCAL_FORBIDDEN_IMPORTS = frozenset(
    {"socket", "http", "urllib", "urllib2", "ftplib", "telnetlib", "smtplib",
     "ctypes", "multiprocessing", "pty", "signal", "webbrowser", "pip",
     "setuptools", "importlib.util"}
)
LOCAL_FORBIDDEN_CALLS = frozenset({"eval", "exec", "compile", "__import__", "breakpoint"})


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_s: float
    timed_out: bool = False
    backend: str = "?"
    isolation: str = "?"
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    ERROR_PREVIEW_CHARS = 600

    def to_json(self) -> dict[str, object]:
        """Span attributes for a sandbox execution.

        A failure carries an ``error_preview`` — the tail of stderr, where the
        exception actually is. The first version recorded only ``stderr_chars``,
        and the first real model run produced five sandbox failures that were
        impossible to diagnose from the trace at all. Recording the length of an
        error instead of the error is not observability.

        Success does not carry stdout: that is the agent's own output, it can be
        large, and it is already the observation folded into the next prompt.
        """
        payload: dict[str, object] = {
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_s": round(self.duration_s, 3),
            "backend": self.backend,
            "isolation": self.isolation,
            "stdout_chars": len(self.stdout),
            "stderr_chars": len(self.stderr),
            "truncated": self.truncated,
        }
        if not self.ok and self.stderr:
            tail = self.stderr.strip()[-self.ERROR_PREVIEW_CHARS:]
            payload["error_preview"] = tail
            payload["error_kind"] = _first_exception_name(self.stderr)
        return payload


def newest_mtime(root: Path) -> float:
    """Most recent mtime anywhere under ``root``, or 0.0 if empty/unreadable."""
    newest = 0.0
    try:
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                try:
                    newest = max(newest, os.stat(os.path.join(dirpath, name)).st_mtime)
                except OSError:
                    continue
    except OSError:
        return newest
    return newest


@dataclass
class SandboxLimits:
    wall_clock_s: float = 120.0
    cpu_s: int = 60
    memory_mb: int = 2048
    pids: int = 128
    max_stdout_chars: int = 20_000
    network: bool = False
    # How long to wait for a container's writes to become visible on the host.
    # Docker Desktop's shared filesystem propagates guest writes with a short
    # delay, so reading the workspace the instant the container exits can miss
    # the artefacts the step just produced — and the agent then builds its next
    # context believing it wrote nothing. Bounded, and skipped entirely when
    # evidence of a write has already appeared.
    settle_ms: int = 1_500


class Sandbox(Protocol):
    backend: str

    def isolation_level(self) -> str: ...
    def available(self) -> bool: ...
    def run(self, code: str, *, limits: SandboxLimits | None = None) -> ExecResult: ...


# ``python -I`` implies ``-E``, which discards PYTHONPATH — so the tool package
# cannot be made importable through the environment. The bootstrap injects it into
# sys.path instead, and is deliberately kept to a SINGLE line: every line of
# preamble shifts the line numbers in any traceback the agent sees, and the
# convergence loop quotes those numbers back at it. One line is a constant offset
# that :data:`PREAMBLE_LINES` accounts for; a multi-line preamble would make the
# offset a maintenance hazard.
PREAMBLE_LINES = 1


def _bootstrap(tools_path: str | None, rpc_dir: str | None) -> str:
    parts = ["import sys as _s, os as _o"]
    if tools_path:
        parts.append(f"_s.path.insert(0, {tools_path!r})")
    if rpc_dir:
        parts.append(f"_o.environ.setdefault('TEAMCLAW_RPC_DIR', {rpc_dir!r})")
    return "; ".join(parts)


def wrap_code(code: str, *, tools_path: str | None, rpc_dir: str | None) -> str:
    return f"{_bootstrap(tools_path, rpc_dir)}\n{code}"


def shift_traceback_lines(text: str, offset: int = PREAMBLE_LINES) -> str:
    """Rewrite reported line numbers so they match the agent's own code.

    Covers both delivery forms: the local backend runs a ``step.py`` file, the
    container backend reads the program from stdin and reports ``<stdin>``.
    Without this the agent is told the error is on line 8 of code whose line 8 is
    something else, and the convergence loop's "fix line 8" feedback sends it to
    the wrong place.
    """
    if offset <= 0:
        return text

    def _fix(match: re.Match[str]) -> str:
        n = int(match.group(2))
        return f"{match.group(1)}{max(1, n - offset)}"

    return re.sub(r"((?:step\.py|<stdin>)\", line )(\d+)", _fix, text)


# Exception names in a traceback are frequently dotted —
# ``json.decoder.JSONDecodeError``, ``subprocess.CalledProcessError`` — so the
# pattern has to allow the module path and then keep only the class.
_EXC_NAME = re.compile(
    r"^((?:\w+\.)*\w*(?:Error|Exception|Exit|Interrupt|Warning))\s*:", re.MULTILINE
)


def _first_exception_name(stderr: str) -> str:
    """Best-effort exception class from a traceback, for grouping failures.

    Takes the last match: a chained traceback ends with the exception that
    actually escaped, and that is the one worth grouping on.
    """
    names = _EXC_NAME.findall(stderr or "")
    return names[-1].rsplit(".", 1)[-1] if names else ""


def _truncate(text: str, cap: int) -> tuple[str, bool]:
    """Keep both ends: the head shows what happened, the tail shows the failure."""
    if len(text) <= cap:
        return text, False
    head = text[: cap // 2]
    tail = text[-cap // 2 :]
    return f"{head}\n...[{len(text) - cap} chars elided]...\n{tail}", True


class DockerSandbox:
    backend = "docker"

    def __init__(
        self,
        *,
        workspace: Path,
        image: str = DEFAULT_IMAGE,
        datasets: Path | None = None,
        tools_dir: Path | None = None,
        limits: SandboxLimits | None = None,
        allow_hosts: Sequence[str] = (),
        tracer: Tracer | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.image = image
        self.datasets = Path(datasets) if datasets else None
        self.tools_dir = Path(tools_dir) if tools_dir else None
        self.limits = limits or SandboxLimits()
        self.allow_hosts = tuple(allow_hosts)
        self.tracer = tracer
        self._mount_ok: bool | None = None

    def isolation_level(self) -> str:
        return "container" if not self.limits.network else "container+egress"

    def available(self) -> bool:
        if shutil.which("docker") is None:
            return False
        try:
            proc = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True, text=True, timeout=8,
            )
        except (subprocess.SubprocessError, OSError):
            return False
        return proc.returncode == 0 and bool(proc.stdout.strip())

    def image_present(self) -> bool:
        try:
            proc = subprocess.run(
                ["docker", "image", "inspect", self.image],
                capture_output=True, text=True, timeout=15,
            )
        except (subprocess.SubprocessError, OSError):
            return False
        return proc.returncode == 0

    def _argv(self) -> list[str]:
        argv = [
            "docker", "run", "--rm",
            "--user", "1000:1000",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--read-only",                       # rootfs read-only
            "--tmpfs", "/tmp:rw,size=256m,noexec",
            "--memory", f"{self.limits.memory_mb}m",
            "--memory-swap", f"{self.limits.memory_mb}m",
            "--cpus", "1.5",
            "--pids-limit", str(self.limits.pids),
            "--network", "none" if not self.limits.network else "bridge",
            "-v", f"{self.workspace.resolve()}:/workspace:rw",
            "-w", "/workspace",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "PYTHONUNBUFFERED=1",
            "-e", "HOME=/tmp",
            "-i",  # keep stdin open: the program is delivered through it
        ]
        if self.datasets is not None:
            argv += ["-v", f"{self.datasets.resolve()}:/datasets:ro"]
        if self.tools_dir is not None:
            argv += ["-v", f"{self.tools_dir.resolve()}:/opt/tools:ro",
                     "-e", "PYTHONPATH=/opt/tools"]
        # ``-`` reads the program from stdin. Two earlier approaches both failed
        # on Docker Desktop's shared filesystem: mounting the script from a temp
        # directory hit the unshared-path problem, and writing it into the
        # workspace hit a propagation race — the container starts before the
        # host's write is visible in the guest, so roughly half the runs died
        # with "can't open file". stdin is delivered over the daemon socket and
        # depends on no filesystem at all.
        argv += [self.image, "python", "-I", "-"]
        return argv

    MOUNT_PROBE_ATTEMPTS = 3

    def mount_works(self) -> bool:
        """Probe that a bind mount of this workspace is actually visible inside.

        Retried, and the retries are the substance rather than defensiveness.
        Docker Desktop's shared filesystem propagates host writes to the guest
        with a short delay, so a single-shot probe fails intermittently on a path
        that is perfectly shareable — which is exactly the flakiness this method
        exists to eliminate. A path that is genuinely unshared never appears, no
        matter how long you wait, so a couple of retries separate "slow" from
        "never" cleanly.

        The result is cached per (image, path) at module level: the answer cannot
        change during a run, and each attempt costs a container start.
        """
        key = (self.image, str(self.workspace.resolve()))
        if key in _MOUNT_PROBE_CACHE:
            return _MOUNT_PROBE_CACHE[key]

        self.workspace.mkdir(parents=True, exist_ok=True)
        sentinel = self.workspace / ".teamclaw-mount-probe"
        sentinel.write_text("probe", encoding="utf-8")
        ok = False
        try:
            for attempt in range(self.MOUNT_PROBE_ATTEMPTS):
                if attempt:
                    time.sleep(0.4 * attempt)
                try:
                    proc = subprocess.run(
                        ["docker", "run", "--rm", "--network", "none",
                         "-v", f"{self.workspace.resolve()}:/probe:ro",
                         self.image, "cat", "/probe/.teamclaw-mount-probe"],
                        capture_output=True, text=True, timeout=45,
                    )
                except (subprocess.SubprocessError, OSError):
                    break
                if proc.returncode == 0 and "probe" in proc.stdout:
                    ok = True
                    break
        finally:
            sentinel.unlink(missing_ok=True)

        _MOUNT_PROBE_CACHE[key] = ok
        self._mount_ok = ok
        return ok

    def unavailable_reason(self) -> str:
        """Why the container backend cannot be used, for the operator."""
        if shutil.which("docker") is None:
            return "the docker CLI is not on PATH"
        if not self.available():
            return "the docker daemon is not reachable"
        if not self.image_present():
            return (f"image {self.image} is absent — build it with "
                    "`docker build -t teamclaw-sandbox:latest -f docker/Dockerfile.sandbox docker/`")
        if not self.mount_works():
            return (f"docker cannot bind-mount {self.workspace} — Docker Desktop "
                    "shares only configured host paths, and an unshared path mounts "
                    "as an empty directory. Move the workspace under a shared path "
                    "(your home directory) or add this one in Docker Desktop's "
                    "File Sharing settings.")
        return ""

    def run(self, code: str, *, limits: SandboxLimits | None = None) -> ExecResult:
        lim = limits or self.limits
        started = time.time()

        self.workspace.mkdir(parents=True, exist_ok=True)
        program = wrap_code(
            code,
            tools_path="/opt/tools" if self.tools_dir else None,
            rpc_dir="/workspace/.rpc",
        )

        try:
            proc = subprocess.run(
                self._argv(), input=program,
                capture_output=True, text=True, timeout=lim.wall_clock_s,
            )
            out, t1 = _truncate(proc.stdout, lim.max_stdout_chars)
            err, t2 = _truncate(shift_traceback_lines(proc.stderr), lim.max_stdout_chars)
            result = ExecResult(
                stdout=out, stderr=err, exit_code=proc.returncode,
                duration_s=time.time() - started, backend=self.backend,
                isolation=self.isolation_level(), truncated=t1 or t2,
            )
        except subprocess.TimeoutExpired:
            result = ExecResult(
                stdout="", stderr=f"timed out after {lim.wall_clock_s}s",
                exit_code=124, duration_s=time.time() - started,
                timed_out=True, backend=self.backend,
                isolation=self.isolation_level(),
            )
        self._trace(result)
        return result

    def _trace(self, result: ExecResult) -> None:
        if self.tracer is None:
            return
        with self.tracer.span(SpanKind.SANDBOX_EXEC, self.backend) as sp:
            sp.set(**result.to_json())


class LocalSandbox:
    """Subprocess fallback. Weaker isolation, honestly labelled."""

    backend = "local-subprocess"

    def __init__(
        self,
        *,
        workspace: Path,
        tools_dir: Path | None = None,
        datasets: Path | None = None,
        limits: SandboxLimits | None = None,
        tracer: Tracer | None = None,
        static_check: bool = True,
    ) -> None:
        self.workspace = Path(workspace)
        self.tools_dir = Path(tools_dir) if tools_dir else None
        self.datasets = Path(datasets) if datasets else None
        self.limits = limits or SandboxLimits()
        self.tracer = tracer
        self.static_check = static_check

    def isolation_level(self) -> str:
        return "process-rlimits-only"

    def available(self) -> bool:
        return True

    # -- static pre-checks -------------------------------------------------
    def _check(self, code: str) -> str | None:
        """Reject obviously dangerous code before running it.

        Returns an error string, or None when the code passes. This is a
        speed-bump for a fallback backend, not a sandbox; the docstring at module
        level says so plainly.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"SyntaxError: {exc}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in LOCAL_FORBIDDEN_IMPORTS or alias.name in LOCAL_FORBIDDEN_IMPORTS:
                        return f"forbidden import in local sandbox: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in LOCAL_FORBIDDEN_IMPORTS:
                    return f"forbidden import in local sandbox: {node.module}"
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in LOCAL_FORBIDDEN_CALLS:
                    return f"forbidden call in local sandbox: {node.func.id}()"
        return None

    @staticmethod
    def _preexec(cpu_s: int, memory_mb: int):
        def _apply() -> None:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            nbytes = memory_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
            except (ValueError, OSError):
                pass  # macOS often refuses RLIMIT_AS; CPU + wall clock still apply
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            os.setsid()
        return _apply

    def run(self, code: str, *, limits: SandboxLimits | None = None) -> ExecResult:
        lim = limits or self.limits
        started = time.time()

        if self.static_check and (problem := self._check(code)) is not None:
            result = ExecResult(
                stdout="", stderr=problem, exit_code=126,
                duration_s=time.time() - started, backend=self.backend,
                isolation=self.isolation_level(),
            )
            self._trace(result)
            return result

        self.workspace.mkdir(parents=True, exist_ok=True)
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(self.workspace),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "TEAMCLAW_SANDBOX": "local",
            "LC_ALL": "C.UTF-8",
        }
        # No PYTHONPATH: `-I` discards it. The bootstrap line handles sys.path.
        if self.datasets is not None:
            env["TEAMCLAW_DATASETS"] = str(self.datasets.resolve())

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "step.py"
            script.write_text(
                wrap_code(
                    code,
                    tools_path=str(self.tools_dir.resolve()) if self.tools_dir else None,
                    rpc_dir=str((self.workspace / ".rpc").resolve()),
                ),
                encoding="utf-8",
            )
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(script)],
                    capture_output=True, text=True, timeout=lim.wall_clock_s,
                    cwd=str(self.workspace), env=env,
                    preexec_fn=self._preexec(lim.cpu_s, lim.memory_mb),
                )
                out, t1 = _truncate(proc.stdout, lim.max_stdout_chars)
                err, t2 = _truncate(shift_traceback_lines(proc.stderr), lim.max_stdout_chars)
                result = ExecResult(
                    stdout=out, stderr=err, exit_code=proc.returncode,
                    duration_s=time.time() - started, backend=self.backend,
                    isolation=self.isolation_level(), truncated=t1 or t2,
                )
            except subprocess.TimeoutExpired:
                result = ExecResult(
                    stdout="", stderr=f"timed out after {lim.wall_clock_s}s",
                    exit_code=124, duration_s=time.time() - started,
                    timed_out=True, backend=self.backend,
                    isolation=self.isolation_level(),
                )
        self._trace(result)
        return result

    def _trace(self, result: ExecResult) -> None:
        if self.tracer is None:
            return
        with self.tracer.span(SpanKind.SANDBOX_EXEC, self.backend) as sp:
            sp.set(**result.to_json())


def build_sandbox(
    *,
    workspace: Path,
    tools_dir: Path | None = None,
    datasets: Path | None = None,
    limits: SandboxLimits | None = None,
    tracer: Tracer | None = None,
    prefer_docker: bool = True,
) -> Sandbox:
    """Docker when the daemon and image are both present, local otherwise.

    The choice is recorded in every ``ExecResult`` so a result set always carries
    the isolation level it was produced under.
    """
    if prefer_docker:
        docker = DockerSandbox(
            workspace=workspace, datasets=datasets, tools_dir=tools_dir,
            limits=limits or SandboxLimits(), tracer=tracer,
        )
        if docker.available() and docker.image_present() and docker.mount_works():
            return docker
    return LocalSandbox(
        workspace=workspace, tools_dir=tools_dir, datasets=datasets,
        limits=limits or SandboxLimits(), tracer=tracer,
    )


SANDBOX_DOCKERFILE = textwrap.dedent(
    """\
    # Frozen dependency set: the agent may not install packages.
    FROM python:3.13-slim

    RUN groupadd -g 1000 sandbox && useradd -u 1000 -g 1000 -m sandbox

    # Pinned so an ablation arm run next week compares against the same environment.
    RUN pip install --no-cache-dir \\
            pandas==2.2.3 \\
            pyarrow==18.1.0 \\
            numpy==2.2.1 \\
            lxml==5.3.0 \\
            beautifulsoup4==4.12.3 \\
        && rm -rf /root/.cache

    RUN mkdir -p /workspace /datasets /opt/tools && chown -R 1000:1000 /workspace
    USER 1000:1000
    WORKDIR /workspace
    ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    """
)
