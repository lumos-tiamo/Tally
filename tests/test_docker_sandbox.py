"""Container-backend isolation properties.

Skipped unless the daemon, the image, and a *working* bind mount are all present
— which is why ``mount_works`` exists as a separate probe. Docker Desktop shares
only configured host paths and mounts an unshared one as an empty directory, so
"docker is available" is not sufficient to conclude the backend will work.

The workspace deliberately lives under the repository rather than in ``tmp_path``:
pytest's temp directories are under ``/var/folders`` on macOS, which Docker
Desktop does not share, so a fixture-based workspace would skip these tests on
the machine they most need to run on.
"""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

import pytest

from teamclaw.execution.bridge import ToolBridge
from teamclaw.execution.registry import ToolParam, ToolRegistry, ToolSpec
from teamclaw.execution.sandbox import DockerSandbox, SandboxLimits, build_sandbox
from teamclaw.execution.stubgen import write_package

PROBE_ROOT = Path(__file__).resolve().parents[1] / "workspace" / "_pytest_docker"


def docker_usable() -> tuple[bool, str]:
    sandbox = DockerSandbox(workspace=PROBE_ROOT / "probe")
    reason = sandbox.unavailable_reason()
    return (not reason), reason


USABLE, REASON = docker_usable()
pytestmark = pytest.mark.skipif(not USABLE, reason=f"container backend unusable: {REASON}")


@pytest.fixture
def container_workspace(request):
    """A distinct path per test, never a cleared-and-reused one.

    Reusing one path made these tests flaky in a way that took a while to pin
    down: the guest caches directory entries, so a file the host deleted between
    tests could not be recreated by the container at the same path. Unique paths
    remove the interaction entirely.
    """
    root = PROBE_ROOT / f"ws_{request.node.name[:40]}_{uuid.uuid4().hex[:6]}"
    root.mkdir(parents=True, exist_ok=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def container(container_workspace: Path):
    sandbox = build_sandbox(workspace=container_workspace, prefer_docker=True)
    assert sandbox.backend == "docker", "the container backend should have been chosen"
    return sandbox


def test_the_frozen_dependency_set_is_present(container):
    result = container.run("import pandas, numpy, pyarrow, bs4, lxml; print('ok')")
    assert result.ok, result.stderr
    assert "ok" in result.stdout


def test_it_runs_as_a_non_root_user(container):
    result = container.run("import os; print(os.getuid())")
    assert result.stdout.strip() == "1000"


def test_the_network_is_unreachable_at_the_kernel_level(container):
    """Not the local backend's static import check — an actual refused connection."""
    result = container.run(
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 80), timeout=3)\n"
        "    print('REACHABLE')\n"
        "except OSError as exc:\n"
        "    print('blocked', type(exc).__name__)\n"
    )
    assert "blocked" in result.stdout
    assert "REACHABLE" not in result.stdout


def test_the_root_filesystem_is_read_only(container):
    result = container.run(
        "try:\n"
        "    open('/etc/probe', 'w').write('x')\n"
        "    print('WROTE')\n"
        "except OSError:\n"
        "    print('read-only')\n"
    )
    assert "read-only" in result.stdout


def test_the_workspace_is_writable_and_lands_on_the_host(container, container_workspace):
    """Guest writes reach the host — eventually, and the platform waits for it.

    Measured propagation on Docker Desktop runs to about 1.1 s, so the assertion
    allows for it rather than pretending the shared filesystem is synchronous.
    The sandbox itself waits (boundedly) for the same reason: otherwise the agent
    can read a workspace digest that omits the artefact it just wrote.
    """
    result = container.run("open('artifact.txt', 'w').write('from container'); print('wrote')")
    assert result.ok, result.stderr
    target = container_workspace / "artifact.txt"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not target.exists():
        time.sleep(0.05)
    assert target.exists(), "the container's write never reached the host"
    assert target.read_text() == "from container"


def test_packages_cannot_be_installed(container):
    """A run whose dependency set can change is not a valid ablation comparison."""
    result = container.run(
        "import subprocess, sys\n"
        "p = subprocess.run([sys.executable, '-m', 'pip', 'install', 'requests'],"
        " capture_output=True, text=True)\n"
        "print('rc', p.returncode)\n"
    )
    assert "rc 0" not in result.stdout


def test_the_memory_cap_is_enforced(container):
    result = container.run(
        "x = bytearray(3_000_000_000); print('ALLOCATED')",
        limits=SandboxLimits(memory_mb=256, wall_clock_s=60),
    )
    assert not result.ok
    assert "ALLOCATED" not in result.stdout


def test_the_wall_clock_is_enforced(container):
    result = container.run("while True: pass", limits=SandboxLimits(wall_clock_s=5))
    assert result.timed_out and result.exit_code == 124


def test_traceback_lines_match_the_agents_own_code(container):
    """The bootstrap is one line and stdin delivery must not shift the count."""
    result = container.run("a = 1\nb = 2\nc = b / 0\n")
    assert 'line 3' in result.stderr


def test_the_host_bridge_works_from_a_network_free_container(container, container_workspace):
    seen: list[str] = []

    def fetch(cik: str) -> dict:
        seen.append(cik)
        return {"cik": cik, "revenue": 391_035_000_000}

    registry = ToolRegistry().add(ToolSpec(
        module="sec", func="fetch", summary="fetch", params=[ToolParam("cik", "str")],
        returns="dict", requires_network=True, handler=fetch,
    ))
    tools_dir = PROBE_ROOT / f"tools_{uuid.uuid4().hex[:6]}"
    tools_dir.mkdir(parents=True, exist_ok=True)
    write_package(tools_dir, registry)

    sandbox = build_sandbox(workspace=container_workspace, tools_dir=tools_dir,
                            prefer_docker=True)
    bridge = ToolBridge(rpc_dir=container_workspace / ".rpc").register_registry(registry)
    with bridge.serving():
        result = sandbox.run("from tools import sec\nprint(sec.fetch(cik='0000320193'))")

    assert result.ok, result.stderr
    assert seen == ["0000320193"]
    shutil.rmtree(tools_dir, ignore_errors=True)


def test_an_unshared_workspace_path_is_detected_rather_than_silently_broken(tmp_path: Path):
    """The failure that reads as a bug in the agent's code.

    ``tmp_path`` is under /var/folders on macOS, which Docker Desktop does not
    share. The probe must catch that and produce an actionable reason instead of
    letting the mount become an empty directory.
    """
    sandbox = DockerSandbox(workspace=tmp_path / "unshared")
    if sandbox.mount_works():
        pytest.skip("this platform shares the temp directory; nothing to detect")
    reason = sandbox.unavailable_reason()
    assert "bind-mount" in reason
    # And the factory must degrade rather than hand back a broken container.
    assert build_sandbox(workspace=tmp_path / "unshared",
                         prefer_docker=True).backend == "local-subprocess"
