"""
docker_exec.py
==============
Runs arbitrary code in a persistent, network-isolated, resource-capped
Docker container using the gVisor (runsc) runtime for kernel isolation.

Agents call run_code() or run_tests(). Never runs as root.

"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import threading
import uuid
import textwrap
import time
import random
from pathlib import Path
from typing import Any, Literal
from functools import wraps

import docker
from crewai.tools import tool
from docker.errors import ContainerError, ImageNotFound, APIError, NotFound
from tools.logging_utils import rprint


# Config (from environment)

SANDBOX_IMAGE   = "agent-exec:latest"
QUALITY_IMAGE   = "agent-quality:latest"
RUNTIME         = os.getenv("SANDBOX_RUNTIME", "runsc")   # runsc = gVisor
MEMORY_LIMIT    = os.getenv("SANDBOX_MEMORY", "4g")
CPU_QUOTA       = int(os.getenv("SANDBOX_CPUS", "4")) * 100_000  # Docker units
WORKSPACE_VOL   = os.getenv("WORKSPACE_VOLUME", "agent-workspace")
RESOURCES_VOL   = os.getenv("RESOURCES_VOLUME", "agent-venv")
TIMEOUT_SECONDS = 120

# Retry configuration
MAX_RETRIES = int(os.getenv("DOCKER_MAX_RETRIES", "3"))
INITIAL_BACKOFF = float(os.getenv("DOCKER_INITIAL_BACKOFF", "1.0"))
MAX_BACKOFF = float(os.getenv("DOCKER_MAX_BACKOFF", "30.0"))
JITTER = float(os.getenv("DOCKER_JITTER", "0.1"))  # 10% jitter

# Optimization flags
USE_WARM_CONTAINERS = os.getenv("DOCKER_WARM_CONTAINERS", "true").lower() not in ("0", "false", "no")
CACHE_ENABLED = os.getenv("DOCKER_CACHE_ENABLED", "true").lower() not in ("0", "false", "no")

_client = docker.from_env()


# ============================================================================
# Custom Exception Classes
# ============================================================================

class DockerTimeoutError(Exception):
    """Raised when a Docker operation times out."""
    pass

class DockerImageNotFoundError(Exception):
    """Raised when a Docker image is not found."""
    pass

class DockerAPIError(Exception):
    """Raised when Docker API encounters an error."""
    pass

class GitConflictError(Exception):
    """Raised when Git encounters a merge conflict."""
    pass

class GitAuthenticationError(Exception):
    """Raised when Git authentication fails."""
    pass

class NetworkError(Exception):
    """Raised when network operations fail."""
    pass


# ============================================================================
# Retry with Exponential Backoff and Jitter
# ============================================================================

def retry_with_backoff(
    max_retries: int = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF,
    max_backoff: float = MAX_BACKOFF,
    jitter: float = JITTER,
    retryable_exceptions: tuple = (DockerTimeoutError, NetworkError, APIError),
):
    """
    Decorator for retrying operations with exponential backoff and jitter.

    Args:
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial delay in seconds
        max_backoff: Maximum delay in seconds
        jitter: Random jitter factor (0-1)
        retryable_exceptions: Tuple of exceptions that trigger retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            backoff = initial_backoff

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        # Calculate delay with jitter
                        delay = backoff * (1 + random.uniform(-jitter, jitter))
                        rprint(f"[yellow]⚠ Retry {attempt + 1}/{max_retries} after {delay:.2f}s delay...[/yellow]")
                        time.sleep(delay)
                        # Exponential backoff with cap
                        backoff = min(backoff * 2, max_backoff)
                    else:
                        rprint(f"[bold red]❌ Max retries ({max_retries}) exceeded[/bold red]")
                        raise
                except Exception as e:
                    # Non-retryable exception - fail immediately
                    last_exception = e
                    raise

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
        return wrapper
    return decorator


# ============================================================================
# Warm Container Pool
# ============================================================================

_warm_containers: dict[str, Any] = {}
_containers_lock = threading.Lock()

_BASE_VOLUMES = {
    WORKSPACE_VOL: {"bind": "/workspace", "mode": "rw"},
    RESOURCES_VOL: {"bind": "/home/agent/.local", "mode": "rw"},
}


def _container_key(image: str, allow_network: bool) -> str:
    return f"{image}:{'net' if allow_network else 'nonet'}"


def _get_warm_container(image: str, allow_network: bool) -> Any:
    """Return a running warm container, creating one if necessary."""
    key = _container_key(image, allow_network)
    with _containers_lock:
        container = _warm_containers.get(key)
        if container is not None:
            try:
                container.reload()
                if container.status == "running":
                    rprint(f"[green]♻ Reusing warm container [{key}][/green]")
                    return container
            except Exception:
                pass
            _warm_containers.pop(key, None)

        rprint(f"[yellow]🚀 Starting warm container: {image}[/yellow]")
        container = _client.containers.run(
            image=image,
            command="sleep infinity",
            detach=True,
            volumes=_BASE_VOLUMES,
            working_dir="/workspace",
            network_mode="none" if not allow_network else "bridge",
            mem_limit=MEMORY_LIMIT,
            cpu_quota=CPU_QUOTA,
            tmpfs={"/tmp": "size=256m,noexec"},
            runtime=RUNTIME,
            user="1000:1000",
            labels={"agent.warm": "true"},
        )
        # gVisor (runsc) sandbox initialization can lag behind Docker's "started"
        # signal. Poll until the container is actually running before returning.
        deadline = time.time() + 15
        while time.time() < deadline:
            container.reload()
            if container.status == "running":
                break
            time.sleep(0.25)
        else:
            container.remove(force=True)
            raise RuntimeError(f"Warm container failed to reach running state within 15s (status: {container.status})")
        _warm_containers[key] = container
        return container


def _cleanup_warm_containers() -> None:
    """Stop and remove all warm containers. Registered with atexit."""
    for key, container in list(_warm_containers.items()):
        try:
            container.stop(timeout=5)
            container.remove(force=True)
            rprint(f"[dim]Cleaned up warm container: {key}[/dim]")
        except Exception:
            pass


atexit.register(_cleanup_warm_containers)


def _exec_in_warm_container(image: str, command: str, allow_network: bool = False) -> dict:
    """
    Execute a command in a persistent warm container via exec_run.
    Falls back to a fresh container on failure.
    """
    job_id = uuid.uuid4().hex[:10]

    for attempt in range(3):
        try:
            container = _get_warm_container(image, allow_network)
            rprint(f"[yellow]⚡ Exec [{image} job:{job_id}][/yellow]")

            exit_code, output = container.exec_run(
                ["sh", "-c", command],
                workdir="/workspace",
                stdout=True,
                stderr=True,
                user="1000:1000",
            )
            return {
                "success": exit_code == 0,
                "output": output.decode(errors="replace") if output else "",
                "exit_code": exit_code,
                "job_id": job_id,
            }
        except Exception as e:
            # Evict dead container from pool and retry with a fresh one.
            # "in state stopped" typically means the container was OOM-killed
            # or crashed between the status check and the exec call.
            key = _container_key(image, allow_network)
            with _containers_lock:
                _warm_containers.pop(key, None)
            if attempt == 2:
                return {"success": False, "output": str(e), "job_id": job_id}
            rprint(f"[yellow]⚠ Container exec failed (attempt {attempt + 1}/3), retrying with fresh container...[/yellow]")

    return {"success": False, "output": "Failed after container retry", "job_id": job_id}


# ============================================================================
# Throwaway Container (fallback / extra_volumes path)
# ============================================================================

def _run_throwaway_container(
    image: str,
    command: str,
    extra_volumes: dict | None = None,
    allow_network: bool = False,
) -> dict:
    """
    Spin up a throwaway container, wait for it, return results.
    Used when extra_volumes are required (warm pool doesn't support ad-hoc mounts).
    """
    job_id = uuid.uuid4().hex[:10]
    volumes = dict(_BASE_VOLUMES)
    if extra_volumes:
        volumes.update(extra_volumes)

    kwargs = dict(
        image=image,
        command=[command],
        volumes=volumes,
        working_dir="/workspace",
        network_mode="none" if not allow_network else "bridge",
        mem_limit=MEMORY_LIMIT,
        cpu_quota=CPU_QUOTA,
        tmpfs={"/tmp": "size=256m,noexec"},
        runtime=RUNTIME,
        remove=True,
        stdout=True,
        stderr=True,
        user="1000:1000",
        labels={"agent.job": job_id},
    )

    rprint(f"[yellow]🚀 Starting container: {image} (job:{job_id})[/yellow]")

    @retry_with_backoff(
        max_retries=MAX_RETRIES,
        initial_backoff=INITIAL_BACKOFF,
        max_backoff=MAX_BACKOFF,
        jitter=JITTER,
        retryable_exceptions=(APIError, DockerTimeoutError, NetworkError)
    )
    def _run_with_retry():
        try:
            output = _client.containers.run(**kwargs)
            return {
                "success": True,
                "output": output.decode(errors="replace"),
                "job_id": job_id,
            }
        except ContainerError as e:
            error_output = e.stderr.decode(errors="replace") if e.stderr else str(e)
            if "timeout" in error_output.lower():
                raise DockerTimeoutError(f"Container timeout: {error_output}")
            return {
                "success": False,
                "output": error_output,
                "exit_code": e.exit_status,
                "job_id": job_id,
            }
        except ImageNotFound:
            raise DockerImageNotFoundError(f"Image '{image}' not found")
        except NotFound:
            raise DockerImageNotFoundError(f"Resource not found: {image}")
        except APIError as e:
            raise DockerAPIError(f"Docker API error: {e}")
        except Exception as e:
            error_str = str(e)
            if "network" in error_str.lower() or "connection" in error_str.lower():
                raise NetworkError(f"Network error: {e}")
            raise

    try:
        return _run_with_retry()
    except (DockerTimeoutError, DockerImageNotFoundError, DockerAPIError, NetworkError) as e:
        return {"success": False, "output": str(e), "exit_code": -1, "job_id": job_id}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "output": str(e), "job_id": job_id}


# ============================================================================
# Unified container dispatch
# ============================================================================

def _run_container(
    image: str,
    command: str,
    extra_volumes: dict | None = None,
    allow_network: bool = False,
) -> dict:
    """
    Run a command in a container.
    Uses warm container pool by default; falls back to throwaway when
    extra_volumes are needed or warm containers are disabled.
    """
    if USE_WARM_CONTAINERS and not extra_volumes:
        return _exec_in_warm_container(image, command, allow_network)
    return _run_throwaway_container(image, command, extra_volumes, allow_network)


# ============================================================================
# Hash-based caching
# ============================================================================

def _cache_dir() -> Path:
    workspace = Path(os.getenv("WORKSPACE_PATH", "/workspace"))
    return workspace / ".cache" / "docker_exec"


def _file_hash(path: Path) -> str:
    """MD5 hash of a single file."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _source_hash(workspace: Path) -> str:
    """MD5 hash of all Python source files (excludes .cache and .git)."""
    h = hashlib.md5()
    for f in sorted(workspace.glob("**/*.py")):
        if ".cache" in f.parts or ".git" in f.parts:
            continue
        h.update(str(f.relative_to(workspace)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _cache_get(key: str) -> str | None:
    if not CACHE_ENABLED:
        return None
    cache_file = _cache_dir() / f"{key}.json"
    try:
        if cache_file.exists():
            return json.loads(cache_file.read_text()).get("result")
    except Exception:
        pass
    return None


def _cache_set(key: str, result: str) -> None:
    if not CACHE_ENABLED:
        return
    try:
        d = _cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{key}.json").write_text(json.dumps({"result": result, "ts": time.time()}))
    except Exception:
        pass


# ============================================================================
# Public agent tools
# ============================================================================

@tool("Run Python code in sandbox")
def run_python(code: str) -> str:
    """
    Execute Python code in an isolated, network-free container.
    Returns stdout + stderr. Timeout: 120s. Memory: 4 GB.

    Use for: running scripts, quick validation, data transformations.
    Do NOT use for long test suites — use run_tests() instead.
    """
    result = _run_container(SANDBOX_IMAGE, f"python -c {_quote(code)}")
    return _format_result(result)


@tool("Run shell command in sandbox")
def run_shell(command: str) -> str:
    """
    Execute a shell command in an isolated, network-free container.
    The /workspace directory is shared — files written persist to the
    agent workspace volume.

    Use for: file manipulation, build commands, linters.
    """
    result = _run_container(SANDBOX_IMAGE, command)
    return _format_result(result)


@tool("Sync project dependencies from requirements.txt")
def sync_dependencies() -> str:
    """
    Install Python packages listed in /workspace/requirements.txt.
    Packages are stored in a persistent volume and remain available
    across all subsequent sandbox runs.

    Skips installation if requirements.txt is unchanged since last run.
    Requires a requirements.txt file to exist in the workspace.
    """
    workspace = Path(os.getenv("WORKSPACE_PATH", "/workspace"))
    req_file = workspace / "requirements.txt"

    if req_file.exists():
        req_hash = _file_hash(req_file)
        cache_key = f"deps_{req_hash}"
        cached = _cache_get(cache_key)
        if cached is not None:
            rprint("[green]⚡ Dependency cache hit — requirements.txt unchanged, skipping install[/green]")
            return f"[cache hit] Dependencies already installed (requirements.txt unchanged)\n{cached}"

    cmd = "pip install --no-cache-dir --user -r requirements.txt"
    result = _run_container(SANDBOX_IMAGE, cmd, allow_network=True)
    formatted = _format_result(result)

    if result.get("success") and req_file.exists():
        _cache_set(f"deps_{_file_hash(req_file)}", formatted)

    return formatted


@tool("Run pytest in sandbox")
def run_tests(
    test_path: str = ".",
    extra_args: str = "--tb=line -q --maxfail=5 --cov=. --cov-report=term-missing:skip-covered",
    use_cache: bool = False,
) -> str:
    """
    Run the test suite with pytest inside the sandbox.
    Returns pass/fail counts and any failure details.

    Args:
        test_path: path relative to /workspace (default: whole project)
        extra_args: additional pytest flags (coverage flags included by default)
        use_cache: if True, return cached results when source files are unchanged
    """
    if use_cache:
        workspace = Path(os.getenv("WORKSPACE_PATH", "/workspace"))
        src_hash = _source_hash(workspace)
        cache_key = f"tests_{src_hash}_{test_path.replace('/', '_')}"
        cached = _cache_get(cache_key)
        if cached is not None:
            rprint("[green]⚡ Test cache hit — source unchanged, skipping re-run[/green]")
            return f"[cache hit] Test results unchanged since last run\n{cached}"

    cmd = f"python -m pytest {test_path} {extra_args}"
    result = _run_container(SANDBOX_IMAGE, cmd)
    formatted = _format_result(result)

    if use_cache and result.get("success"):
        workspace = Path(os.getenv("WORKSPACE_PATH", "/workspace"))
        src_hash = _source_hash(workspace)
        _cache_set(f"tests_{src_hash}_{test_path.replace('/', '_')}", formatted)

    return formatted


@tool("Run static analysis and security scan")
def run_quality_gate(target_path: str = ".") -> str:
    """
    Run semgrep (SAST), bandit (Python security), and detect-secrets
    against the target path. Returns a structured report.

    Skips re-scan when source files are unchanged since last run.
    Use after the developer agent writes code, before the reviewer approves.
    """
    workspace = Path(os.getenv("WORKSPACE_PATH", "/workspace"))
    src_hash = _source_hash(workspace)
    cache_key = f"quality_{src_hash}"
    cached = _cache_get(cache_key)
    if cached is not None:
        rprint("[green]⚡ Quality gate cache hit — source unchanged, skipping re-scan[/green]")
        return f"[cache hit] Quality gate unchanged since last scan\n{cached}"

    cmd = textwrap.dedent(f"""
        echo '=== SEMGREP ===' &&
        semgrep --config=auto {target_path} --json 2>/dev/null | python -c "
import sys, json
data = json.load(sys.stdin)
findings = data.get('results', [])
print(f'Findings: {{len(findings)}}')
for f in findings[:10]:
    print(f'  [{{f[\"severity\"]}}] {{f[\"check_id\"]}} @ {{f[\"path\"]}}:{{f[\"start\"][\"line\"]}}')
        " &&
        echo '=== BANDIT ===' &&
        bandit -r {target_path} -f txt -ll 2>/dev/null | tail -20 &&
        echo '=== SECRETS ===' &&
        detect-secrets scan {target_path} 2>/dev/null | python -c "
import sys, json
data = json.load(sys.stdin)
results = data.get('results', {{}})
total = sum(len(v) for v in results.values())
print(f'Potential secrets found: {{total}}')
for path, secrets in list(results.items())[:5]:
    for s in secrets:
        print(f'  {{path}}:{{s[\"line_number\"]}} — {{s[\"type\"]}}')
        "
    """).strip()

    result = _run_container(QUALITY_IMAGE, cmd)
    formatted = _format_result(result)

    if result.get("success"):
        _cache_set(cache_key, formatted)

    return formatted


@tool("Write file to workspace")
def write_file(path: str, content: str) -> str:
    """
    Write content to a file in the shared agent workspace.
    Path is relative to /workspace. Creates parent directories.

    Use this to persist code the developer agent generates.
    """
    workspace = Path(os.getenv("WORKSPACE_PATH", "/workspace"))
    # If path already starts with /workspace (absolute from agent perspective),
    # make it relative to the workspace root for Path join
    if path.startswith(str(workspace)):
        path = os.path.relpath(path, str(workspace))

    target = (workspace / path.lstrip("/")).resolve()

    # Security check: ensure target is within workspace
    if not str(target).startswith(str(workspace)):
        return f"Error: Access denied. Path {path} is outside the workspace."

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    # Verify write immediately
    if target.exists() and target.stat().st_size == len(content.encode("utf-8")):
        agent = os.getenv("CREWAI_CURRENT_AGENT", "Unknown")
        return f"SUCCESS: Written {target} ({len(content)} chars) for {agent}"
    return f"ERROR: Failed to verify file write for {target}"


@tool("Read file from workspace")
def read_file(path: str) -> str:
    """
    Read a file from the shared agent workspace.
    Path is relative to /workspace.
    """
    workspace = Path(os.getenv("WORKSPACE_PATH", "/workspace"))
    if path.startswith(str(workspace)):
        path = os.path.relpath(path, str(workspace))

    target = (workspace / path.lstrip("/")).resolve()

    if not str(target).startswith(str(workspace)):
        return f"Error: Access denied. Path {path} is outside the workspace."

    if not target.exists():
        return f"Error: file not found: {target}"
    return target.read_text(encoding="utf-8")


@tool("List workspace files")
def list_files(directory: str = ".") -> str:
    """
    List files in the workspace directory (recursive, up to 5 levels).
    """
    workspace = Path(os.getenv("WORKSPACE_PATH", "/workspace"))
    if directory.startswith(str(workspace)):
        directory = os.path.relpath(directory, str(workspace))

    target = (workspace / directory.lstrip("/")).resolve()

    if not str(target).startswith(str(workspace)):
        return f"Error: Access denied. Directory {directory} is outside the workspace."

    if not target.exists():
        return f"Directory not found: {directory}"

    lines = [f"Listing files in {target}:"]
    # We use a custom walker to ensure we don't miss anything and have clear distinction
    for root, dirs, files in os.walk(target):
        try:
            rel_root = Path(root).relative_to(target)
            depth = len(rel_root.parts)
            if depth > 5:
                continue

            indent = "  " * depth
            if depth > 0:
                lines.append(f"{indent[:-2]}📁 {rel_root.name}/")

            for d in sorted(dirs):
                if d.startswith(".") and d != ".git": continue
                # We don't print them here, os.walk will visit them
                pass

            for f in sorted(files):
                if f.startswith(".") and f not in [".env", ".gitignore"]: continue
                lines.append(f"{indent}📄 {f}")

        except ValueError:
            continue

    return "\n".join(lines) if len(lines) > 1 else f"Listing files in {target}: (empty)"



# Helpers

def _quote(s: str) -> str:
    """Shell-safe single-quote wrapping."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _format_result(result: dict) -> str:
    status = "✓" if result["success"] else "✗"
    out = result.get("output", "").strip()
    job = result.get("job_id", "?")
    agent = os.getenv("CREWAI_CURRENT_AGENT", "Unknown")

    prefix = f"[{status} Agent: {agent} | job:{job}]"

    # Tool Output Optimization: Truncate long outputs with summary
    MAX_OUTPUT_LENGTH = int(os.getenv("MAX_TOOL_OUTPUT_LENGTH", "500"))

    if len(out) <= MAX_OUTPUT_LENGTH:
        return f"{prefix}\n{out}" if out else f"{prefix} (no output)"

    # Truncate with summary
    first_chunk = out[:250]
    last_chunk = out[-250:]
    summary = f"[Output truncated: {len(out)} chars total. Showing first 250 + last 250 chars.]"

    return f"{prefix}\n{first_chunk}\n\n... (truncated) ...\n\n{last_chunk}\n\n{summary}"
