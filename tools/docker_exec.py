"""
docker_exec.py
==============
Runs arbitrary code in a throwaway, network-isolated, resource-capped
Docker container using the gVisor (runsc) runtime for kernel isolation.

Agents call run_code() or run_tests(). Never runs as root.
"""

from __future__ import annotations

import os
import uuid
import tempfile
import textwrap
from pathlib import Path
from typing import Literal

import docker
from crewai.tools import tool
from docker.errors import ContainerError, ImageNotFound
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

_client = docker.from_env()


def _run_container(
    image: str,
    command: str,
    extra_volumes: dict | None = None,
    allow_network: bool = False,
) -> dict:
    """
    Low-level: spin up a container, wait for it to finish, return results.
    Always removed after exit (auto_remove=True).
    """
    job_id = uuid.uuid4().hex[:10]
    volumes = {
        WORKSPACE_VOL: {"bind": "/workspace", "mode": "rw"},
        RESOURCES_VOL: {"bind": "/home/agent/.local", "mode": "rw"},
    }
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
    try:
        output = _client.containers.run(**kwargs)
        return {
            "success": True,
            "output": output.decode(errors="replace"),
            "job_id": job_id,
        }
    except ContainerError as e:
        return {
            "success": False,
            "output": e.stderr.decode(errors="replace") if e.stderr else str(e),
            "exit_code": e.exit_status,
            "job_id": job_id,
        }
    except ImageNotFound:
        return {
            "success": False,
            "output": f"Image '{image}' not found. Run: docker build -t {image} docker/exec/",
            "job_id": job_id,
        }
    except Exception as e:  # noqa: BLE001
        return {"success": False, "output": str(e), "job_id": job_id}



# Public agent tools

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

    Requires a requirements.txt file to exist in the workspace.
    """
    cmd = "pip install --no-cache-dir --user -r requirements.txt"
    result = _run_container(SANDBOX_IMAGE, cmd, allow_network=True)
    return _format_result(result)


@tool("Run pytest in sandbox")
def run_tests(test_path: str = ".", extra_args: str = "--tb=short -q") -> str:
    """
    Run the test suite with pytest inside the sandbox.
    Returns pass/fail counts and any failure details.

    Args:
        test_path: path relative to /workspace (default: whole project)
        extra_args: additional pytest flags
    """
    cmd = f"python -m pytest {test_path} {extra_args} --cov=. --cov-report=term-missing"
    result = _run_container(SANDBOX_IMAGE, cmd)
    return _format_result(result)


@tool("Run static analysis and security scan")
def run_quality_gate(target_path: str = ".") -> str:
    """
    Run semgrep (SAST), bandit (Python security), and detect-secrets
    against the target path. Returns a structured report.

    Use after the developer agent writes code, before the reviewer approves.
    """
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
    return _format_result(result)


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
    return f"Written: {target} ({len(content)} chars)"


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
    List files in the workspace directory (recursive, up to 3 levels).
    """
    workspace = Path(os.getenv("WORKSPACE_PATH", "/workspace"))
    if directory.startswith(str(workspace)):
        directory = os.path.relpath(directory, str(workspace))
        
    target = (workspace / directory.lstrip("/")).resolve()
    
    if not str(target).startswith(str(workspace)):
        return f"Error: Access denied. Directory {directory} is outside the workspace."
        
    if not target.exists():
        return f"Directory not found: {directory}"

    lines = []
    for p in sorted(target.rglob("*")):
        try:
            rel_p = p.relative_to(target)
            depth = len(rel_p.parts)
            if depth > 3:
                continue
            indent = "  " * (depth - 1)
            suffix = "/" if p.is_dir() else ""
            lines.append(f"{indent}{p.name}{suffix}")
        except ValueError:
            continue
    return "\n".join(lines) or "(empty)"



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
    return f"{prefix}\n{out}" if out else f"{prefix} (no output)"
