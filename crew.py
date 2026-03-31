"""
crew.py
=======
Main entry point. Defines the full task graph and runs the crew.

Pipeline:
  1. PM         → decompose feature into tasks, write tasks.md
  2. Developer  → implement each task, write code + tests
  3. Test Eng.  → verify coverage ≥ 95%, add tests and small code fixes if needed
  4. Security   → scan with semgrep/trivy/detect-secrets
  5. Sec Review → qualitative security review
  6. Reviewer   → BASSPC review, open PR
  7. Docs       → update README + CHANGELOG
  8. Architect  → update architecture.md
  9. Human gate → you review and merge in Gitea

Usage:
  python crew.py "task file path" "workspace folder path" "workspace-copy folder path" "workspace-progress folder path"

Both "copy" and "progress" are mounted in the Docker container and files will be written to them using WANTED_UID and WANTED_GID using rsync
The tool runs entirely within a container, we want to be able to see the files on the host machine, this is a simple solution to that problem.


"""



from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import time
import hashlib

from crewai import Crew, Task, Process
from dotenv import load_dotenv
import mlflow
from tools.logging_utils import rprint
from rich.panel import Panel
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from tools.git_tool import (
    sync_workspace,
    initialize_gitea_repository,
    find_existing_branch_for_feature,
    cleanup_merged_branches,
    create_optimized_branch_name,
    get_branch_stats,
    GitConflictError,
    GitAuthenticationError,
)

import subprocess
import random


load_dotenv()

from agents.definitions import (
    technical_pm,
    senior_developer,
    test_engineer,
    security_auditor,
    security_reviewer,
    software_architect,
    code_reviewer,
    docs_writer,
)

import litellm

# MLflow tracing
if os.getenv("MLFLOW_TRACKING_URI"):
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "agent-crew"))
        
        # Configure MLflow autologging for CrewAI and LiteLLM
        # Re-enabled: Fixed compatibility with crewai 0.102.0+
        try:
            mlflow.crewai.autolog()
            rprint("[green]✓ MLflow CrewAI autologging enabled[/green]")
        except AttributeError:
            # Fallback to LiteLLM autologging if CrewAI autolog fails
            mlflow.litellm.autolog()
            rprint("[yellow]⚠ Using LiteLLM autologging (CrewAI autolog unavailable)[/yellow]")

        litellm.set_verbose = os.getenv("LITELLM_VERBOSE", "false").lower() == "true"

        litellm.success_callback = ["mlflow"]
        litellm.failure_callback = ["mlflow"]

        rprint("[green]✓ MLflow tracing active (LiteLLM autologging)[/green]")
    except Exception as e:
        rprint(f"[yellow]⚠ MLflow unavailable: {e}[/yellow]")
else:
    # The tool is not designed to work without MLflow as an observability tool
    rprint("[yellow]⚠ MLflow not configured (missing MLFLOW_TRACKING_URI)[/yellow]")
    exit(1)


# Task graph factory
def build_tasks(feature_request: str, branch_name: str) -> list[Task]:
    """
    Build the full ordered task graph for a feature request.
    Each task has a clear expected_output so CrewAI knows when it's done.
    Context dependencies ensure tasks run in the right order.
    """

    # Each Task relies on an agent (from dscription.py) and a potential context (list of previous tasks) coming from a previous task
    # Each agent (in dscription.py) has a list of tools it can use (from tools/ directory)

    # Task 1: PM decomposes the feature
    plan_task = Task(
        description=textwrap.dedent(f"""
            A new feature has been requested:

            ---
            {feature_request}
            ---

            Your job:
            1. Read the existing codebase structure with list_files() to understand context.
            2. Break this feature into 3-7 concrete implementation tasks.
            3. Write the plan to /workspace/tasks.md with this exact format:

            # Feature: <name>
            ## Tasks
            ### T1: <title>
            - **Acceptance criteria**: ...
            - **Complexity**: S | M | L
            - **Agent**: Senior Developer | Frontend Developer
            - **Notes**: ...

            Be specific. "Add endpoint" is not enough — specify the route, method,
            request/response schema, and error cases.
        """).strip(),
        expected_output=(
            "A tasks.md file written to /workspace/tasks.md containing 3-7 "
            "well-defined implementation tasks with acceptance criteria."
        ),
        agent=technical_pm,
    )

    # Task 2: Developer implements the tasks
    implement_task = Task(
        description=textwrap.dedent(f"""
            Read /workspace/tasks.md and implement ALL listed tasks.

            Rules:
            - Write tests FIRST (TDD). Tests go in tests/ directory.
            - Every function needs a docstring.
            - Run your code with run_tests() after each task to confirm it passes.
            - Keep files small (< 300 lines). Split if larger.
            - Use the existing code style (run list_files() to understand structure).
            - After all tasks are done, commit with:
              git_commit("feat: {feature_request[:60]}", branch="{branch_name}")

            Do NOT open a PR yet — that's the reviewer's job.
            Do NOT skip tests to save time.
        """).strip(),
        expected_output=(
            "All tasks from tasks.md implemented with tests passing. "
            f"A single commit on branch '{branch_name}' containing the implementation."
        ),
        agent=senior_developer,
        context=[plan_task],
    )       

    # Task 3: Test Engineer checks coverage
    test_task = Task(
        description=textwrap.dedent("""
            Run the full test suite and check coverage:
              run_tests(".", "--cov=. --cov-report=term-missing --cov-fail-under=95")

            If coverage is below 95%:
            1. Read the coverage report to find uncovered lines.
            2. First try to improve coverage by writing the missing tests.
            3. If the uncovered code is hard to test because of a small defect,
               poor testability, or missing defensive handling, you may modify
               the production code as well, but keep the change tightly scoped
               to improving correctness and enabling coverage.
            4. Re-run tests to confirm coverage now passes.
            5. Commit: git_commit("test: improve coverage to ≥95%")

            If coverage already passes, just confirm and output the report.
        """).strip(),
        expected_output=(
            "Test suite passing with ≥ 95% coverage. Coverage report output. "
            "Any missing tests and narrowly scoped code fixes written and committed."
        ),
        agent=test_engineer,
        context=[implement_task],
    )

    # Task 4: Security audit
    security_task = Task(
        description=textwrap.dedent(f"""
            Run a security audit on the new code in branch '{branch_name}'.

            Steps:
            1. run_quality_gate(".") — semgrep + bandit + detect-secrets
            2. Read the output carefully.
            3. If any CRITICAL or HIGH findings:
               - Report them clearly (tool, file, line, description, remediation)
               - Mark this task as FAILED with a clear explanation
               - Do NOT approve
            4. If only MEDIUM/LOW findings, list them and mark as passed with notes.
            5. If clean, confirm and mark as passed.

            Output a structured security report.
        """).strip(),
        expected_output=(
            "Security audit report. Either: PASSED (no critical/high findings) "
            "or FAILED (list of critical/high findings with file:line references)."
        ),
        agent=security_auditor,
        context=[implement_task],
    )

    # Task 5: Qualitative security review
    security_review_task = Task(
        description=textwrap.dedent(f"""
            Perform a qualitative security review of the implementation in branch '{branch_name}'.
            Consider logic flaws, trust boundary issues, and subtle vulnerabilities.
            Reference the automated scan results in your context.
        """).strip(),
        expected_output="A qualitative security review report included in the PR or as a separate document.",
        agent=security_reviewer,
        context=[security_task],
    )

    # Task 6: Code review (BASSPC) + open PR
    review_task = Task(
        description=textwrap.dedent(f"""
            Review the implementation on branch '{branch_name}' using BASSPC,
            then open a pull request.

            Step 1 — Read the diff:
              git_diff("main", "{branch_name}")

            Step 2 — BASSPC checklist (check each explicitly):
              [ ] Bloat: Is there unnecessary code, dead imports, or over-engineering?
              [ ] Assumptions: Are assumptions documented in comments?
              [ ] Scope: Does this exceed what tasks.md specified?
              [ ] Sycophancy: Does it blindly follow bad patterns in the existing code?
              [ ] Post-cleanup: Are debug prints, TODOs, temp files removed?
              [ ] CLI/IO: Is all input validated? Are errors handled?

            Step 3 — Based on findings, EITHER:
              A) All 6 pass → add_pr_review(pr_number, body, approve=True)
              B) Any fail → add_pr_review(pr_number, body, approve=False)
                 Body must list specific file:line issues for each failing dimension.

            Step 4 — Push the branch:
              git_push("{branch_name}")

            Step 5 — Open the PR:
              open_pull_request(
                title="feat: {feature_request[:60]}",
                body="## Summary\\n...\\n## BASSPC Review\\n...",
                head_branch="{branch_name}",
                base_branch="main",
                labels=["agent-generated"]
              )

            The security audit result and qualitative review are in your context — reference them in the PR body.
        """).strip(),
        expected_output=(
            f"Pull request opened on Gitea for branch '{branch_name}'. "
            "BASSPC review completed and posted as a PR review. "
            "PR URL returned."
        ),
        agent=code_reviewer,
        context=[implement_task, test_task, security_task, security_review_task],
    )

    # Task 7: Docs update
    docs_task = Task(
        description=textwrap.dedent(f"""
            Update documentation for the shipped feature.

            1. Read the implementation files to understand what was built.
            2. Update README.md:
               - Add/update the relevant section describing the new feature.
               - Update the "Getting started" section if setup steps changed.
            3. Add a CHANGELOG.md entry:
               ## [{datetime.now().strftime('%Y-%m-%d')}]
               ### Added
               - {feature_request[:80]}
            4. Ensure all public functions have accurate docstrings
               (read them — don't assume the developer wrote good ones).
            5. Commit: git_commit("docs: update README and CHANGELOG for feature")
            6. Push: git_push()

            Docs must be accurate. Read the actual code before writing anything.
        """).strip(),
        expected_output=(
            "README.md updated with feature documentation. "
            "CHANGELOG.md updated with a dated entry. "
            "Committed to the feature branch."
        ),
        agent=docs_writer,
        context=[implement_task, review_task],
    )

    # Task 8: Architecture documentation + Final Sync
    architect_task = Task(
        description=textwrap.dedent("""
            Update /workspace/artifact/architecture.md to reflect the latest state
            of the system and the features implemented in this loop.
            
            This is the FINAL step in the pipeline. Your job is to:
            1. Read existing and new files to understand the updated architecture.
            2. Update /workspace/artifact/architecture.md with detailed component descriptions.
            3. Run git_commit("docs: update architecture and artifacts") to stage and commit ALL remaining changes.
            4. Run git_push() to send the final state of the branch to Gitea.
            5. If git_push() fails, diagnose why and resolve it before finishing.
            
            Do NOT finish until all changes are successfully pushed to Gitea.
        """).strip(),
        expected_output="Architecture documentation updated. All changes committed and pushed to Gitea.",
        agent=software_architect,
        context=[docs_task],
    )

    return [plan_task, implement_task, test_task, security_task, security_review_task, review_task, docs_task, architect_task]


# Global context for callbacks
_LOG_CONTEXT = {
    "loop_index": 0,
    "trace_path": None,
    "workspace_path": "/workspace",
    "workspace_copy": "/workspace-copy",
    "workspace_progress": "/workspace-progress",
}

# ============================================================================
# MLflow Detailed Tracking Helpers
# ============================================================================

def log_task_metrics(task_index: int, task_role: str, duration: float, 
                     success: bool, attempt: int, metrics: dict | None = None):
    """
    Log detailed task-level metrics to MLflow.
    
    Args:
        task_index: Task number in the crew
        task_role: Agent role name
        duration: Task duration in seconds
        success: Whether the task succeeded
        attempt: Attempt number (0-indexed)
        metrics: Additional metrics to log
    """
    try:
        mlflow.log_metric(f"task_{task_index}_duration_sec", duration)
        mlflow.log_metric(f"task_{task_index}_success", 1 if success else 0)
        mlflow.log_metric(f"task_{task_index}_attempts", attempt + 1)
        
        if metrics:
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
                
        rprint(f"[cyan]✓ Logged task metrics for {task_role}[/cyan]")
    except Exception as e:
        rprint(f"[yellow]⚠ Failed to log task metrics: {e}[/yellow]")

def log_agent_performance(role: str, success_rate: float, avg_duration: float, 
                          total_attempts: int):
    """
    Log agent performance metrics to MLflow.
    
    Args:
        role: Agent role name
        success_rate: Success rate (0-1)
        avg_duration: Average task duration in seconds
        total_attempts: Total attempts across all tasks
    """
    try:
        mlflow.log_metric(f"{role}_success_rate", success_rate)
        mlflow.log_metric(f"{role}_avg_duration_sec", avg_duration)
        mlflow.log_metric(f"{role}_total_attempts", total_attempts)
    except Exception as e:
        rprint(f"[yellow]⚠ Failed to log agent performance: {e}[/yellow]")

def log_resource_utilization(container_name: str, memory_mb: float, cpu_percent: float):
    """
    Log resource utilization metrics to MLflow.
    
    Args:
        container_name: Name of the container
        memory_mb: Memory usage in MB
        cpu_percent: CPU usage percentage
    """
    try:
        mlflow.log_metric(f"{container_name}_memory_mb", memory_mb)
        mlflow.log_metric(f"{container_name}_cpu_percent", cpu_percent)
    except Exception as e:
        rprint(f"[yellow]⚠ Failed to log resource utilization: {e}[/yellow]")

def log_branch_stats(stats: dict):
    """
    Log branch management statistics to MLflow.
    
    Args:
        stats: Dictionary of branch statistics
    """
    try:
        mlflow.log_metric("total_branches", stats.get("total_branches", 0))
        mlflow.log_metric("agent_branches", stats.get("agent_branches", 0))
        mlflow.log_param("current_branch", stats.get("current_branch", ""))
        mlflow.log_param("branch_prefix", stats.get("agent_branch_prefix", ""))
    except Exception as e:
        rprint(f"[yellow]⚠ Failed to log branch stats: {e}[/yellow]")

def step_callback(step):
    """Callback triggered after each agent step. Defined at module level for Pydantic."""
    loop_index = _LOG_CONTEXT.get("loop_index", 0)
    trace_path = _LOG_CONTEXT.get("trace_path")
    workspace_path = _LOG_CONTEXT.get("workspace_path", "/workspace")
    workspace_copy = _LOG_CONTEXT.get("workspace_copy", "/workspace-copy")
    workspace_progress = _LOG_CONTEXT.get("workspace_progress", "/workspace-progress")
    
    uid = os.getenv("WANTED_UID")
    gid = os.getenv("WANTED_GID")
    
    if trace_path:
        # Manual MLflow Tracing (restoring visibility)
        try:
            span_name = f"Agent Step: {step.agent}" if hasattr(step, "agent") else "Agent Step"
            if hasattr(step, "tool") and step.tool:
                span_name = f"Tool: {step.tool}"
            
            with mlflow.start_span(name=span_name) as span:
                if hasattr(step, "thought") and step.thought:
                    span.set_attribute("mlflow.spanInputs", {"thought": step.thought})
                if hasattr(step, "tool") and step.tool:
                    span.set_attribute("tool", step.tool)
                    if hasattr(step, "tool_input"):
                        span.set_attribute("tool_input", str(step.tool_input))
                if hasattr(step, "result") and step.result:
                    span.set_attribute("mlflow.spanOutputs", {"result": str(step.result)})
        except Exception as e:
            rprint(f"[yellow]⚠ MLflow step trace failed: {e}[/yellow]")

        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(f"### [{datetime.now().strftime('%H:%M:%S')}] Agent Step\n")
            if hasattr(step, "agent"):
                f.write(f"**Agent:** {step.agent}\n\n")
            if hasattr(step, "thought") and step.thought:
                f.write(f"**Thought:**\n> {step.thought}\n\n")
            if hasattr(step, "tool") and step.tool:
                f.write(f"**Tool:** `{step.tool}`\n")
                if hasattr(step, "tool_input"):
                    f.write(f"**Input:** `{step.tool_input}`\n")
            f.write("---\n\n")

            # Copy trace to workspace-progress as WANTED_UID and WANTED_GID
            rprint(f"[cyan]Copy trace to {workspace_progress}...[/cyan]")
            cmd = ["rsync", "-av", f"--chown={uid}:{gid}", trace_path, f"{workspace_progress}/"]
            subprocess.run(cmd, check=True, timeout=120)

    # Copy code from workspace to workspace-copy as WANTED_UID and WANTED_GID
    rprint(f"[cyan]Copy code from {workspace_path} to {workspace_copy}...[/cyan]")
    cmd = ["rsync", "-av", f"--chown={uid}:{gid}", f"{workspace_path}/", f"{workspace_copy}/", "--exclude", ".git", "--exclude", "__pycache__", "--delete"]
    subprocess.run(cmd, check=True, timeout=120)

    # Copy log file to workspace-progress as WANTED_UID and WANTED_GID
    log_file = f"/tmp/crew_execution_loop_{loop_index}.log"
    if os.path.exists(log_file):
        rprint(f"[cyan]Copy log file to {workspace_progress}...[/cyan]")
        cmd = ["rsync", "-av", f"--chown={uid}:{gid}", log_file, f"{workspace_progress}/"]
        subprocess.run(cmd, check=True, timeout=120)


# Run
def run(feature_request: str, loop_index: int = 0, workspace_path: str = "/workspace", workspace_copy: str = "/workspace-copy", workspace_progress: str = "/workspace-progress") -> None:
    # Synchronize workspace before starting
    sync_result = sync_workspace()
    rprint(f"[cyan]Workspace sync:[/cyan] {sync_result}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Branch Management Optimization: Check for existing branch reuse
    existing_branch = find_existing_branch_for_feature(feature_request)
    if existing_branch:
        branch_name = existing_branch
        rprint(f"[cyan]🔄 Reusing existing branch: {branch_name}[/cyan]")
    else:
        branch_name = create_optimized_branch_name(feature_request, loop_index)
        rprint(f"[cyan]📝 Created optimized branch name: {branch_name}[/cyan]")

    rprint(Panel(
        f"[bold]Feature:[/bold] {feature_request}\n"
        f"[bold]Loop:[/bold]    {loop_index}\n"
        f"[bold]Branch:[/bold]  {branch_name}\n"
        f"[bold]Model:[/bold]   {os.getenv('OLLAMA_MODEL', 'qwen3-coder-next')}",
        title="[bold cyan]Agent Crew Starting[/bold cyan]",
        border_style="cyan",
    ))

    tasks = build_tasks(feature_request, branch_name)

    # Global state for step_callback
    trace_path = f"/tmp/trace_loop_{loop_index}.md"
    _LOG_CONTEXT.update({
        "loop_index": loop_index,
        "trace_path": trace_path,
        "workspace_path": workspace_path,
        "workspace_copy": workspace_copy,
        "workspace_progress": workspace_progress,
    })

    if not os.path.exists(trace_path):
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write(f"# Trace: Loop {loop_index}\n\n")

    uid = os.getenv("WANTED_UID")
    gid = os.getenv("WANTED_GID")

    # Branch Management: Log branch stats
    branch_stats = get_branch_stats()
    log_branch_stats(branch_stats)
    rprint(f"[cyan]📊 Branch stats: {branch_stats['agent_branches']} agent branches[/cyan]")

    # Sequential Task Execution with progress indicators and improved error handling
    rprint(Rule(f"[cyan]Crew execution (Loop {loop_index})[/cyan]"))
    
    results = []
    failure_occurred = False
    non_blocking_failure_roles = {"Test Engineer"}
    
    # Progress tracking
    total_tasks = len(tasks)
    completed_tasks = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        transient=True,
        console=None  # Uses Rich's default console
    ) as progress:
        
        main_task = progress.add_task("[cyan]Executing crew tasks...", total=total_tasks)
        
        for i, task in enumerate(tasks):
            task_desc = f"[bold cyan]Task {i+1}/{total_tasks}: {task.agent.role}[/bold cyan]"
            progress.update(main_task, description=task_desc)
            rprint(f"\n{task_desc}")
            
            max_retries = int(os.getenv("TASK_RETRIES", "2"))
            last_result = None
            task_failed_permanently = False
            original_description = task.description  # Capture once before retry loop

            for attempt in range(max_retries + 1):
                execution_error = False
                if attempt > 0:
                    # Exponential backoff for retries
                    backoff = min(2 ** attempt, 30)  # Cap at 30 seconds
                    jitter = random.uniform(-0.1, 0.1)  # 10% jitter
                    delay = backoff * (1 + jitter)
                    rprint(f"[yellow]🔁 Retrying Task {i+1} (Attempt {attempt}/{max_retries}) after {delay:.1f}s delay...[/yellow]")
                    time.sleep(delay)
                    # Inject failure context into task description for the retry
                    failure_context = f"\n\n[RETRY CONTEXT] Your previous attempt FAILED with the following output:\n---\n{last_result}\n---\nPlease analyze the error and try again. Ensure all requested files are written and verified."
                    task.description = original_description + failure_context

                # End any auto-created run (e.g. from mlflow autologging or orphaned log_metric calls)
                if mlflow.active_run() is not None:
                    mlflow.end_run()
                with mlflow.start_run(run_name=f"loop-{loop_index}-{timestamp}--task-{i+1}-{task.agent.role}-att-{attempt}"):
                    mlflow.set_tag("agent", task.agent.role)
                    mlflow.set_tag("attempt", attempt)
                    os.environ["CREWAI_CURRENT_AGENT"] = task.agent.role
                    mlflow.log_params({
                        "feature_request": feature_request,
                        "loop_index": loop_index,
                        "branch_name": branch_name,
                        "attempt": attempt
                    })

                    task_crew = Crew(
                        agents=[task.agent],
                        tasks=[task],
                        process=Process.sequential,
                        verbose=True,
                        memory=False,
                        max_rpm=60,
                        share_crew=False,
                        step_callback=step_callback,
                        output_log_file=f"/tmp/crew_execution_loop_{loop_index}_task_{i}_att_{attempt}.log",
                        max_execution_time=14400
                    )
                    
                    with mlflow.start_span(name=f"Crew Kickoff: {task.agent.role} (att {attempt})") as span:
                        span.set_attribute("task", task.description)
                        try:
                            result = task_crew.kickoff()
                            last_result = result.raw
                        except GitConflictError as e:
                            execution_error = True
                            rprint(f"[bold red]⚠ Git conflict during Task {i+1} ({task.agent.role}):[/bold red] {e}")
                            last_result = f"GIT_CONFLICT: {str(e)}"
                            from unittest.mock import MagicMock
                            result = MagicMock()
                            result.raw = last_result
                        except GitAuthenticationError as e:
                            execution_error = True
                            rprint(f"[bold red]⚠ Git authentication error during Task {i+1} ({task.agent.role}):[/bold red] {e}")
                            last_result = f"GIT_AUTH_ERROR: {str(e)}"
                            from unittest.mock import MagicMock
                            result = MagicMock()
                            result.raw = last_result
                        except Exception as e:
                            execution_error = True
                            rprint(f"[bold red]⚠ Exception during Task {i+1} ({task.agent.role}):[/bold red] {e}")
                            last_result = f"EXCEPTION during execution: {str(e)}"
                            # Create a mock result object to satisfy the rest of the loop
                            from unittest.mock import MagicMock
                            result = MagicMock()
                            result.raw = last_result
                    
                    # --- Synchronize State (Code + Logs) ---
                    rprint(f"[cyan]Copying latest state to host...[/cyan]")
                    cmd_code = ["rsync", "-av", f"--chown={uid}:{gid}", f"{workspace_path}/", f"{workspace_copy}/", "--exclude", ".git", "--exclude", "__pycache__", "--delete"]
                    subprocess.run(cmd_code, check=True, timeout=120)
                    
                    log_file = f"/tmp/crew_execution_loop_{loop_index}_task_{i}_att_{attempt}.log"
                    if os.path.exists(log_file):
                        cmd_log = ["rsync", "-av", f"--chown={uid}:{gid}", log_file, f"{workspace_progress}/"]
                        subprocess.run(cmd_log, check=True, timeout=120)

                    # --- Failure Gate with early termination on critical failures ---
                    raw_result = getattr(result, "raw", "") or ""
                    clean_result = raw_result.strip().upper()
                    failure_prefixes = (
                        "FAILED:",
                        "ERROR:",
                        "EXCEPTION:",
                        "CRITICAL FAILURE:",
                        "CRITICAL ERROR:",
                        "FATAL:",
                        "ABORT:",
                        "GIT_CONFLICT:",
                        "GIT_AUTH_ERROR:",
                    )

                    # Only treat structured failure prefixes or raised exceptions
                    # as task failure. This avoids false negatives when the model
                    # echoes retry instructions containing words like "FAILED".
                    has_failure_marker = execution_error or clean_result.startswith(failure_prefixes)
                    
                    if has_failure_marker:
                        rprint(f"\n[bold red]✖ Task {i+1} ({task.agent.role}) reported FAILURE on attempt {attempt}.[/bold red]")
                        is_non_blocking_failure = task.agent.role in non_blocking_failure_roles
                        
                        # Check for critical failure markers that warrant early termination
                        critical_prefixes = (
                            "CRITICAL FAILURE:",
                            "CRITICAL ERROR:",
                            "FATAL:",
                            "ABORT:",
                        )
                        is_critical = clean_result.startswith(critical_prefixes)
                        
                        if is_critical:
                            rprint(f"[bold red]⚠ CRITICAL FAILURE DETECTED - Early termination initiated[/bold red]")
                            failure_occurred = True
                            task_failed_permanently = True
                            break  # Out of retry loop - don't retry critical failures
                        elif attempt < max_retries:
                            rprint(f"[yellow]Triggering retry...[/yellow]")
                            failure_occurred = True
                        else:
                            panel_style = "yellow" if is_non_blocking_failure else "red"
                            panel_title = (
                                f"[bold yellow]Non-blocking Failure: {task.agent.role}[/bold yellow]"
                                if is_non_blocking_failure
                                else f"[bold red]Final Failure: {task.agent.role}[/bold red]"
                            )
                            rprint(Panel(raw_result, title=panel_title, border_style=panel_style))

                            if is_non_blocking_failure:
                                rprint(
                                    f"[yellow]Continuing pipeline despite {task.agent.role} failure; "
                                    "later agents will still receive this output in context.[/yellow]"
                                )
                                results.append(result)
                                failure_occurred = False
                                task_failed_permanently = False
                                break  # Out of retry loop; continue to next task

                            failure_occurred = True
                            task_failed_permanently = True
                            break  # Out of retry loop
                    else:
                        rprint(f"[bold green]✓ Task {i+1} ({task.agent.role}) PASSED on attempt {attempt}.[/bold green]")
                        results.append(result)
                        failure_occurred = False
                        task_failed_permanently = False
                        break  # Out of retry loop successes
            
            # Update progress
            if not task_failed_permanently:
                completed_tasks += 1
                progress.update(main_task, completed=completed_tasks)
            
            # Early termination on critical failures
            if task_failed_permanently:
                rprint(f"[red]Stopping pipeline execution for this feature loop due to critical failure.[/red]\n")
                if mlflow.active_run() is not None:
                    mlflow.log_metric(f"task_{i+1}_failed", 1)
                    mlflow.log_metric("early_termination", 1)
                break  # Out of tasks loop
    
    # Final progress update
    progress.update(main_task, description="[green]Crew execution complete![/green]")
    progress.update(main_task, completed=total_tasks)
    
    if failure_occurred:
        rprint(Rule(f"[red]Loop {loop_index} FAILED[/red]"))
        sys.exit(1)

    if mlflow.active_run() is not None:
        mlflow.log_metric("loop_complete", 1)
    
    # Log agent performance metrics
    agent_metrics = {}
    for i, task in enumerate(tasks):
        role = task.agent.role
        if role not in agent_metrics:
            agent_metrics[role] = {"successes": 0, "attempts": 0}
        agent_metrics[role]["attempts"] += 1
    
    for role, metrics in agent_metrics.items():
        success_rate = metrics["successes"] / metrics["attempts"] if metrics["attempts"] > 0 else 0
        log_agent_performance(role, success_rate, 0, metrics["attempts"])
    
    rprint(Rule(f"[green]Loop {loop_index} Complete[/green]"))
    rprint(Panel(
        str(results[-1]), # show final result
        title=f"[bold green]Final Output (Loop {loop_index})[/bold green]",
        border_style="green",
    ))
    
    # Branch Management: Cleanup merged branches
    cleanup_result = cleanup_merged_branches()
    rprint(f"[cyan]🧹 Branch cleanup: {cleanup_result}[/cyan]")


# Entry point
if __name__ == "__main__":
    task_file = sys.argv[1] if len(sys.argv) > 1 else "/app/CrewAI-multiagent_tasks.md"
    workspace_path = sys.argv[2] if len(sys.argv) > 2 else "/workspace"
    workspace_copy = sys.argv[3] if len(sys.argv) > 3 else "/workspace-copy"
    workspace_progress = sys.argv[4] if len(sys.argv) > 4 else "/workspace-progress"

    # Make sure WANTED_UID and WANTED_GID are set
    uid = os.getenv("WANTED_UID")
    gid = os.getenv("WANTED_GID")
    if not uid or not gid:
        rprint(f"[red]Error:[/red] WANTED_UID and WANTED_GID must be set.")
        sys.exit(1)

    try:
        feature = Path(task_file).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        rprint(f"[red]Error:[/red] Task file '{task_file}' not found.\n"
               f"Create it with your feature request, e.g.:\n\n"
               f"  echo 'Add JWT auth' > task.txt\n\n"
               f"Or pass a different file: python crew.py my_task.txt")
        sys.exit(1)
    if not feature:
        rprint(f"[red]Error:[/red] '{task_file}' is empty.")
        sys.exit(1)

    # 0. delete the COMPLETED file if it exists + Copy workspace-copy to workspace to initialize it
    if os.path.exists(f"{workspace_copy}/COMPLETED"):
        os.remove(f"{workspace_copy}/COMPLETED")

    local_uid = os.getuid()
    local_gid = os.getgid()

    cmd = ["rsync", "-av", f"--chown={local_uid}:{local_gid}", workspace_copy+"/", workspace_path+"/"]
    subprocess.run(cmd, check=True, timeout=120)

    # Change to workspace path, DO NOT USE any other path for content to work from
    os.chdir(workspace_path)

    # 1. Initialize Gitea before any loops start
    # This is the "calling code" handling the failure-prone setup
    try:
        rprint("[cyan]Initializing Gitea repository...[/cyan]")
        init_result = initialize_gitea_repository()
        rprint(f"[green]Initialization success:[/green] {init_result}")
    except Exception as e:
        rprint(f"\n[bold red]FATAL: Initialization failed[/bold red]\n{e}")
        sys.exit(1)

    # 2. Run agent loops
    num_loops = int(os.getenv("RALPH_LOOP", "1"))
    for i in range(num_loops):
        run(feature, loop_index=i)

    # 3. Exit with success code
    rprint("[green]All loops completed successfully.[/green]")
    # Copy a  "COMPLETED" file to workspace_copy (the location OUTSIDE of the container)
    with open("/tmp/COMPLETED", "w", encoding="utf-8") as f:
        f.write("All loops completed successfully.")
    # Copy the completed file to workspace_copy
    cmd = ["rsync", "-av", f"--chown={uid}:{gid}", "/tmp/COMPLETED", f"{workspace_copy}/"]
    subprocess.run(cmd, check=True, timeout=120)

    exit(0)
