"""
crew.py
=======
Main entry point. Defines the full task graph and runs the crew.

Pipeline:
  1. PM         → decompose feature into tasks, write tasks.md
  2. Developer  → implement each task, write code + tests
  3. Test Eng.  → verify coverage ≥ 95%, write missing tests
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

from crewai import Crew, Task, Process
from dotenv import load_dotenv
import mlflow
from tools.logging_utils import rprint
from rich.panel import Panel
from rich.rule import Rule

from tools.git_tool import sync_workspace, initialize_gitea_repository

import subprocess

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
        # This gives the best tracing experience for multi-agent loops
        # mlflow.crewai.autolog() # Temporarily disabled due to AttributeError in crewai 0.102.0
        mlflow.litellm.autolog()

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
            2. Write tests for the uncovered code in the relevant test file.
            3. Re-run tests to confirm coverage now passes.
            4. Commit: git_commit("test: improve coverage to ≥95%")

            If coverage already passes, just confirm and output the report.
        """).strip(),
        expected_output=(
            "Test suite passing with ≥ 95% coverage. Coverage report output. "
            "Any missing tests written and committed."
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
    branch_name = f"agent/loop-{loop_index}-{timestamp}"

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

    # Sequential Task Execution (Split Traces)
    rprint(Rule(f"[cyan]Crew execution (Loop {loop_index})[/cyan]"))
    
    results = []
    for i, task in enumerate(tasks):
        rprint(f"[bold cyan]>>> Executing Task {i+1}/{len(tasks)}: {task.agent.role}[/bold cyan]")
        
        with mlflow.start_run(run_name=f"loop-{loop_index}-{timestamp}--task-{i+1}-{task.agent.role}"):
            mlflow.set_tag("agent", task.agent.role)
            os.environ["CREWAI_CURRENT_AGENT"] = task.agent.role
            mlflow.log_params({
                "feature_request": feature_request,
                "loop_index": loop_index,
                "branch_name": branch_name,
                "model": os.getenv('OLLAMA_MODEL', 'qwen3-coder-next')
            })
            # Each task gets a focused Crew instance to trigger a separate MLflow trace
            task_crew = Crew(
                agents=[task.agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
                memory=False,
                max_rpm=60,
                share_crew=False,
                step_callback=step_callback, # Use module-level function
                output_log_file=f"/tmp/crew_execution_loop_{loop_index}_task_{i}.log",
                max_execution_time=14400 # 4 hours to match agent limit
            )
            
            # Kickoff task-specific trace
            with mlflow.start_span(name=f"Crew Kickoff: {task.agent.role}") as span:
                span.set_attribute("task", task.description)
                result = task_crew.kickoff()
            results.append(result)

        # Copy code from workspace to workspace-copy as WANTED_UID and WANTED_GID
        rprint(f"[cyan]Copy code from {workspace_path} to {workspace_copy}...[/cyan]")
        cmd = ["rsync", "-av", f"--chown={uid}:{gid}", f"{workspace_path}/", f"{workspace_copy}/", "--exclude", ".git", "--exclude", "__pycache__", "--delete"]
        subprocess.run(cmd, check=True, timeout=120)

        # Copy log file to workspace-progress as WANTED_UID and WANTED_GID
        log_file = f"/tmp/crew_execution_loop_{loop_index}_task_{i}.log"
        if os.path.exists(log_file):
            rprint(f"[cyan]Copy log file to {workspace_progress}...[/cyan]")
            cmd = ["rsync", "-av", f"--chown={uid}:{gid}", log_file, f"{workspace_progress}/"]
            subprocess.run(cmd, check=True, timeout=120)


    mlflow.log_metric("loop_complete", 1)
    rprint(Rule(f"[green]Loop {loop_index} Complete[/green]"))
    rprint(Panel(
        str(results[-1]), # show final result
        title=f"[bold green]Final Output (Loop {loop_index})[/bold green]",
        border_style="green",
    ))


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

    local_uid = os.getuid()
    local_gid = os.getgid()
    # Copy workspace-copy to workspace to initialize it
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