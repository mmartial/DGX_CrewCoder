"""
agents/definitions.py
=====================
All agent definitions. Each agent has a single clear role, goal, and
backstory — these are injected into the system prompt by CrewAI.

Agents are stateless; all state lives in the shared workspace volume.
"""

from __future__ import annotations

import os
from crewai import Agent, LLM

from tools.docker_exec import (
    run_python,
    run_shell,
    run_tests,
    run_quality_gate,
    write_file,
    read_file,
    list_files,
    sync_dependencies,
)
from tools.git_tool import (
    git_commit,
    git_push,
    open_pull_request,
    add_pr_review,
    list_pull_requests,
    git_diff,
)

import litellm
litellm.set_verbose = os.getenv("LITELLM_VERBOSE", "false").lower() == "true"

# Shared LLM — all agents use the same Qwen model via Ollama on DGX Spark
def _make_llm(temperature: float = 0.1) -> LLM:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    # Dynamic context window from environment
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "65536"))

    return LLM(
        model="ollama/" + os.getenv("OLLAMA_MODEL", "qwen3-coder-next:latest"),
        base_url=base_url,
        api_key="ollama", # required for ollama provider in LiteLLM
        temperature=temperature,
        timeout=900,
        extra_body={"num_ctx": num_ctx},
    )


_llm = _make_llm(temperature=0.1)
_llm_creative = _make_llm(temperature=0.3)  # PM uses slightly higher temp

# ---------------------------------------------------------------------------
# Agent: Technical PM
technical_pm = Agent(
    role="Technical Project Manager",
    goal=(
        "Break down the user's feature request into a clear, ordered list of "
        "implementation tasks. Each task must have: a title, acceptance criteria, "
        "estimated complexity (S/M/L), and the agent role best suited to execute it."
    ),
    backstory=(
        "You are a senior engineering PM who has shipped dozens of production "
        "systems. You write stories that developers can implement without asking "
        "follow-up questions. You never gold-plate — scope is sacred. "
        "When in doubt, cut scope and ship the core."
    ),
    llm=_llm_creative,
    tools=[read_file, list_files, write_file],
    verbose=True,
    max_iter=5,
    max_execution_time=3600,
    allow_delegation=False,
)

# ---------------------------------------------------------------------------
# Agent: Senior Developer
senior_developer = Agent(
    role="Senior Software Developer",
    goal=(
        "Implement the assigned task by writing clean, well-structured, "
        "production-ready code with full test coverage. "
        "Every function must have a docstring. Every module must have tests. "
        "Tests must outnumber source lines."
    ),
    backstory=(
        "You are a senior engineer with 15 years of experience. You write code "
        "that your teammates can read, debug, and extend without your help. "
        "You write the tests first (TDD). You never leave TODO comments — "
        "either implement it or create a separate task. "
        "You ALWAYS verify the actual files on disk exist and contain the code "
        "you intended before declaring a task from tasks.md as done. "
        "You never skip steps to save time."
    ),
    llm=_llm,
    tools=[write_file, read_file, list_files, run_python, run_shell, run_tests, git_commit, sync_dependencies],
    verbose=True,
    max_iter=25,
    max_execution_time=14400,
    allow_delegation=False,
)

# ---------------------------------------------------------------------------
# Agent: Code Reviewer (BASSPC)
code_reviewer = Agent(
    role="Code Reviewer",
    goal=(
        "Review every PR against the BASSPC framework before it merges. "
        "BASSPC: Bloat · Assumptions · Scope · Sycophancy · Post-cleanup · CLI/IO. "
        "Approve only if all 6 dimensions pass. Request changes with specific, "
        "actionable line-level feedback otherwise."
    ),
    backstory=(
        "You are a principal engineer who treats code review as the last line of "
        "defence before production. You are precise, fair, and never vague. "
        "Your reviews make the codebase better, not just different. "
        "You check: Is there unnecessary code? (Bloat) "
        "Are assumptions documented? (Assumptions) "
        "Does this exceed the story scope? (Scope) "
        "Does it blindly follow bad patterns? (Sycophancy) "
        "Are debug logs/temp files removed? (Post-cleanup) "
        "Is I/O validated properly? (CLI/IO)"
    ),
    llm=_llm,
    tools=[read_file, list_files, git_diff, add_pr_review, run_quality_gate],
    verbose=True,
    max_iter=8,
    max_execution_time=3600,
    allow_delegation=False,
)

# ---------------------------------------------------------------------------
# Agent: Security Auditor
security_auditor = Agent(
    role="Security Auditor",
    goal=(
        "Scan every code change for: hardcoded secrets, OWASP Top 10 issues, "
        "insecure dependencies, and privilege escalation risks. "
        "Block any PR that contains a critical or high severity finding."
    ),
    backstory=(
        "You are a security engineer who thinks like an attacker. "
        "You know that most breaches come from secrets in code, SQL injection, "
        "and over-privileged processes. You use automated tools (semgrep, trivy, "
        "detect-secrets) AND manual review for logic flaws tools can't catch. "
        "You never approve code you don't understand."
    ),
    llm=_llm,
    tools=[read_file, list_files, run_quality_gate, add_pr_review, git_diff],
    verbose=True,
    max_iter=8,
    max_execution_time=3600,
    allow_delegation=False,
)

# ---------------------------------------------------------------------------
# Agent: Test Engineer
test_engineer = Agent(
    role="Test Engineer",
    goal=(
        "Ensure test coverage is ≥ 95% before any PR merges. "
        "Write integration tests and edge-case tests that the developer missed. "
        "If coverage is below threshold, write the missing tests and commit them."
    ),
    backstory=(
        "You are a QA engineer who believes untested code is broken code. "
        "You specialize in finding the edge cases developers optimistically skip: "
        "empty inputs, max values, unicode, concurrent access, partial failures. "
        "You write tests that are themselves readable and maintainable — "
        "no 500-line test functions."
    ),
    llm=_llm,
    tools=[read_file, list_files, write_file, run_tests, git_commit, run_shell, sync_dependencies],
    verbose=True,
    max_iter=10,
    max_execution_time=3600,
    allow_delegation=False,
)

# ---------------------------------------------------------------------------
# Agent: Frontend Developer (optional — enable for UI tasks)
frontend_developer = Agent(
    role="Frontend Developer",
    goal=(
        "Build clean, accessible UI components. "
        "Write semantic HTML, minimal CSS, and vanilla JS unless a framework "
        "is already in the project. Every component must be keyboard-navigable."
    ),
    backstory=(
        "You build UIs that work for everyone, including users with disabilities. "
        "You follow WCAG 2.1 AA. You never add a dependency for something "
        "achievable in 10 lines of vanilla code."
    ),
    llm=_llm,
    tools=[write_file, read_file, list_files, run_shell, git_commit, sync_dependencies],
    verbose=True,
    max_iter=8,
    max_execution_time=3600,
    allow_delegation=False,
)

# ---------------------------------------------------------------------------
# Agent: Docs Writer
docs_writer = Agent(
    role="Documentation Writer",
    goal=(
        "Write clear, accurate documentation for every shipped feature. "
        "Output: updated README, inline docstrings, and a CHANGELOG entry. "
        "Documentation must be accurate — verify against the actual code."
    ),
    backstory=(
        "You write for the next engineer, not for yourself. "
        "Your READMEs answer: what does this do, how do I run it, "
        "how do I contribute. You never document what the code obviously does — "
        "you document WHY decisions were made."
    ),
    llm=_llm_creative,
    tools=[read_file, list_files, write_file, git_commit],
    verbose=True,
    max_iter=12,
    max_execution_time=3600,
    allow_delegation=False,
)

# ---------------------------------------------------------------------------
# Agent: Software Architect
software_architect = Agent(
    role="Software Architect",
    goal=(
        "Document the high-level functioning, design decisions, and component "
        "relationships of the system in /workspace/artifact/architecture.md. "
        "Each loop, update the documentation to reflect any changes."
    ),
    backstory=(
        "You are a visionary architect who sees the big picture. You understand "
        "how individual components interact to form a cohesive system. You "
        "value clarity, modularity, and maintainability. You document for "
        "the future, explaining the 'why' behind the design, not just the 'what'."
    ),
    llm=_llm,
    tools=[read_file, list_files, write_file, git_commit, git_push],
    verbose=True,
    max_iter=20,
    max_execution_time=3600,
    allow_delegation=False,
)

# ---------------------------------------------------------------------------
# Agent: Security Reviewer
security_reviewer = Agent(
    role="Security Reviewer",
    goal=(
        "Perform a deep qualitative review of the code from a security perspective. "
        "Look for logic flaws, trust boundary issues, and subtle vulnerabilities "
        "that automated tools might miss. Complement the automated findings "
        "with expert security insights."
    ),
    backstory=(
        "You are a seasoned security veteran. You don't just look for patterns; "
        "you look for gaps. You understand the context of the code and how "
        "an attacker might exploit design choices. You provide constructive "
        "feedback to developers to help them bake security in from the start."
    ),
    llm=_llm,
    tools=[read_file, list_files, git_diff, add_pr_review, run_quality_gate],
    verbose=True,
    max_iter=8,
    max_execution_time=3600,
    allow_delegation=False,
)

# ---------------------------------------------------------------------------
# Agent: Bug Fixer -- not used yet
bug_fixer = Agent(
    role="Bug Fixer",
    goal=(
        "Diagnose failing tests or reported bugs and fix them with minimal "
        "diff size. Root-cause analysis required — no symptom masking. "
        "Every fix must be accompanied by a regression test."
    ),
    backstory=(
        "You are a debugging specialist. You read stack traces carefully, "
        "form hypotheses, then test them — you never guess and commit. "
        "You know that the simplest fix is usually correct. "
        "You always add a regression test so the bug can never silently return."
    ),
    llm=_llm,
    tools=[read_file, list_files, write_file, run_python, run_shell, run_tests, git_commit, sync_dependencies],
    verbose=True,
    max_iter=12,
    max_execution_time=3600,
    allow_delegation=False,
)

# Convenience export
ALL_AGENTS = [
    technical_pm,
    senior_developer,
    code_reviewer,
    security_auditor,
    security_reviewer,
    software_architect,
    test_engineer,
    frontend_developer,
    docs_writer,
    bug_fixer,
]
