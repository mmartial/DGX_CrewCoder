# DGX Spark Multi-Agent Coding Crew

A fully local multi-agent software engineering system powered by:
- **Qwen3-Coder-Next** via Ollama on your DGX Spark
- **CrewAI** for agent orchestration
- **Docker + gVisor** for sandboxed, kernel-isolated code execution
- **Gitea** for local Git + PR workflow (SQLite backed)
- **MLflow** for observability (SQLite backed)

A learning piece for multi-agent integration, with a focus on local, private, and secure AI agents. The tool should produce `python` code.

The oriignal idea was to see what could be run on the DGX in the form of a "coding crew" of agents (this is a learning piece, not production code) and to see how well the LLMs would perform on this task. 

The first MVP was using langfuse2, I switched to MLflow to reduce needed services (such as postgres) and ease of local deployment.
Another MVP (different, unreleased code) is based on AutoGen, and I expect the next MVP to investigate Pi since I keep hearing good things about it ;)

I release this in the hope others will find it useful for their own learning and experimentation. 
If you use it, the MLFlow will allow you to see the LLM prompts and their usage.
Gitea is here to allow the various components to sync the code.
The `workspace` folder is a folder where the code built is copied over at the end of each agent run (ie the agents work in a sandbox, at the end of each step, the code is copied over to the workspace folder using `rsync`).

**Warning: Work in progress -- tested locally on a single-user DGX Spark. This README should be mostly in sync with the code, but not guaranteed.** This README is also a WiP; it works for my tests, but I'm sure it can be improved.

**Note:** The tool should work on smaller GPUs' VRAM using smaller LLMs, but it has only been tested on my DGX Spark with Qwen3-Coder-Next (69GB VRAM usage on average).

**AI usage note:** System prompts are hard to get right, so I used online models as well as local models (if it is running on Ollama already ...) to help me refine the system prompts and the code.

## Architecture

```
DGX Spark (Ollama :11434)
       │
  CrewAI Orchestrator
       │
  ┌────┴─────────────────────────────────────────┐
  │ Agents                                       │
  │  Tech PM → Developer → Test Eng → Security   │
  │  → Reviewer → Docs Writer                    │
  └────┬─────────────────────────────────────────┘
       │
  Tool Layer (filesystem, git, Docker API)
       │
  ┌────┴──────────────────────────┐
  │ Docker Sandbox (gVisor)       │
  │  exec container   (run code)  │
  │  quality container (scan)     │
  │  Shared volume  /workspace    │
  └───────────────────────────────┘
       │
  Gitea (PRs) + MLflow (traces)
```

Note: 
- `agent-runner` mounts `/var/run/docker.sock` to spawn sandbox containers. This grants the agent container full Docker daemon access — it can launch privileged containers, read other containers' filesystems, or escape to the host. This is a trade-off for DGX single-user use.
- The tool is not designed to work without MLflow as an observability tool
- The container mounts the folder this `README.md` is in as `/app`. This allows us the ability to edit the code and see the changes reflected in the container (after a `make down` [or `docker compose down agent-runner`] and `make up`) 

## Prerequisites

- Docker with gVisor runtime (`make gvisor-install`)
- Ollama running on the DGX host with Qwen pulled:
  ```bash
  ollama pull qwen3-coder-next
  ```
- Python 3.12+

## Quick Start

```bash
# Optional: create a workspace/code directory and place your existing code in it
# Important: delete any .git in that copy to avoid conflicts
# mkdir -p ~/workspace/code
# rsync -avWPR ~/myProjects/ProjectTuring/./. ~/workspace/code/.

# Install gVisor (kernel sandbox)
make gvisor-install
make gvisor-test

# Populate your .env
make dotenv
# Adapt your .env file to your preferences
# Security consideration: It is recommeneded to change the default password for Gitea
# Edit the RALPH_LOOP in particular to run it as many times as you want

# Build sandbox images and start services
make setup

# Modify the CrewAI-multiagent_tasks.md file with the feature request
...
# an example one if provided

# Run the crew on a feature
make up

# Follow the build
make logs

# Wait until all the loops are completed. You can also check the logs to see the progress.
# You will see a message like "All loops completed" when the crew is done.
# You will also see a workspace/COMMPLETED file added to the workspace folder.
# Gitea and MLflow will continue to run so you can check the code and the various agent's calls (prompts, tokens, etc.) and learn the various agents' capabilities.

# Note: To take everything down
make down

# Optional: Clean the docker compose volumes
make clean-volumes
```

## Clean everything and restart?

```bash
# Optional: update the .env file
make dotenv

# Stop the crew
make down

# Clean everything and restart
make reset
```

## Agents

Full definition in `agents/definitions.py`.
Order of execution in `crew.py` (`build_tasks` function).

| Agent | Role | Key Tools |
|---|---|---|
| Tech PM | Decomposes features into tasks | read_file, write_file |
| Senior Developer | Implements code with TDD (tests first, never leave TODO comments) | write_file, run_tests, git_commit |
| Test Engineer | Ensures ≥95% coverage and can make small code fixes to unlock it | run_tests, write_file |
| Security Auditor | Semgrep + Trivy + secrets scan | run_quality_gate |
| Security Review | Vulnerability review | git_diff, add_pr_review |
| Code Reviewer | BASSPC framework review | git_diff, add_pr_review |
| Docs Writer | README + CHANGELOG | read_file, write_file |
| Software Architect | Architecture review | read_file, write_file |

PS: I have a couple extra agents that are not used in `definitions.py` (like a frontend developer). We can add them later if needed.

Each loop of those agents relies on the "Tech PM" to read the existing code and set a context for the other agents. 
This is the `tasks.md` file is stored in `workspace/code/tasks.md`.

### BASSPC Review Framework

Every PR is reviewed against 6 dimensions before merge:

1. **B**loat — no unnecessary code or dead imports
2. **A**ssumptions — all assumptions documented
3. **S**cope — doesn't exceed the story definition
4. **S**ycophancy — doesn't blindly copy bad patterns
5. **P**ost-cleanup — debug logs and temp files removed
6. **C**LI/IO — all input validated, errors handled

## Observability

MLflow traces every LLM call. Visit `http://localhost:5000` to see:
- Which agent called what
- Token usage per agent
- Task latency
- Full prompt/response history via LiteLLM call logs
- Loop-level execution summaries

## Human Gates

Agents handle implementation volume:
- Architecture decisions (edit `CrewAI-multiagent_tasks.md` before the developer starts)

## Sandbox Security

Code runs in gVisor containers with:
- `--network=none` (no internet access)
- `--memory=4g` cap
- `--cpu-quota` limits
- Non-root user (uid 1000)
- `tmpfs` for `/tmp` (no persistence outside workspace volume)
- gVisor `runsc` runtime (separate kernel, no host escape)
