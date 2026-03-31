"""
git_tool.py
===========
Git operations and Gitea PR automation for the agent crew.
Agents commit code, open PRs, and request reviews — just like the slide shows.
"""

from __future__ import annotations

import os
import time
import random
from pathlib import Path
from datetime import datetime

import httpx
from crewai.tools import tool
from git import Repo, InvalidGitRepositoryError, GitCommandError, GitCommandNotFound
import functools

# Config
GITEA_URL   = os.getenv("GITEA_URL", "http://localhost:3000")
GITEA_TOKEN = os.getenv("GITEA_TOKEN", "")
GITEA_USER  = os.getenv("GITEA_USER", "agent-bot")
GITEA_REPO  = os.getenv("GITEA_REPO", "workspace")
WORKSPACE   = Path(os.getenv("WORKSPACE_PATH", "/workspace"))

_headers = {
    "Authorization": f"token {GITEA_TOKEN}",
    "Content-Type": "application/json",
}

# Set once the first pull has been completed in this pipeline run.
# Subsequent sync_workspace() calls commit local changes but skip the remote pull
# because agents run sequentially on a shared workspace volume — the previous
# agent already pushed its work, so pulling again is redundant.
_workspace_pulled_this_session: bool = False

# Retry configuration
GIT_MAX_RETRIES = int(os.getenv("GIT_MAX_RETRIES", "3"))
GIT_INITIAL_BACKOFF = float(os.getenv("GIT_INITIAL_BACKOFF", "1.0"))
GIT_MAX_BACKOFF = float(os.getenv("GIT_MAX_BACKOFF", "30.0"))
GIT_JITTER = float(os.getenv("GIT_JITTER", "0.1"))  # 10% jitter


# ============================================================================
# Custom Git Exception Classes
# ============================================================================

class GitConflictError(Exception):
    """Raised when Git encounters a merge conflict."""
    pass

class GitAuthenticationError(Exception):
    """Raised when Git authentication fails."""
    pass

class GitNetworkError(Exception):
    """Raised when Git network operations fail."""
    pass

class GitRepositoryError(Exception):
    """Raised when Git repository operations fail."""
    pass


# ============================================================================
# Retry with Exponential Backoff and Jitter for Git Operations
# ============================================================================

def retry_git_operation(
    max_retries: int = GIT_MAX_RETRIES,
    initial_backoff: float = GIT_INITIAL_BACKOFF,
    max_backoff: float = GIT_MAX_BACKOFF,
    jitter: float = GIT_JITTER,
):
    """
    Decorator for retrying Git operations with exponential backoff and jitter.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial delay in seconds
        max_backoff: Maximum delay in seconds
        jitter: Random jitter factor (0-1)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            backoff = initial_backoff
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (GitNetworkError, GitConflictError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        # Calculate delay with jitter
                        delay = backoff * (1 + random.uniform(-jitter, jitter))
                        rprint(f"[yellow]⚠ Git retry {attempt + 1}/{max_retries} after {delay:.2f}s delay...[/yellow]")
                        time.sleep(delay)
                        # Exponential backoff with cap
                        backoff = min(backoff * 2, max_backoff)
                    else:
                        rprint(f"[bold red]❌ Git max retries ({max_retries}) exceeded[/bold red]")
                        raise
                except Exception as e:
                    # Non-retryable exception - fail immediately
                    last_exception = e
                    raise
            
            if last_exception:
                raise last_exception
        return wrapper
    return decorator


# ============================================================================
# Git Error Handling Helpers
# ============================================================================

def _handle_git_error(func):
    """Decorator to handle Git errors with specific exception types."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except GitCommandError as e:
            error_output = str(e).lower()
            
            # Check for merge conflicts
            if "conflict" in error_output or "merge conflict" in error_output:
                raise GitConflictError(f"Git merge conflict: {e}")
            
            # Check for authentication errors
            if "authentication" in error_output or "permission denied" in error_output:
                raise GitAuthenticationError(f"Git authentication error: {e}")
            
            # Check for network errors
            if "network" in error_output or "connection" in error_output:
                raise GitNetworkError(f"Git network error: {e}")
            
            # Default to repository error
            raise GitRepositoryError(f"Git command error: {e}")
        except GitCommandNotFound as e:
            raise GitRepositoryError(f"Git command not found: {e}")
    return wrapper

@functools.lru_cache(maxsize=1)
def get_gitea_username() -> str:
    """Fetch the actual username associated with GITEA_TOKEN."""
    if not GITEA_TOKEN:
        return GITEA_USER
    try:
        resp = httpx.get(f"{GITEA_URL}/api/v1/user", headers=_headers, timeout=10)
        resp.raise_for_status()
        return resp.json().get("username", GITEA_USER)
    except Exception as e:
        print(f"Warning: Could not fetch Gitea username, falling back to {GITEA_USER}: {e}")
        return GITEA_USER

def ensure_repo_exists() -> None:
    """Ensure the repo exists on Gitea, create it if not."""
    username = get_gitea_username()
    try:
        _api("GET", f"/repos/{username}/{GITEA_REPO}")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # Create it
            _api("POST", f"/user/repos", json={"name": GITEA_REPO, "private": False})

def initialize_gitea_repository() -> str:
    """
    Perform one-time initialization of the Gitea repository.
    Creates the repo if missing, and pushes existing workspace content if .git is missing.
    Raises RuntimeError on critical failures (e.g., push conflicts).
    """
    ensure_repo_exists()
    
    username = get_gitea_username()
    auth_url = GITEA_URL.replace("http://", f"http://{username}:{GITEA_TOKEN}@").replace("https://", f"https://{username}:{GITEA_TOKEN}@")
    remote_url = f"{auth_url}/{username}/{GITEA_REPO}.git"

    repo_path = WORKSPACE
    dot_git = repo_path / ".git"
    
    is_new_init = False
    if not dot_git.exists():
        if any(repo_path.iterdir()):
            print(f"-- No .git found but files exist in {repo_path}. Initializing...")
            repo = Repo.init(repo_path)
            repo.create_remote("origin", remote_url)
            is_new_init = True
            # Ensure branch is main
            if "main" not in [b.name for b in repo.branches]:
                repo.git.checkout("-b", "main")
        else:
            # Empty workspace, safe to clone
            import subprocess
            subprocess.run(["git", "clone", remote_url, "."], cwd=repo_path, check=True)
            return "Workspace initialized by cloning empty repo."
    
    repo = _repo()
    
    # Ensure remote is correct
    if "origin" not in [r.name for r in repo.remotes]:
        repo.create_remote("origin", remote_url)
    else:
        repo.remotes.origin.set_url(remote_url)

    # Initial backup push
    if repo.is_dirty(untracked_files=True) or is_new_init:
        print("-- Performing initial workspace backup to Gitea...")
        if repo.is_dirty(untracked_files=True):
            repo.git.add("-A")
            repo.git.commit("-m", "bootstrap: automated initial backup of local workspace content")
        
        try:
            current_branch = repo.active_branch.name
            repo.git.push("origin", current_branch, "--set-upstream")
            return f"Initial sync complete. Workspace pushed to Gitea on '{current_branch}'."
        except Exception as e:
            error_msg = f"CRITICAL: Failed to push initial content to Gitea: {e}"
            if is_new_init:
                error_msg += "\nNOTE: This might be because the Gitea volume was wiped (e.g., by 'make reboot') while the local workspace still has files. You may need to manually resolve this or use 'git push --force' if you are sure."
            raise RuntimeError(error_msg)

    return "Workspace already initialized and in sync with Gitea."

def sync_workspace() -> str:
    """
    Safe per-step synchronization.
    Always commits local changes; pulls from Gitea only on the first call per
    pipeline run. Subsequent calls skip the pull because agents run sequentially
    on a shared workspace volume — each agent pushes its work, so the next agent
    does not need to re-pull.
    Assumes initialize_gitea_repository() was called at startup.
    """
    global _workspace_pulled_this_session
    repo = _repo()

    # 1. Commit any local work from previous agent steps
    if repo.is_dirty(untracked_files=True):
        print(f"-- Sync: Committing local changes in {WORKSPACE}...")
        repo.git.add("-A")
        try:
            repo.git.commit("-m", "automation: sync point before pull")
        except GitConflictError as e:
            print(f"-- Sync Warning: Merge conflict during commit: {e}")
            # Try to resolve by aborting and continuing
            try:
                repo.git.merge("--abort")
            except Exception:
                pass
        except GitAuthenticationError as e:
            print(f"-- Sync Error: Authentication failure: {e}")
            raise
        except Exception as e:
            print(f"-- Sync Warning: Could not commit local changes: {e}")

    # 2. Pull only once per pipeline run — workspace volume is shared between
    #    sequential agents, so after the first pull the workspace is already current.
    if _workspace_pulled_this_session:
        print("-- Sync: Skipping remote pull (already pulled this pipeline run)")
        final_branch = repo.active_branch.name
        files = [f for f in os.listdir(WORKSPACE) if not f.startswith(".")]
        return f"Workspace synced (pull skipped — already pulled this run) on branch '{final_branch}'. Files: {files}"

    # 3. Sync with Gitea main branch
    try:
        # Switch to main branch
        current_branch = repo.active_branch.name
        if current_branch != "main":
            print(f"-- Sync: Switching from '{current_branch}' to 'main' for sync.")
            if "main" not in [b.name for b in repo.branches]:
                repo.git.checkout("-b", "main")
            else:
                repo.git.checkout("main")
        
        # Pull latest changes (e.g., user manual commits)
        print("-- Sync: Fetching from origin...")
        repo.remotes.origin.fetch()
        
        print("-- Sync: Pulling from origin main...")
        try:
            # Use --no-rebase to avoid complicated states, but handle failures
            repo.git.pull("origin", "main", "--no-rebase")
        except GitConflictError as e:
            print(f"-- Sync CRITICAL: Merge conflicts detected! Attempting to abort...")
            repo.git.merge("--abort")
            raise GitConflictError(f"Merge conflict during sync: {e}")
        except GitAuthenticationError as e:
            print(f"-- Sync Error: Authentication failure during pull: {e}")
            raise
        except GitNetworkError as e:
            print(f"-- Sync Warning: Network error during pull: {e}. Retrying...")
            # Will be handled by retry decorator if used
            raise
        except Exception as e:
            print(f"-- Sync Warning: Pull failed: {e}. Checking for conflicts...")
            if repo.git.execute(["git", "ls-files", "-u"]):
                print("-- Sync CRITICAL: Merge conflicts detected! Attempting to abort...")
                repo.git.merge("--abort")
            else:
                print("-- Sync: Pull failed but no conflicts detected (maybe already up to date or unrelated history).")

    except (GitConflictError, GitAuthenticationError, GitNetworkError):
        raise
    except Exception as e:
        print(f"Warning: Could not sync with Gitea main: {e}")

    _workspace_pulled_this_session = True
    final_branch = repo.active_branch.name
    files = [f for f in os.listdir(WORKSPACE) if not f.startswith(".")]
    return f"Workspace synced on branch '{final_branch}' at {datetime.now().strftime('%H:%M:%S')}. Files: {files}"


def _api(method: str, path: str, **kwargs) -> dict:
    url = f"{GITEA_URL}/api/v1{path}"
    resp = httpx.request(method, url, headers=_headers, timeout=30, **kwargs)
    resp.raise_for_status()
    try:
        return resp.json() if resp.content else {}
    except ValueError:
        return {}


def _repo() -> Repo:
    # Fix dubious ownership issue
    import subprocess
    subprocess.run(["git", "config", "--global", "--add", "safe.directory", str(WORKSPACE)], check=False)
    # Ensure identity is set
    subprocess.run(["git", "config", "--global", "user.name", "Agent Bot"], check=False)
    subprocess.run(["git", "config", "--global", "user.email", "agent@local"], check=False)
    
    try:
        return Repo(WORKSPACE)
    except Exception:
        repo = Repo.init(WORKSPACE)
        return repo


# Tools

@tool("Git commit all changes")
def git_commit(message: str, branch: str | None = None) -> str:
    """
    Stage all changes in /workspace and create a commit.
    If branch is provided, creates and checks out that branch first.
    Uses git commit -a for faster commits by skipping the staging area.

    Args:
        message: commit message (use conventional commits: feat:, fix:, test:)
        branch: optional branch name to create before committing
    """
    agent = os.getenv("CREWAI_CURRENT_AGENT", "Unknown")
    prefix = f"[Agent: {agent}]"
    repo = _repo()

    if branch:
        # Check if we are on an unborn branch (new repo)
        if not repo.head.is_valid():
            repo.git.checkout("-b", branch)
        elif branch not in [b.name for b in repo.branches]:
            repo.git.checkout("-b", branch)
        else:
            repo.git.checkout(branch)

    # Use git commit -a for faster commits (auto-stages tracked files)
    try:
        # First add any new files, then commit all changes
        repo.git.add("-A")
        commit = repo.index.commit(message)
        branches = [b.name for b in repo.branches]
        return f"{prefix} SUCCESS: Committed {commit.hexsha[:8]} on branch '{repo.active_branch.name}'. All branches: {branches}"
    except GitConflictError as e:
        return f"{prefix} ERROR: Merge conflict - {e}"
    except GitAuthenticationError as e:
        return f"{prefix} ERROR: Authentication failure - {e}"
    except GitRepositoryError as e:
        return f"{prefix} ERROR: Repository error - {e}"
    except Exception as e:
        return f"{prefix} ERROR committing: {e}. Are there any changes to commit? Use list_files() to check."


@tool("Git push to Gitea")
@retry_git_operation(
    max_retries=GIT_MAX_RETRIES,
    initial_backoff=GIT_INITIAL_BACKOFF,
    max_backoff=GIT_MAX_BACKOFF,
    jitter=GIT_JITTER
)
def git_push(branch: str | None = None) -> str:
    """
    Push the current branch (or named branch) to the Gitea remote.
    Adds the remote automatically if not configured.
    Uses exponential backoff with jitter for network resilience.
    
    Args:
        branch: branch to push (default: current branch)
        
    Raises:
        GitAuthenticationError: If authentication fails
        GitNetworkError: If network operations fail
        GitConflictError: If push is rejected due to conflicts
    """
    agent = os.getenv("CREWAI_CURRENT_AGENT", "Unknown")
    prefix = f"[Agent: {agent}]"
    
    # Ensure the repo exists on Gitea
    try:
        _api("GET", f"/repos/{GITEA_USER}/{GITEA_REPO}")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # Create it
            _api("POST", f"/user/repos", json={"name": GITEA_REPO, "private": False})
        else:
            raise GitNetworkError(f"Failed to check repo existence: {e}")
    except httpx.RequestError as e:
        raise GitNetworkError(f"Network error checking repo: {e}")

    repo = _repo()
    branch = branch or repo.active_branch.name
    username = get_gitea_username()

    # inject token into URL for HTTP auth
    auth_url = GITEA_URL.replace("http://", f"http://{GITEA_TOKEN}@").replace("https://", f"https://{GITEA_TOKEN}@")
    remote_url = f"{auth_url}/{username}/{GITEA_REPO}.git"
    
    if "origin" not in [r.name for r in repo.remotes]:
        repo.create_remote("origin", remote_url)
    else:
        repo.remotes.origin.set_url(remote_url)

    try:
        repo.git.push("origin", branch, "--set-upstream", "--force-with-lease")
        return f"{prefix} Pushed branch '{branch}' to {GITEA_URL}/{username}/{GITEA_REPO}.git"
    except GitCommandError as e:
        error_str = str(e).lower()
        if "rejected" in error_str or "non-fast-forward" in error_str:
            raise GitConflictError(f"Push rejected - branch may have diverged: {e}")
        elif "authentication" in error_str or "permission" in error_str:
            raise GitAuthenticationError(f"Push authentication failed: {e}")
        elif "network" in error_str or "connection" in error_str:
            raise GitNetworkError(f"Network error during push: {e}")
        else:
            raise GitRepositoryError(f"Git push error: {e}")


@tool("Open a pull request on Gitea")
def open_pull_request(
    title: str,
    body: str,
    head_branch: str,
    base_branch: str = "main",
    labels: list[str] | None = None,
) -> str:
    """
    Open a pull request on the Gitea repo.
    Should be called AFTER git_push().

    Args:
        title: PR title (be descriptive)
        body: PR description — include what changed and why
        head_branch: the feature branch with new code
        base_branch: target branch (usually 'main')
        labels: optional list of label names (e.g. ['needs-review', 'agent-generated'])
    """
    agent = os.getenv("CREWAI_CURRENT_AGENT", "Unknown")
    prefix = f"[Agent: {agent}]"
    
    payload = {
        "title": title,
        "body": body,
        "head": head_branch,
        "base": base_branch,
    }

    username = get_gitea_username()
    pr = _api("POST", f"/repos/{username}/{GITEA_REPO}/pulls", json=payload)
    pr_url = pr.get("html_url", "unknown")
    pr_number = pr.get("number", "?")

    # Add labels if requested
    if labels:
        try:
            _api(
                "POST",
                f"/repos/{username}/{GITEA_REPO}/issues/{pr_number}/labels",
                json={"labels": labels},
            )
        except Exception:  # noqa: BLE001
            pass  # labels are optional

    return f"{prefix} PR #{pr_number} opened: {pr_url}"


@tool("Add review comment to pull request")
def add_pr_review(pr_number: int, body: str, approve: bool = False) -> str:
    """
    Post a review on an open pull request.
    The reviewer agent uses this to surface BASSPC findings.

    Args:
        pr_number: the PR number to review
        body: review body (markdown supported — use checklists for BASSPC)
        approve: if True, approves the PR; if False, requests changes
    """
    agent = os.getenv("CREWAI_CURRENT_AGENT", "Unknown")
    prefix = f"[Agent: {agent}]"
    
    state = "APPROVED" if approve else "REQUEST_CHANGES"
    username = get_gitea_username()
    _api(
        "POST",
        f"/repos/{username}/{GITEA_REPO}/pulls/{pr_number}/reviews",
        json={"body": body, "event": state},
    )
    verdict = "✓ Approved" if approve else "✗ Changes requested"
    return f"{prefix} {verdict} on PR #{pr_number}"


@tool("List open pull requests")
def list_pull_requests(state: str = "open") -> str:
    """
    List pull requests on the Gitea repo.
    Args:
        state: 'open', 'closed', or 'all'
    """
    agent = os.getenv("CREWAI_CURRENT_AGENT", "Unknown")
    prefix = f"[Agent: {agent}]"
    username = get_gitea_username()
    prs = _api("GET", f"/repos/{username}/{GITEA_REPO}/pulls", params={"state": state})
    if not prs:
        return f"{prefix} No {state} pull requests."
    lines = [f"PR #{p['number']}: {p['title']} [{p['state']}]" for p in prs]
    return f"{prefix} Listings:\n" + "\n".join(lines)


@tool("Get git diff for review")
def git_diff(base: str = "main", head: str | None = None) -> str:
    """
    Get the diff between two branches. Used by the reviewer agent to
    inspect what changed before running BASSPC checks.

    Args:
        base: base branch (default: main)
        head: head branch (default: current branch)
    """
    repo = _repo()
    head = head or repo.active_branch.name
    
    # Empty tree hash (Git's universal "null" state)
    EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

    # Check if base exists
    try:
        repo.git.rev_parse(base)
        has_base = True
    except Exception:
        has_base = False

    # Check if head exists
    try:
        repo.git.rev_parse(head)
        has_head = True
    except Exception:
        has_head = False
        existing = [b.name for b in repo.branches]
        return f"Error: head branch '{head}' NOT found. Existing branches: {existing}. Did you forget to commit/branch?"

    try:
        if has_base:
            # Try a triple-dot diff (since common ancestor)
            diff = repo.git.diff(f"{base}...{head}", "--stat")
            full = repo.git.diff(f"{base}...{head}")
        else:
            # Fallback to diff against empty tree (shows full content)
            diff = repo.git.diff(f"{EMPTY_TREE}..{head}", "--stat")
            full = repo.git.diff(f"{EMPTY_TREE}..{head}")
    except Exception as e:  # noqa: BLE001
        try:
            # Fallback to double-dot (direct diff)
            if has_base:
                diff = repo.git.diff(f"{base}..{head}", "--stat")
                full = repo.git.diff(f"{base}..{head}")
            else:
                return f"Error executing tool: {e}"
        except Exception as e:  # noqa: BLE001
            return f"Error executing tool: {e}"

    agent = os.getenv("CREWAI_CURRENT_AGENT", "Unknown")
    prefix = f"[Agent: {agent}]"
    
    # Truncate very large diffs
    if len(full) > 8000:
        full = full[:8000] + "\n... (truncated, use read_file for full content)"
    return f"{prefix} === STAT ===\n{diff}\n\n=== DIFF ===\n{full}"


# ============================================================================
# Branch Management Optimizations
# ============================================================================

def get_branch_prefix() -> str:
    """Get the branch prefix for agent branches."""
    return os.getenv("AGENT_BRANCH_PREFIX", "agent/")

def find_existing_branch_for_feature(feature_request: str) -> str | None:
    """
    Find an existing branch for a similar feature request to enable branch reuse.
    Uses simple string matching on feature keywords.
    
    Args:
        feature_request: The feature request to search for
        
    Returns:
        Branch name if found, None otherwise
    """
    repo = _repo()
    prefix = get_branch_prefix()
    
    # Extract keywords from feature request
    keywords = [w.lower() for w in feature_request.split() if len(w) > 3]
    
    for branch in repo.branches:
        branch_name = branch.name
        if not branch_name.startswith(prefix):
            continue
            
        # Check if branch name or recent commits mention similar keywords
        branch_keywords = [w.lower() for w in branch_name.split() if len(w) > 3]
        matching_keywords = len(set(keywords) & set(branch_keywords))
        
        if matching_keywords >= 2:  # At least 2 matching keywords
            return branch_name
    
    return None

def cleanup_merged_branches() -> str:
    """
    Clean up merged branches to keep git history clean.
    Removes local and remote branches that have been merged.
    
    Returns:
        Summary of cleanup actions
    """
    repo = _repo()
    prefix = get_branch_prefix()
    cleaned = []
    
    # Get current branch
    current_branch = repo.active_branch.name
    
    # Clean up local merged branches
    try:
        merged_branches = repo.git.branch("--merged", current_branch).splitlines()
        for branch in merged_branches:
            branch = branch.strip()
            if branch.startswith(prefix) and branch != current_branch:
                try:
                    repo.git.branch("-d", branch)
                    cleaned.append(f"Local: {branch}")
                except Exception:  # noqa: BLE001
                    # Branch might not be fully merged, skip
                    pass
    except Exception:  # noqa: BLE001
        pass
    
    # Clean up remote merged branches
    try:
        repo.remotes.origin.fetch(prune=True)
        # Note: Gitea doesn't support --merged for remote branches easily
        # We'll just list them for manual cleanup
        remote_branches = [b.name for b in repo.remotes.origin.refs]
        for branch in remote_branches:
            if branch.startswith(prefix):
                # Check if locally merged
                local_branch_name = branch.replace("origin/", "")
                if local_branch_name in [b.name for b in repo.branches]:
                    local_branch = repo.branches[local_branch_name]
                    if local_branch.is_merged_to(current_branch):
                        cleaned.append(f"Remote: {branch} (marked for cleanup)")
    except Exception:  # noqa: BLE001
        pass
    
    if cleaned:
        return f"Cleaned up merged branches:\n" + "\n".join(f"  - {b}" for b in cleaned)
    return "No merged branches to clean up"

def create_optimized_branch_name(feature_request: str, loop_index: int) -> str:
    """
    Create an optimized branch name that enables reuse.
    Uses a hash of the feature request to identify similar features.
    
    Args:
        feature_request: The feature request
        loop_index: Current loop index
        
    Returns:
        Optimized branch name
    """
    import hashlib
    
    # Create a hash-based identifier for the feature
    feature_hash = hashlib.md5(feature_request.encode()).hexdigest()[:8]
    
    # Extract key words for readability
    keywords = [w.lower() for w in feature_request.split() if len(w) > 3][:3]
    keyword_part = "-".join(keywords) if keywords else "feature"
    
    # Format: agent/{keyword}-{hash}-loop{loop_index}
    branch_name = f"{get_branch_prefix()}{keyword_part}-{feature_hash}-loop{loop_index}"
    
    return branch_name

def get_branch_stats() -> dict:
    """
    Get statistics about branches for monitoring.
    
    Returns:
        Dictionary with branch statistics
    """
    repo = _repo()
    prefix = get_branch_prefix()
    
    all_branches = [b.name for b in repo.branches]
    agent_branches = [b for b in all_branches if b.startswith(prefix)]
    
    return {
        "total_branches": len(all_branches),
        "agent_branches": len(agent_branches),
        "current_branch": repo.active_branch.name,
        "agent_branch_prefix": prefix,
    }
