# =============================================================================
# agent-exec — sandboxed code execution image
# No network at runtime (enforced by docker run --network=none)
# Runs as non-root user 'agent' (uid 1000)
# =============================================================================
FROM python:3.12-slim

# System deps for typical projects (no network at runtime so install everything now)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    make \
    gcc \
    g++ \
    nodejs \
    npm \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Common Python packages agents will need
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    pytest-mock \
    black \
    ruff \
    mypy \
    httpx \
    pydantic \
    fastapi \
    sqlalchemy \
    alembic \
    rich

# Create non-root agent user
RUN useradd -m -u 1000 -s /bin/bash agent

# Drop to non-root
USER agent
RUN git config --global user.name "DGX Agent" && \
    git config --global user.email "agent@dgx.local" && \
    git config --global --add safe.directory /workspace

WORKDIR /workspace

# Default: just exec whatever is passed
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["echo 'No command specified'"]
