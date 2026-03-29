# =============================================================================
# agent-quality — static analysis and vulnerability scanning image
# =============================================================================
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    gpg \
    && rm -rf /var/lib/apt/lists/*

# Semgrep (SAST)
RUN pip install --no-cache-dir semgrep bandit

# Trivy (SBOM + vuln scanning) — via official apt repo (works on aarch64)
RUN wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key \
    | gpg --dearmor -o /usr/share/keyrings/trivy-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/trivy-archive-keyring.gpg] \
    https://aquasecurity.github.io/trivy-repo/deb generic main" \
    > /etc/apt/sources.list.d/trivy.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends trivy \
 && rm -rf /var/lib/apt/lists/*

# detect-secrets (secrets scanning)
RUN pip install --no-cache-dir detect-secrets

RUN useradd -m -u 1000 agent
USER agent
RUN git config --global user.name "DGX Agent" && \
    git config --global user.email "agent@dgx.local" && \
    git config --global --add safe.directory /scan

WORKDIR /scan

ENTRYPOINT ["/bin/bash", "-c"]
