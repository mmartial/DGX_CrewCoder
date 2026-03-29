FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git rsync curl docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Get the actual docker socket GID at build time
ARG DOCKER_GID=999
RUN groupmod -g ${DOCKER_GID} docker || groupadd -g ${DOCKER_GID} docker

CMD ["python", "crew.py"]
