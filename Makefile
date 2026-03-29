# =============================================================================
# Makefile — DGX Spark Multi-Agent Stack

ifneq (,$(wildcard .env))
	include .env
	export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' .env)
endif

SHELL := /bin/bash

.PHONY: help setup up down logs run build-images gvisor-install clean-volumes reboot

COMPOSE = docker compose

# Automatically detect UID/GID if not provided in .env
WANTED_UID := $(shell id -u)
WANTED_GID := $(shell id -g)
DOCKER_GID := $(shell stat -c '%g' /var/run/docker.sock)

# Internal autodetection for .env population
DETECTED_IP := $(shell hostname -I | awk '{print $$1}')
DETECTED_DATE := $(shell date +%Y%m%d-%H%M%S)

export WANTED_UID WANTED_GID DOCKER_GID

help:
	@echo ""
	@echo "  make setup          First-time setup (build images, init Gitea/MLflow)"
	@echo "  make reset-env      Force-update DGX_IP and CREW_DATE in .env"
	@echo "  make up             Start all services (Gitea, MLflow, agent-runners)"
	@echo "  make down           Stop all services"
	@echo "  make logs           Tail all service logs"
	@echo "  make build-images   Build exec + quality sandbox images"
	@echo "  make gvisor-install Install gVisor runtime on the DGX host"
	@echo ""

check_workspace:
	@mkdir -p workspace
	@touch workspace/.test || { echo "Unable to create workspace/.test. Check permissions."; exit 1; }
	@rm workspace/.test || { echo "Unable to remove workspace/.test. Check permissions."; exit 1; }
	@mkdir -p workspace/progress workspace/code
	@touch workspace/progress/.test || { echo "Unable to create workspace/progress/.test. Check permissions."; exit 1; }
	@rm workspace/progress/.test || { echo "Unable to remove workspace/progress/.test. Check permissions."; exit 1; }
	@touch workspace/code/.test || { echo "Unable to create workspace/code/.test. Check permissions."; exit 1; }
	@rm workspace/code/.test || { echo "Unable to remove workspace/code/.test. Check permissions."; exit 1; }

# fail is mandatory if environment variables are not set
check: check_workspace
	@for var in DGX_IP GITEA_PORT GITEA_SSH_PORT MLFLOW_PORT GITEA_USER GITEA_PASSWORD GITEA_EMAIL GITEA_REPO SANDBOX_RUNTIME SANDBOX_MEMORY SANDBOX_CPUS OLLAMA_MODEL RALPH_LOOP; do \
		if [[ -z "$${!var}" ]]; then echo "$$var not set"; exit 1; fi; \
	done

## First-time setup
setup: check_workspace build-images
	$(COMPOSE) up -d gitea mlflow
	@echo "Waiting for services to be ready..." && sleep 10
	@echo "Creating admin user for gitea (${GITEA_USER}:${GITEA_PASSWORD})"
	@docker exec -it -u git agent-gitea gitea admin user create \
		--username ${GITEA_USER} \
		--password ${GITEA_PASSWORD} \
		--email ${GITEA_EMAIL} \
		--admin || true
	@echo "Generating access token for ${GITEA_USER}..."
	@T=$$(docker exec -u git agent-gitea gitea admin user generate-access-token \
		--username ${GITEA_USER} \
		--token-name agent-bot-token \
		--scopes all \
		--raw 2>/dev/null || true); \
		if [[ -n "$$T" && ! "$$T" =~ "error" ]]; then \
			sed -i "s|^GITEA_TOKEN=.*|GITEA_TOKEN=$$T|" .env && \
			echo "✓ Gitea Access Token generated and saved to .env"; \
		else \
			echo "⚠ Gitea token already exists or could not be generated. Keeping current .env value."; \
		fi
	@echo ""
	@echo "│  Services starting..."
	@echo "│  Gitea:    http://${DGX_IP}:${GITEA_PORT}"
	@echo "│  MLflow:   http://${DGX_IP}:${MLFLOW_PORT}"
	@echo "|"
	@echo "│  Status:"
	@echo "│  - Gitea: Admin user created (${GITEA_USER}:${GITEA_PASSWORD})"
	@echo "│  - Gitea: API token auto-generated"
	@echo "│  - MLflow: Dashboard available"
	@echo "|"
	@echo "│  Next steps:"
	@echo "│  1. make up"
	@echo ""
	@echo "Check the .env files for alternate ports if needed"

## Force-update dynamic values in .env
dotenv:
	@cp --update=none .env.example .env && echo "✓ Copied .env.example → .env (if not exists)"
	@sed -i "s|^DGX_IP=.*|DGX_IP=${DETECTED_IP}|" .env
	@export DGX_IP=${DETECTED_IP}
	@sed -i "s|^GITEA_REPO=.*|GITEA_REPO=${DETECTED_DATE}|" .env
	@export GITEA_REPO=${DETECTED_DATE}
	@sed -i "s|^MLFLOW_EXPERIMENT_NAME=.*|MLFLOW_EXPERIMENT_NAME=${DETECTED_DATE}|" .env
	@export MLFLOW_EXPERIMENT_NAME=${DETECTED_DATE}
	@echo "✓ Force-updated .env with static values (IP: ${DETECTED_IP}, GITEA_REPO: ${DETECTED_DATE}, MLFLOW_EXPERIMENT_NAME: ${DETECTED_DATE})"

## Build sandbox Docker images
build-images:
	docker build -t agent-exec:latest docker/exec/
	docker build -t agent-quality:latest docker/quality/
	@echo "✓ Sandbox images built"

## Start build process; the code will be in gitea
up: check
	@echo "Starting services (UID: ${WANTED_UID}, GID: ${WANTED_GID}, DOCKER_GID: ${DOCKER_GID})..."
	DOCKER_GID=${DOCKER_GID} $(COMPOSE) build
	DOCKER_GID=${DOCKER_GID} $(COMPOSE) up -d --build

## Stop everything
down:
	$(COMPOSE) down

## Tail logs
logs:
	$(COMPOSE) logs -f

## Install gVisor on the DGX host (Ubuntu)
gvisor-install:
	@echo "Installing gVisor runtime..."
	curl -fsSL https://gvisor.dev/archive.key | sudo gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg
	echo "deb [arch=$$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" | sudo tee /etc/apt/sources.list.d/gvisor.list
	sudo apt-get update && sudo apt-get install -y runsc
	sudo runsc install
	sudo systemctl restart docker
	@echo "✓ gVisor installed. Test with: docker run --runtime=runsc hello-world"

## Verify gVisor is working
gvisor-test:
	docker run --runtime=runsc --rm alpine uname -r
	docker run --runtime=runsc hello-world

## Clean up all volumes
clean-volumes:
	$(COMPOSE) down -v
	@echo "✓ All volumes cleaned"

## Complete reset (Clean, Setup, Up)
reset: check_workspace
	$(MAKE) clean-volumes
	$(MAKE) setup
	$(MAKE) up
