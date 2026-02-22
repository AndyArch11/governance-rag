.PHONY: help install install-dev clean test test-cov lint format check-format type-check quality \
        ingest ingest-reset ingest-dry-run ingest-profile ingest-purge \
        ingest-bitbucket ingest-bitbucket-dry-run \
        query query-purge graph graph-rebuild graph-purge dashboard reset-db setup-dirs

# Default target
.DEFAULT_GOAL := help

# Python executable (prefer local virtual environment when available)
VENV_PYTHON := .venv/bin/python
PYTHON := $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),python3)
PIP := $(PYTHON) -m pip

# Project directories
SCRIPTS_DIR := scripts
TEST_DIR := tests
DATA_RAW := data_raw
LOGS_DIR := logs
MODELS_DIR := models
NOTEBOOKS_DIR := notebooks

# ChromaDB and cache paths (from .env defaults)
RAG_DATA_PATH := rag_data

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install ## Install development dependencies (includes testing tools)
	$(PIP) install -e ".[dev]"

setup-dirs: ## Create required project directories
	@mkdir -p $(DATA_RAW)/downloads
	@mkdir -p $(DATA_RAW)/url_imports
	@mkdir -p $(DATA_RAW)/public_docs
	@mkdir -p $(LOGS_DIR)
	@mkdir -p $(MODELS_DIR)
	@mkdir -p $(NOTEBOOKS_DIR)
	@mkdir -p $(SCRIPTS_DIR)/ingest
	@mkdir -p $(SCRIPTS_DIR)/rag
	@mkdir -p $(SCRIPTS_DIR)/consistency_graph
	@mkdir -p $(SCRIPTS_DIR)/utils
	@mkdir -p $(TEST_DIR)
	@mkdir -p $(RAG_DATA_PATH)
	@mkdir -p $(RAG_DATA_PATH)/chromadb
	@mkdir -p $(RAG_DATA_PATH)/cache
	@mkdir -p $(RAG_DATA_PATH)/consistency_graphs
	@echo "✓ Project directories created"

clean: ## Remove Python cache files and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist .coverage htmlcov .mypy_cache .tox
	@echo "✓ Cleaned cache and build artifacts"

test: ## Run tests
	$(PYTHON) -m pytest $(TEST_DIR) -v

test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest $(TEST_DIR) --cov=$(SCRIPTS_DIR) --cov-report=html --cov-report=term

lint: ## Run pylint checks
	$(PYTHON) -m pylint $(SCRIPTS_DIR) --disable=C0114,C0103,R0913,R0914,C0301,C0303 --max-line-length=100 --fail-under=8

type-check: ## Run mypy type checking
	$(PYTHON) -m mypy $(SCRIPTS_DIR) --ignore-missing-imports

format: ## Format code with black and isort
	$(PYTHON) -m black $(SCRIPTS_DIR) $(TEST_DIR) --line-length 100
	$(PYTHON) -m isort $(SCRIPTS_DIR) $(TEST_DIR) --line-length 100

check-format: ## Check code formatting without modifying
	$(PYTHON) -m black --check $(SCRIPTS_DIR) $(TEST_DIR) --line-length 100
	$(PYTHON) -m isort --check $(SCRIPTS_DIR) $(TEST_DIR) --line-length 100

quality: format lint type-check test ## Run all quality checks (format, lint, type-check, test)
	@echo "✓ All quality checks passed"

# ─────────────────────────────────────────────────────────────
# RAG Pipeline Commands
# ─────────────────────────────────────────────────────────────

ingest: ## Run document ingestion pipeline
	cd $(SCRIPTS_DIR)/ingest && $(PYTHON) ingest.py

ingest-reset: ## Reset ChromaDB collections and re-ingest all documents
	cd $(SCRIPTS_DIR)/ingest && $(PYTHON) ingest.py --reset

ingest-dry-run: ## Simulate ingestion without writing to ChromaDB
	cd $(SCRIPTS_DIR)/ingest && $(PYTHON) ingest.py --dry-run --verbose

ingest-profile: ## Quick validation run with detailed timing analysis
	cd $(SCRIPTS_DIR)/ingest && $(PYTHON) ingest.py --profile

ingest-purge: ## Run ingestion with log purge (Dev/Test only)
	cd $(SCRIPTS_DIR)/ingest && $(PYTHON) ingest.py --purge-logs

ingest-bitbucket: ## Ingest BitBucket repository (pass HOST, PROJECT, REPO, [USERNAME], [PASSWORD])
	@if [ -z "$(HOST)" ] || [ -z "$(PROJECT)" ] || [ -z "$(REPO)" ]; then \
		echo "Usage: make ingest-bitbucket HOST=<url> PROJECT=<key> REPO=<slug> [USERNAME=<user>] [PASSWORD=<pass>] [WORKERS=<n>] [LIMIT=<n>]"; \
		exit 1; \
	fi
	cd $(SCRIPTS_DIR)/ingest && $(PYTHON) ingest_bitbucket.py \
		--host $(HOST) \
		--project $(PROJECT) \
		--repo $(REPO) \
		$(if $(USERNAME),--username $(USERNAME)) \
		$(if $(PASSWORD),--password $(PASSWORD)) \
		$(if $(WORKERS),--workers $(WORKERS)) \
		$(if $(LIMIT),--limit $(LIMIT))

ingest-bitbucket-dry-run: ## Preview BitBucket ingestion (pass HOST, PROJECT, REPO, [USERNAME], [PASSWORD])
	@if [ -z "$(HOST)" ] || [ -z "$(PROJECT)" ] || [ -z "$(REPO)" ]; then \
		echo "Usage: make ingest-bitbucket-dry-run HOST=<url> PROJECT=<key> REPO=<slug> [USERNAME=<user>] [PASSWORD=<pass>]"; \
		exit 1; \
	fi
	cd $(SCRIPTS_DIR)/ingest && $(PYTHON) ingest_bitbucket.py \
		--host $(HOST) \
		--project $(PROJECT) \
		--repo $(REPO) \
		$(if $(USERNAME),--username $(USERNAME)) \
		$(if $(PASSWORD),--password $(PASSWORD)) \
		--dry-run --verbose

query: ## Run RAG query (pass QUERY="..." on command line)
	@if [ -z "$(QUERY)" ]; then \
		echo "Usage: make query QUERY=\"What are our security policies?\""; \
		exit 1; \
	fi
	cd $(SCRIPTS_DIR)/rag && $(PYTHON) query.py "$(QUERY)"

query-purge: ## Run RAG query with log purge (pass QUERY="...", Dev/Test only)
	@if [ -z "$(QUERY)" ]; then \
		echo "Usage: make query-purge QUERY=\"What are our security policies?\""; \
		exit 1; \
	fi
	cd $(SCRIPTS_DIR)/rag && $(PYTHON) query.py "$(QUERY)" --purge-logs

graph: ## Build consistency graph
	cd $(SCRIPTS_DIR)/consistency_graph && $(PYTHON) build_consistency_graph.py

graph-rebuild: ## Rebuild consistency graph with reset
	cd $(SCRIPTS_DIR)/consistency_graph && $(PYTHON) build_consistency_graph.py --reset

graph-purge: ## Build consistency graph with log purge (Dev/Test only)
	cd $(SCRIPTS_DIR)/consistency_graph && $(PYTHON) build_consistency_graph.py --purge-logs

dashboard: ## Launch Plotly Dash dashboard
	cd $(SCRIPTS_DIR)/ui && $(PYTHON) -m scripts.ui.dashboard

# ─────────────────────────────────────────────────────────────
# Database and Cache Management
# ─────────────────────────────────────────────────────────────

reset-db: ## Delete ChromaDB storage (WARNING: destroys all ingested data)
	@echo "⚠️  WARNING: This will delete all ChromaDB data at $(RAG_DATA_PATH)"
	@printf "Are you sure? [y/N] "; read -r REPLY; \
	if [ "$$REPLY" = "y" ] || [ "$$REPLY" = "Y" ]; then \
		rm -rf $(RAG_DATA_PATH); \
		echo "✓ ChromaDB storage deleted"; \
	else \
		echo "Cancelled"; \
	fi

clear-logs: ## Clear all log files
	find $(LOGS_DIR) -type f -name "*.log" -delete 2>/dev/null || true
	find $(SCRIPTS_DIR) -type f -name "*.log" -delete 2>/dev/null || true
	@echo "✓ Log files cleared"

purge-all-logs: ## Purge all .log and .jsonl files (ingest, rag, consistency_graph)
	find $(LOGS_DIR) -type f \( -name "*.log" -o -name "*.jsonl" \) -delete 2>/dev/null || true
	find $(SCRIPTS_DIR) -type f \( -name "*.log" -o -name "*.jsonl" \) -delete 2>/dev/null || true
	@echo "✓ All log and audit files purged"

# ─────────────────────────────────────────────────────────────
# Full Workflow Commands
# ─────────────────────────────────────────────────────────────

full-rebuild: reset-db ingest-reset graph-rebuild ## Complete rebuild: reset DB, ingest, rebuild graph
	@echo "✓ Full rebuild complete"

quick-start: setup-dirs install ingest graph dashboard ## Setup, install, ingest, build graph, and launch dashboard
	@echo "✓ Quick start complete"
