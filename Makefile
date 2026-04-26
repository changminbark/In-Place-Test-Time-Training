# In-Place Test-Time Training — common dev commands.
# Usage: make <target>   (run `make help` for the list)

PYTHON ?= uv run python
PYTEST ?= uv run pytest
UV     ?= uv

PKG          := models/hf_gemma3
TEST_FILE    := $(PKG)/test_gemma3.py
HF_REPO_ID  ?= yourname/gemma3-1b-ttt
CKPT_DIR    ?= checkpoints/gemma3-1b-ttt

.DEFAULT_GOAL := help
HF_USER     ?= yourname
DATASET     ?= tinystories

.PHONY: help install sync lock test test-slow test-watch train train-tinystories train-longalpaca eval clean push-hub login-hf format

help:
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## Install runtime + dev dependencies via uv
	$(UV) sync --all-groups

sync: ## Sync lockfile -> environment (no upgrades)
	$(UV) sync

lock: ## Resolve and write uv.lock
	$(UV) lock

test: ## Run fast test suite (skips @slow tests)
	$(PYTEST) $(TEST_FILE) -v -m "not slow"

test-slow: ## Run @slow tests too (downloads real Gemma3-1B; needs HF auth)
	$(PYTEST) $(TEST_FILE) -v -m slow

test-all: ## Run every test, fast and slow
	$(PYTEST) $(TEST_FILE) -v

test-watch: ## Re-run fast tests on file changes (needs pytest-watch)
	$(UV) run ptw $(TEST_FILE) -- -v -m "not slow"

train: ## Train on $(DATASET) (override DATASET=tinystories|longalpaca, HF_USER=...)
	$(PYTHON) -m train.main --dataset $(DATASET) --hf-user $(HF_USER)

train-tinystories: ## Train + push the TinyStories adapter (HF_USER=...)
	$(PYTHON) -m train.main --dataset tinystories --hf-user $(HF_USER)

train-longalpaca: ## Train + push the LongAlpaca adapter (HF_USER=...)
	$(PYTHON) -m train.main --dataset longalpaca --hf-user $(HF_USER)

eval: ## Run RULER evaluation
	$(PYTHON) eval/run_ruler.py

login-hf: ## Authenticate with HuggingFace Hub
	$(UV) run huggingface-cli login

push-hub: ## Upload $(CKPT_DIR) to $(HF_REPO_ID) (override on the cmdline)
	@if [ ! -d "$(CKPT_DIR)" ]; then \
		echo "checkpoint dir $(CKPT_DIR) not found"; exit 1; \
	fi
	$(UV) run huggingface-cli upload $(HF_REPO_ID) $(CKPT_DIR) . --repo-type model

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .ruff_cache .mypy_cache build dist *.egg-info
