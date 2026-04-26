# In-Place Test-Time Training — common dev commands.
# Usage: make <target>   (run `make help` for the list)

PYTHON ?= uv run python
PYTEST ?= uv run pytest
UV     ?= uv

PKG          := models/hf_gemma3
MODEL_TESTS    := $(PKG)/test_gemma3.py
TRAIN_TESTS  := train/test_main.py
HF_REPO_ID  ?= yourname/gemma3-1b-ttt
CKPT_DIR    ?= checkpoints/gemma3-1b-ttt

.DEFAULT_GOAL := help
HF_USER     ?= yourname
DATASET     ?= tinystories

.PHONY: help install sync lock test test-slow test-watch train train-tinystories train-longalpaca eval clean push-hub login-hf format

help:
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## Install dependencies + RULER submodule + nltk data + PG essay haystack
	$(UV) sync --all-groups
	git submodule update --init --recursive
	$(UV) run python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)"
	@if [ ! -f third_party/RULER/scripts/data/synthetic/json/PaulGrahamEssays.json ]; then \
		cd third_party/RULER/scripts/data/synthetic/json && \
		$(UV) run python download_paulgraham_essay.py && \
		rm -rf essay_repo essay_html; \
	else \
		echo "PG essays already downloaded"; \
	fi

sync: ## Sync lockfile -> environment (no upgrades; skips submodule + data)
	$(UV) sync

lock: ## Resolve and write uv.lock
	$(UV) lock

test: ## Run fast test suite (skips @slow tests)
	$(PYTEST) $(MODEL_TESTS) $(TRAIN_TESTS) -v -m "not slow"

test-slow: ## Run @slow tests too (downloads real Gemma3-1B; needs HF auth)
	$(PYTEST) $(MODEL_TESTS) $(TRAIN_TESTS) -v -m slow

test-all: ## Run every test, fast and slow
	$(PYTEST) $(MODEL_TESTS) $(TRAIN_TESTS) -v

test-watch: ## Re-run fast tests on file changes (needs pytest-watch)
	$(UV) run ptw $(MODEL_TESTS) -- -v -m "not slow"

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
