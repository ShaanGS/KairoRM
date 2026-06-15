# KairoRM — common tasks. Run `make help` to list them.
# These call the project virtualenv directly, so you don't need to activate it.

VENV := .venv/bin
SRC  ?= .

.PHONY: help run test test-unit lint fmt

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

run:  ## Analyse a repo and launch the report server (override target: make run SRC=<url|path>)
	$(VENV)/kairo map $(SRC)

test:  ## Run the full test suite
	$(VENV)/pytest

test-unit:  ## Run unit tests only
	$(VENV)/pytest tests/unit/

lint:  ## Lint with ruff
	$(VENV)/ruff check .

fmt:  ## Auto-format with ruff
	$(VENV)/ruff format .
