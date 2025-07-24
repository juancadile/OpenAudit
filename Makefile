# OpenAudit Development Makefile
.PHONY: help install install-dev setup test test-verbose lint format type-check security clean run-bias run-web run-unified pre-commit-install pre-commit-run

# Auto-detect Python command (python3 preferred, fallback to python)
PYTHON := $(shell command -v python3 2> /dev/null || command -v python 2> /dev/null || echo "python")

# Default target
help: ## Show this help message
	@echo "OpenAudit Development Commands:"
	@echo "================================"
	@echo "🐍 Using Python: $(PYTHON)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation and setup
install: ## Install production dependencies
	@echo "🔧 Installing production dependencies..."
	$(PYTHON) -m pip install -r requirements.txt || $(PYTHON) -m pip install --user -r requirements.txt

install-dev: ## Install development dependencies
	@echo "🔧 Installing development dependencies..."
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt || \
	$(PYTHON) -m pip install --user -r requirements.txt -r requirements-dev.txt

setup: install-dev ## Complete development environment setup
	pre-commit install
	@echo "✅ Development environment setup complete!"
	@echo "📝 Don't forget to:"
	@echo "   1. Copy .env.example to .env and add your API keys"
	@echo "   2. Activate your virtual environment if you haven't already"

# Testing
test: ## Run tests with pytest
	$(PYTHON) -m pytest tests/ -v

test-verbose: ## Run tests with verbose output and coverage
	$(PYTHON) -m pytest tests/ -v --tb=long --cov=core --cov=utils --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage (faster)
	$(PYTHON) -m pytest tests/ -v --tb=short

test-watch: ## Run tests in watch mode (requires pytest-watch)
	ptw tests/ -- -v

# Code quality
lint: ## Run all linting checks
	flake8 .
	black --check .
	isort --check-only .

format: ## Format code with black and isort
	black .
	isort .
	@echo "✅ Code formatted successfully!"

type-check: ## Run type checking with mypy
	mypy core/ utils/ --ignore-missing-imports

security: ## Run security checks
	bandit -r . -ll -x tests/
	safety check

# Combined quality checks
check: lint type-check security test ## Run all quality checks and tests

# Pre-commit
pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Application commands
run-bias: ## Run bias testing framework
	$(PYTHON) core/bias_testing_framework.py

run-web: ## Run web interface (historical dashboard)
	$(PYTHON) utils/web_interface.py

run-unified: ## Run unified dashboard interface
	$(PYTHON) unified_interface.py

run-live: ## Run live experiment interface
	$(PYTHON) templates/archive/live_experiment.py

# Development utilities
clean: ## Clean up cache files and build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "✅ Cache files cleaned!"

clean-results: ## Clean experiment results (use with caution!)
	@echo "⚠️  This will delete all experiment results in runs/"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf runs/*.json runs/*.txt; \
		echo "🗑️  Experiment results cleaned!"; \
	else \
		echo "❌ Cancelled"; \
	fi

# Documentation
docs: ## Generate documentation (if sphinx is set up)
	@echo "📚 Documentation generation not yet implemented"
	@echo "   You can add sphinx documentation generation here"

# Environment management
check-python: ## Check Python installation and version
	@echo "🐍 Python Detection:"
	@echo "  Using: $(PYTHON)"
	@echo "  Version: $$($(PYTHON) --version 2>/dev/null || echo 'Not found')"
	@echo "  Location: $$(which $(PYTHON) 2>/dev/null || echo 'Not found')"
	@echo "  pip version: $$($(PYTHON) -m pip --version 2>/dev/null || echo 'pip not available')"

check-env: ## Check if .env file exists and has required keys
	@if [ ! -f .env ]; then \
		echo "❌ .env file not found. Copy .env.example to .env and add your API keys"; \
		exit 1; \
	fi
	@echo "✅ .env file exists"
	@grep -q "OPENAI_API_KEY" .env || echo "⚠️  OPENAI_API_KEY not found in .env"
	@grep -q "ANTHROPIC_API_KEY" .env || echo "⚠️  ANTHROPIC_API_KEY not found in .env"

# Quick start
quick-start: setup check-env ## Complete setup and environment check
	@echo "🚀 Quick start complete! You can now run:"
	@echo "   make run-unified    # Start the unified dashboard"
	@echo "   make test          # Run the test suite"
	@echo "   make run-bias      # Run bias testing framework"

# Development workflow
dev: format lint test ## Format, lint, and test (typical dev workflow)
	@echo "✅ Development workflow complete!"

# CI simulation
ci: install-dev lint type-check security test ## Simulate CI pipeline locally
	@echo "🔄 CI pipeline simulation complete!"