help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

PROJECT_NAME=pyidk
MAX_LINE_LEN=100
TEST_DIR=tests
RELEASE_DIR=dist
TEST_MIN_COVERAGE=75.0
TEST_OUTPUT_DIR=reports/tests
COVERAGE_OUTPUT_DIR=reports/coverage

.PHONY: env
env: ## Create virtual environment
	python3 -m venv .venv
	. .venv/bin/activate && python -m pip install -e ".[dev]"

.PHONY: format
format: ## Run all code through formatting
	ruff format src tests examples
	ruff check --fix src tests examples

.PHONY: lint
lint: ## Run all code through Static Analysis
	ruff format --check src tests examples
	ruff check src tests examples
	mypy src tests examples

.PHONY: test
test: ## Run all tests.
	pytest -v -o junit_family=xunit1 \
			--basetemp=/tmp/pytest \
			--junitxml=$(TEST_OUTPUT_DIR)/xml/nosetests.xml \
			--html=$(TEST_OUTPUT_DIR)/html/index.html \
			--cov-report term-missing \
			--cov-fail-under $(TEST_MIN_COVERAGE) \
			--cov=$(PROJECT_NAME) $(TEST_DIR)/ \
			--cov-report xml:$(COVERAGE_OUTPUT_DIR)/xml/coverage.xml

.PHONY: release
release: ## Release the module.
	if [ -d $(RELEASE_DIR) ]; then rm -Rf $(RELEASE_DIR); fi
	python setup.py sdist bdist_wheel -d $(RELEASE_DIR)
