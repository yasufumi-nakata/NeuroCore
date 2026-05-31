PYTHON ?= $(shell \
	if [ -x .venv/bin/python ]; then printf '%s' .venv/bin/python; \
	elif command -v python3.14 >/dev/null 2>&1; then printf '%s' python3.14; \
	elif command -v python3.13 >/dev/null 2>&1; then printf '%s' python3.13; \
	elif command -v python3.12 >/dev/null 2>&1; then printf '%s' python3.12; \
	elif command -v python3.11 >/dev/null 2>&1; then printf '%s' python3.11; \
	elif command -v python3.10 >/dev/null 2>&1; then printf '%s' python3.10; \
	else printf '%s' python3; fi)
NPM ?= npm

.PHONY: test self-test lint package packages pages oss-health frontend-install frontend-build

test:
	$(PYTHON) -m pytest

self-test:
	PYTHONPATH=backend $(PYTHON) -m neurocore.cli self-test --json

lint:
	$(PYTHON) -m ruff check backend scripts

package:
	$(PYTHON) -m build

packages:
	$(PYTHON) scripts/build_package_artifacts.py

pages:
	$(PYTHON) scripts/build_pages.py
	$(PYTHON) scripts/validate_pages.py

oss-health:
	$(PYTHON) scripts/oss_health_check.py

frontend-install:
	cd frontend && $(NPM) install

frontend-build:
	cd frontend && $(NPM) run build
