.PHONY: test self-test lint package packages pages oss-health frontend-install frontend-build

test:
	python -m pytest

self-test:
	PYTHONPATH=backend python -m neurocore.cli self-test --json

lint:
	ruff check backend scripts

package:
	python -m build

packages:
	python scripts/build_package_artifacts.py

pages:
	python scripts/build_pages.py
	python scripts/validate_pages.py

oss-health:
	python scripts/oss_health_check.py

frontend-install:
	cd frontend && npm install

frontend-build:
	cd frontend && npm run build
