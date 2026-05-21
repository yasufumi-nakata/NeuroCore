.PHONY: test self-test lint package frontend-install frontend-build

test:
	python -m pytest

self-test:
	PYTHONPATH=backend python -m neurocore.cli self-test --json

lint:
	ruff check backend

package:
	python -m build

frontend-install:
	cd frontend && npm install

frontend-build:
	cd frontend && npm run build
