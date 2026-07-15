# Makefile — Legion / OpenManus
# SAFE-PORT quality-gate pattern (PROP-2026-0714-001, 2026-07-15)
# Adapted from Banxe pattern; NO FCA invariants, NO banxe-specific rules.
# All targets operate on openmanus_rl/ (Legion module layout).

.PHONY: lint fmt-check test test-cov quality-gate

lint:
	ruff check .

fmt-check:
	ruff format --check .

test:
	pytest -q

test-cov:
	pytest --cov=openmanus_rl --cov-report=term-missing --cov-fail-under=20 -q

# Advisory quality-gate: semgrep uses '|| true' so it never hard-fails this target.
# CI runs this with continue-on-error: true for the same reason.
quality-gate:
	ruff check .
	ruff format --check .
	semgrep --config auto . || true
	pytest --cov=openmanus_rl --cov-fail-under=20 -q
