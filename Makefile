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
	pytest --cov=openmanus_rl --cov-report=term-missing --cov-fail-under=35 -q

# Advisory quality-gate: semgrep uses '|| true' so it never hard-fails this target.
# Coverage floor is 35% (actual ~42%, ~7pp buffer). CI test step is blocking.
quality-gate:
	ruff check .
	ruff format --check .
	semgrep --config auto . || true
	pytest --cov=openmanus_rl --cov-fail-under=35 -q
