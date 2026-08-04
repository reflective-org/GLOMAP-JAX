# ---------------------------------------------------------------------------
# GLOMAP-JAX
#
#   make venv       create .venv and install with the dev extra
#   make test       pytest -m "not slow"  -- must be green at EVERY commit
#   make test-all   full suite including slow/soak
#   make lint       ruff check + format --check
#   make fmt        ruff format
#   make docs       mkdocs build --strict
#   make goldens    regenerate Fortran reference fixtures (NEVER run in CI)
#   make bench      throughput curves
#   make clean      remove build products
# ---------------------------------------------------------------------------

VENV   ?= .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF   := $(VENV)/bin/ruff
MKDOCS := $(VENV)/bin/mkdocs

.PHONY: venv test test-all lint fmt docs goldens bench clean

venv:
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev]"
	@echo "installed into $(VENV)"

test:
	$(PYTEST) -q -m "not slow"

test-all:
	$(PYTEST) -q

lint:
	$(RUFF) check src tests validation scripts benchmarks
	$(RUFF) format --check src tests validation scripts benchmarks

fmt:
	$(RUFF) format src tests validation scripts benchmarks
	$(RUFF) check --fix src tests validation scripts benchmarks

docs:
	$(MKDOCS) build --strict

# Goldens are generated ONCE on a pinned toolchain and committed. They are not
# bit-reproducible across platforms: the vendored Makefile leaves
# -ffp-contract at gfortran's default, and EXP/LOG/ERF come from the platform
# libm. So this target is deliberately never wired into CI -- see
# docs/REFERENCE_BUILD.md.
goldens:
	$(PY) scripts/capture_reference.py --all

bench:
	$(PY) benchmarks/bench_throughput.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
