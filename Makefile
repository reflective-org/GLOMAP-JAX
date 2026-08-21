# ---------------------------------------------------------------------------
# GLOMAP-JAX
#
#   make venv       create .venv and install with the dev extra
#   make test       pytest -m "not slow"  -- must be green at EVERY commit
#   make test-all   full suite including slow/soak
#   make lint       ruff check + format --check
#   make fmt        ruff format
#   make docs       mkdocs build --strict
#   make capture    build the reference and re-run every capture script
#   make goldens    the above, then report what drifted (NEVER run in CI).
#                   Does NOT re-bless: it exits 1 if a golden moved.
#   make goldens-bless  record the regenerated fixtures in MANIFEST.json
#   make bench      throughput curves
#   make clean      remove build products
# ---------------------------------------------------------------------------

VENV   ?= .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF   := $(VENV)/bin/ruff
MKDOCS := $(VENV)/bin/mkdocs

.PHONY: venv test test-all lint fmt docs capture goldens goldens-bless bench clean

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
	$(RUFF) check .
	$(RUFF) format --check .

fmt:
	$(RUFF) format .
	$(RUFF) check --fix .

docs:
	$(MKDOCS) build --strict

# Goldens are generated ONCE on a pinned toolchain and committed. They are not
# bit-reproducible across platforms: the vendored Makefile leaves
# -ffp-contract at gfortran's default, and EXP/LOG/ERF come from the platform
# libm. So this target is deliberately never wired into CI -- see
# docs/REFERENCE_BUILD.md.
capture:
	./validation/build_reference.sh both
	./validation/build_f2py.sh
	$(PY) validation/capture_reference.py
	$(PY) validation/capture_leaf.py
	$(PY) validation/capture_modes.py
	$(PY) validation/capture_gas_indices.py
	$(PY) validation/capture_budget_indices.py
	$(PY) validation/capture_coag_mode.py
	$(PY) validation/capture_vapour_leaf.py

# Regenerate, then REPORT. This target used to end in `--write`, so the one
# command that rewrites every golden also re-blessed every one of them --
# against docs/harness.md, which says auto-blessing would make the drift gate
# silent the one time it matters, and against every capture script, which prints
# the same advice. Blessing is now a second, explicit command.
#
# Exiting non-zero when something moved is the point: a regeneration that
# changed a golden is a finding.
goldens: capture
	@echo "==> comparing the regenerated captures against the committed manifest"
	@$(PY) validation/goldens_manifest.py --check || { \
	  echo ""; \
	  echo "The above is what moved. A golden that moved is a finding, not a"; \
	  echo "knob. Once -- and only once -- every change above is understood and"; \
	  echo "intended, record it explicitly with:  make goldens-bless"; \
	  exit 1; }

goldens-bless:
	$(PY) validation/goldens_manifest.py --write

bench:
	$(PY) benchmarks/bench_throughput.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
