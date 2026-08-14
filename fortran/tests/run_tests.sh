#!/usr/bin/env bash
# Regression tests for the standalone GLOMAP-mode box model.
#
# Each test isolates one microphysical process and asserts a property that
# must hold for that process alone, so a failure points at a specific
# pathway rather than "the model changed".
#
# Run with:  make test    (or  ./tests/run_tests.sh  from the repo root)

set -uo pipefail

cd "$(dirname "$0")/.."
BIN=./bin/glomap_box
CHECK="python3 tests/check_csv.py"
mkdir -p out

pass=0
fail=0

report() {
  if [ "$1" -eq 0 ]; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
    echo "  --> FAILED"
  fi
}

banner() {
  echo
  echo "=== $1 ==="
}

run_case() {
  # run_case <namelist> <expected_csv>
  #
  # Deletes the expected output first and requires the run to recreate it.
  # Checking only the exit status is not sufficient on its own: a stale CSV
  # from a previous run would otherwise be asserted against, so a crashed or
  # skipped run could report as a pass.
  local nml="$1" csv="$2"
  rm -f "$csv"
  if ! $BIN "$nml" > /dev/null 2>&1; then
    echo "  model run FAILED (non-zero exit) for $nml"
    return 1
  fi
  if [ ! -s "$csv" ]; then
    echo "  model run produced no output at $csv"
    return 1
  fi
  return 0
}

# The suite must be able to fail. Confirm the binary reports a fatal error with
# a non-zero exit status before trusting any run_case result below.
banner "-1. harness self-check: a failing run is detected"
if $BIN /nonexistent-namelist-for-self-check.nml > /dev/null 2>&1; then
  echo "  the binary exited 0 on a missing namelist -- every result below"
  echo "  would be meaningless. Refusing to continue."
  exit 1
fi
echo "  ok: fatal errors produce a non-zero exit status"
pass=$((pass + 1))

# ---------------------------------------------------------------------------
banner "0. no external module dependencies"
python3 tools/gen_build_order.py --check src
report $?

# ---------------------------------------------------------------------------
banner "1. all processes off => state is invariant"
# With condensation, nucleation and coagulation off and no gas production,
# nothing should move: number, size and mass must all hold exactly.
if run_case tests/namelists/all_off.nml out/test_all_off.csv; then
  $CHECK out/test_all_off.csv finite            && \
  $CHECK out/test_all_off.csv constant N_aitsol 1e-12 && \
  $CHECK out/test_all_off.csv constant N_accsol 1e-12 && \
  $CHECK out/test_all_off.csv constant Ddry_aitsol 1e-12 && \
  $CHECK out/test_all_off.csv sum_constant M_ 1e-12
  report $?
else
  report 1
fi

# ---------------------------------------------------------------------------
banner "2. coagulation only => number falls, dry mass conserved"
# Coagulation moves mass between modes but creates and destroys none, so the
# summed component mass must be conserved while number decreases.
if run_case tests/namelists/coag_only.nml out/test_coag_only.csv; then
  $CHECK out/test_coag_only.csv finite               && \
  $CHECK out/test_coag_only.csv decreases N_aitsol   && \
  $CHECK out/test_coag_only.csv sum_constant M_ 1e-6
  report $?
else
  report 1
fi

# ---------------------------------------------------------------------------
banner "3. condensation only => mass grows, number conserved"
# Condensation adds vapour mass to existing particles without changing their
# number, and the modes must grow.
if run_case tests/namelists/cond_only.nml out/test_cond_only.csv; then
  $CHECK out/test_cond_only.csv finite                    && \
  $CHECK out/test_cond_only.csv constant N_aitsol 1e-9    && \
  $CHECK out/test_cond_only.csv increases M_               && \
  $CHECK out/test_cond_only.csv increases Ddry_aitsol
  report $?
else
  report 1
fi

# ---------------------------------------------------------------------------
banner "4. cold clean case => nucleation burst"
# Low temperature and a small condensation sink must let binary homogeneous
# nucleation populate the nucleation mode.
if run_case namelists/free_troposphere.nml out/free_troposphere.csv; then
  $CHECK out/free_troposphere.csv finite && \
  $CHECK out/free_troposphere.csv exceeds N_nucsol 1.0e4
  report $?
else
  report 1
fi

# ---------------------------------------------------------------------------
banner "5. polluted BL case => condensational growth, no runaway nucleation"
if run_case namelists/boundary_layer.nml out/boundary_layer.csv; then
  $CHECK out/boundary_layer.csv finite && \
  $CHECK out/boundary_layer.csv increases Ddry_aitsol && \
  $CHECK out/boundary_layer.csv decreases N_aitsol
  report $?
else
  report 1
fi

# ---------------------------------------------------------------------------
banner "6. 5-mode config => ageing drains the insoluble Aitken mode"
# In the SUSSBCOC 5-mode setup, condensation of H2SO4/organics onto the
# insoluble Aitken mode ages it into the soluble Aitken mode.
if run_case namelists/marine_bcoc.nml out/marine_bcoc.csv; then
  $CHECK out/marine_bcoc.csv finite && \
  $CHECK out/marine_bcoc.csv decreases N_aitins && \
  $CHECK out/marine_bcoc.csv increases N_aitsol
  report $?
else
  report 1
fi

# ---------------------------------------------------------------------------
banner "7. every supported mode configuration runs and stays finite"
# Guards the i_mode_setup dispatch in glomap_box_config_mod against a mode/gas
# index pairing that only fails for one configuration. Setup 6 (dust-only) is
# the important one: it has no soluble modes and no gas-phase chemistry at all.
ms_fail=0
for ms in 1 2 3 4 5 6 8; do
  sed -e "s/@MS@/$ms/" tests/namelists/mode_setup.nml.in > out/ms_$ms.nml
  if ! run_case "out/ms_$ms.nml" "out/ms_$ms.csv"; then
    echo "  i_mode_setup=$ms FAILED to run"
    ms_fail=1
    continue
  fi
  if ! $CHECK "out/ms_$ms.csv" finite > /dev/null; then
    echo "  i_mode_setup=$ms produced non-finite or negative output"
    ms_fail=1
    continue
  fi
  echo "  ok: i_mode_setup=$ms ran and output is finite"
done
report $ms_fail

# ---------------------------------------------------------------------------
echo
echo "======================================================="
echo " passed: $pass    failed: $fail"
echo "======================================================="
[ "$fail" -eq 0 ] || exit 1
