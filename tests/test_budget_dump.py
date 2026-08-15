"""Task 14: the per-process budget dump carries real, attributable signal.

`ukca_aero_step` fills `bud_aer_mas` with per-process mass fluxes and the
shipped driver zeroes it every step and never writes it out. Exposing it is what
turns validation from "the final number matches" into "H2SO4 condensation onto
the accumulation mode matches at step 7".

The slot indices asserted below are from `ukca_indices_suss_4mode`
(`i_mode_setup = 1`):

    31 nmascondsuaitsol   32 nmascondsuaccsol   33 nmascondsucorsol
    38 nmascoagsuintr23   39 nmascoagsuintr24   40 nmascoagsuintr34

i.e. H2SO4 condensation onto the three populated soluble modes, and inter-modal
coagulation 2->3, 2->4, 3->4. For the boundary-layer case -- sustained H2SO4
production, condensation-dominated, no nucleation burst -- that is exactly the
set physics predicts, which is what makes this a check rather than a snapshot.
"""

import csv
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FORTRAN = REPO / "fortran"

pytestmark = pytest.mark.fortran
needs_gfortran = pytest.mark.skipif(
    shutil.which("gfortran") is None, reason="gfortran not available"
)

CONDENSATION_SLOTS = {31, 32, 33}
COAGULATION_SLOTS = {38, 39, 40}


@pytest.fixture(scope="module")
def budgets(tmp_path_factory):
    exe = FORTRAN / "bin-ref-f64" / "glomap_box"
    if not exe.is_file():
        pytest.skip("reference not built; run validation/build_reference.sh")
    d = tmp_path_factory.mktemp("bud")
    out, bud, nml = d / "o.csv", d / "bud.csv", d / "o.nml"
    text = (FORTRAN / "namelists" / "boundary_layer.nml").read_text(encoding="utf-8")
    text = text.replace("out/boundary_layer.csv", str(out))
    text = text.replace("  verbose      = 0", f"  verbose      = 0\n  budget_file  = '{bud}'")
    nml.write_text(text, encoding="utf-8")
    subprocess.run([str(exe), str(nml)], check=True, capture_output=True)
    raw = bud.read_text(encoding="utf-8")
    rows = list(csv.reader(bud.open()))
    # The raw text is returned alongside the parsed floats because precision is
    # a property of the DIGITS, and parsing to float throws them away.
    return (
        [h.strip() for h in rows[0]],
        [[float(x) for x in r] for r in rows[1:]],
        raw,
    )


def _slot(row, i):
    return row[2 + i]  # columns are step, time_s, then bud0..budN


@needs_gfortran
def test_one_row_per_chemistry_step(budgets):
    _, data, _ = budgets
    assert len(data) == 48


@needs_gfortran
def test_slot_zero_is_never_written(budgets):
    """Upstream guards every write with IF (nmasxxx > 0), so slot 0 is dead.

    Asserted rather than assumed: a non-zero value here would mean a budget
    index was left unset and the write fell through to the null slot, which
    would corrupt an unrelated diagnostic in a port that scatters into it.
    """
    _, data, _ = budgets
    assert all(_slot(r, 0) == 0.0 for r in data)


@needs_gfortran
def test_condensation_slots_are_active(budgets):
    _, data, _ = budgets
    for slot in sorted(CONDENSATION_SLOTS):
        peak = max(abs(_slot(r, slot)) for r in data)
        assert peak > 0.0, f"budget slot {slot} (H2SO4 condensation) is empty"


@needs_gfortran
def test_coagulation_slots_are_active(budgets):
    _, data, _ = budgets
    for slot in sorted(COAGULATION_SLOTS):
        peak = max(abs(_slot(r, slot)) for r in data)
        assert peak > 0.0, f"budget slot {slot} (inter-modal coagulation) is empty"


@needs_gfortran
def test_only_the_physically_expected_slots_are_active(budgets):
    """The boundary-layer case is condensation-dominated with no nucleation
    burst, so precisely condensation and inter-modal coagulation should fire.

    A slot lighting up outside that set means either the physics changed or a
    budget index is being written through the wrong path -- both worth knowing.
    """
    header, data, _ = budgets
    n_slots = len(header) - 2
    active = {i for i in range(n_slots) if any(abs(_slot(r, i)) > 0.0 for r in data)}
    expected = CONDENSATION_SLOTS | COAGULATION_SLOTS
    assert active == expected, (
        f"unexpected active slots: {sorted(active - expected)}, "
        f"missing: {sorted(expected - active)}"
    )


@needs_gfortran
def test_budgets_carry_full_precision(budgets):
    """The phase B review found this asserted only `peak > 1e6` -- a magnitude
    check with no bearing on precision. Reverting the budget overlay to ES14.6
    did not fail it, while the equivalent test in test_state_dump.py did.

    Count the mantissa digits, as that test does."""
    _, data, text = budgets
    peak = max(abs(_slot(r, 32)) for r in data)
    assert peak > 1.0e6, "accumulation-mode condensation flux implausibly small"

    sample = next(
        f.strip()
        for line in text.splitlines()[1:]
        for f in line.split(",")[2:]
        if "E" in f and float(f) != 0.0
    )
    mantissa = sample.split("E")[0].replace("-", "").replace(".", "")
    assert len(mantissa.rstrip("0")) >= 10, (
        f"budget field {sample!r} carries only {len(mantissa.rstrip('0'))} "
        f"significant digits; the ES24.16 overlay is not in effect"
    )
