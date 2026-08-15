"""Task 15: per-process state snapshots localise a divergence to one call.

`ukca_aero_step` calls thirteen process routines per chemistry step. Comparing
only the end state tells you THAT a port diverged; comparing after each call
tells you WHERE. Budgets (task 14) carry mass fluxes only -- no `nd`, no `mdt`,
no `drydp` -- so they cannot localise the failures that actually happen in
coagulation and mode merging.

The counts below are exact rather than approximate, and reconcile as:

    inner sites (conden, calcnucrate, coagwithnucl, ageing)
        3 steps x 15 nzts x 96 values = 4320 each
    drydiam      4 calls/step  -> 3 x 4 x 96 = 1152
    volume_mode  3 calls/step  -> 3 x 3 x 96 =  864
    remode       2 calls/step  -> 3 x 2 x 96 =  576

where 96 = 8 modes x (6 scalar fields + 6 components). Asserting the exact
counts means a call site silently gained or lost an invocation -- which is
precisely the kind of splitting-order change the port must not make -- fails
here rather than as a mysterious trajectory mismatch later.
"""

import csv
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FORTRAN = REPO / "fortran"

pytestmark = pytest.mark.fortran
needs_gfortran = pytest.mark.skipif(
    shutil.which("gfortran") is None, reason="gfortran not available"
)

NSTEPS = 3
NZTS = 15
VALUES_PER_SNAPSHOT = 8 * (6 + 6)  # 8 modes x (6 scalar fields + 6 components)

# calcnucrate carries the aerosol snapshot AND four gas-phase fields. It needs
# the latter: ukca_calcnucrate writes only h2so4, delh2so4_nucl, jrate and
# s_cond_s, none of which the aerosol arrays carry, so before overlay 0005 the
# site was byte-identical to the preceding conden snapshot in every record.
GAS_FIELDS_PER_CALL = 4

EXPECTED_RECORDS = {
    "conden": NSTEPS * NZTS * VALUES_PER_SNAPSHOT,
    "calcnucrate": NSTEPS * NZTS * (VALUES_PER_SNAPSHOT + GAS_FIELDS_PER_CALL),
    "coagwithnucl": NSTEPS * NZTS * VALUES_PER_SNAPSHOT,
    "ageing": NSTEPS * NZTS * VALUES_PER_SNAPSHOT,
    "drydiam": NSTEPS * 4 * VALUES_PER_SNAPSHOT,
    "volume_mode": NSTEPS * 3 * VALUES_PER_SNAPSHOT,
    "remode": NSTEPS * 2 * VALUES_PER_SNAPSHOT,
}


@pytest.fixture(scope="module")
def dump(tmp_path_factory):
    exe = FORTRAN / "bin-ref-f64" / "glomap_box"
    if not exe.is_file():
        pytest.skip("reference not built; run validation/build_reference.sh")
    d = tmp_path_factory.mktemp("sd")
    out, state, nml = d / "o.csv", d / "state.csv", d / "o.nml"
    text = (FORTRAN / "namelists" / "boundary_layer.nml").read_text(encoding="utf-8")
    text = text.replace("out/boundary_layer.csv", str(out))
    text = text.replace("  nsteps       = 48", f"  nsteps       = {NSTEPS}")
    text = text.replace("  verbose      = 0", f"  verbose      = 0\n  state_file   = '{state}'")
    nml.write_text(text, encoding="utf-8")
    subprocess.run([str(exe), str(nml)], check=True, capture_output=True)
    return list(csv.DictReader(state.open()))


@needs_gfortran
def test_all_thirteen_call_sites_are_instrumented(dump):
    assert set(Counter(r["site"] for r in dump)) == set(EXPECTED_RECORDS)


@needs_gfortran
@pytest.mark.parametrize("site", sorted(EXPECTED_RECORDS))
def test_each_site_fires_the_expected_number_of_times(dump, site):
    """A site gaining or losing an invocation is a splitting-order change."""
    actual = sum(1 for r in dump if r["site"] == site)
    assert actual == EXPECTED_RECORDS[site], (
        f"{site} produced {actual} records, expected {EXPECTED_RECORDS[site]}. "
        f"A call site changed how often it runs -- check the nmts/nzts nesting."
    )


@needs_gfortran
def test_every_nzts_substep_is_individually_resolved(dump):
    """Without this the dump would only localise to a chemistry step, which is
    far too coarse: the competition loop runs 15 times inside each one."""
    izts = {int(r["izts"]) for r in dump if r["site"] == "conden"}
    assert izts == set(range(1, NZTS + 1))


@needs_gfortran
def test_outer_sites_report_zero_substep_counters(dump):
    """remode runs outside the nzts loop, so its counters are not in scope."""
    assert {r["izts"] for r in dump if r["site"] == "remode"} == {"0"}


@needs_gfortran
def test_values_carry_full_precision(dump):
    sample = next(r for r in dump if r["field"] == "drydp" and float(r["value"]) > 0)
    mantissa = sample["value"].split("E")[0].replace("-", "").replace(".", "")
    assert len(mantissa.rstrip("0")) >= 10


@needs_gfortran
def test_state_evolves_between_substeps(dump):
    """If consecutive snapshots were identical the instrumentation would be
    reporting a stale copy rather than the live state."""
    vals = [
        float(r["value"])
        for r in dump
        if r["site"] == "conden" and r["field"] == "mdt" and r["imode"] == "2" and r["step"] == "1"
    ]
    assert len(set(vals)) > 1, "mdt is identical across all conden substeps"


# --------------------------------------------------------------------------
# What overlay 0005 fixed
# --------------------------------------------------------------------------


@needs_gfortran
def test_records_are_uniquely_keyed(dump):
    """`ukca_calc_drydiam` and `ukca_volume_mode` each run TWICE per `imts` —
    once before `ukca_remode` and once after — and both were tagged with the
    same `(site, imts, izts=0)`.

    The phase B review measured the consequence in a committed golden: 13,824
    duplicate keys, 397 of them carrying two different values, distinguishable
    only by file row order. A consumer joining on the advertised columns keeps
    one of each pair, so a port whose post-remode `volume_mode` is wrong but
    whose pre-remode one is right passes or fails by luck.

    Overlay 0005 adds a per-step call-sequence counter."""
    keys = [
        (r["step"], r["seq"], r["site"], r["imts"], r["izts"], r["field"], r["imode"], r["icp"])
        for r in dump
    ]
    assert len(set(keys)) == len(keys), f"{len(keys) - len(set(keys))} duplicate keys"


@needs_gfortran
def test_the_sequence_counter_restarts_each_step_and_is_contiguous(dump):
    """Otherwise it is an opaque row id rather than a call index."""
    from collections import defaultdict

    per_step = defaultdict(set)
    for r in dump:
        per_step[r["step"]].add(int(r["seq"]))
    for step, seqs in per_step.items():
        assert seqs == set(range(1, len(seqs) + 1)), f"step {step}: {sorted(seqs)[:5]}…"


@needs_gfortran
def test_calcnucrate_records_what_calcnucrate_actually_writes(dump):
    """`ukca_calcnucrate` touches no aerosol array, so before overlay 0005 its
    snapshot was byte-identical to the preceding `conden` one — 21.7% of the
    whole dump was repeated records, and a wrong nucleation rate would first
    have shown up at `coagwithnucl`."""
    fields = {r["field"] for r in dump if r["site"] == "calcnucrate"}
    assert {"h2so4", "delh2so4_nucl", "sec_org", "s_cond_s"} <= fields

    evolving = [
        float(r["value"]) for r in dump if r["site"] == "calcnucrate" and r["field"] == "h2so4"
    ]
    assert len(set(evolving)) == len(evolving), (
        "h2so4 is identical across every substep; the site is still reporting a "
        "stale copy rather than live state"
    )
