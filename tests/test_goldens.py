"""Task 19: the committed reference fixtures, and what they must satisfy.

**No `fortran` marker and no gfortran skip anywhere in this file.** That is the
point of the task. Every physics test from phase C onwards compares against
these archives, and if loading them needed a Fortran toolchain then none of that
would run in CI. Goldens are generated once on a pinned toolchain (ADR-005) and
committed; from here on the Fortran build is needed only to *regenerate* them.

The assertions here are structural, not numerical. They answer "is this fixture
a usable reference at all" — right shape, no NaNs, physically sane signs,
consistent across variants — leaving "does the port reproduce it" to the phases
that do the porting. Getting that split wrong is how a fixture suite ends up
asserting the port's own bugs.

The f32/f64 pair is the one place numbers appear, because the gap between them
IS the measurement (task 13). Bounding it from both sides catches the two ways
the pair can be wrong: a variant that silently built in the wrong precision, and
a divergence that is no longer precision at all. That bound is asserted
per-case, not globally — 3.7e-4 holds for the three setup-1 runs and does not
hold for `marine_bcoc`, which is a finding in its own right and is pinned by its
own test below.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDENS = REPO / "tests" / "goldens"
FORTRAN = REPO / "fortran"
sys.path.insert(0, str(REPO / "validation"))
import goldens_manifest as gm  # noqa: E402

CASES = ["bl_nmts3", "boundary_layer", "free_troposphere", "marine_bcoc"]
MODES = ["trajectory", "budgets", "state", "branches"]
# Leaf sweeps (task 21): not per-case, and not produced by the box binary.
LEAF_ARCHIVES = {
    "numerics.f64.leaf.npz",
    "modes.f64.tables.npz",
    "gasidx.f64.tables.npz",
    "budidx.f64.tables.npz",
    "coagmode.f64.tables.npz",
    "vapour.f64.leaf.npz",
    "drydiam.f64.leaf.npz",
}

# Task 13. The floor is a property of the trajectory over 48 steps of 1800 s.
PRECISION_FLOOR = 3.7e-4


def load(case, variant, mode):
    path = GOLDENS / f"{case}.{variant}.{mode}.npz"
    assert path.is_file(), f"{path.name} missing -- run `make goldens`"
    return np.load(path, allow_pickle=False)


def test_the_expected_archives_are_committed():
    """A fixture set that quietly shrinks is a coverage loss that no other test
    would notice, because the tests that use a fixture skip when it is absent."""
    expected = {f"{c}.f64.{m}.npz" for c in CASES for m in MODES}
    expected |= {f"{c}.f32.trajectory.npz" for c in CASES}
    expected |= LEAF_ARCHIVES
    assert {p.name for p in GOLDENS.glob("*.npz")} == expected


def test_the_fixtures_load_without_a_fortran_toolchain():
    """Stated as its own test because it is the acceptance criterion, and
    because it is the property most easily lost by accident later."""
    for path in sorted(GOLDENS.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            assert data.files, f"{path.name} is empty"


def test_the_manifest_covers_every_committed_fixture():
    assert gm.verify() == []
    recorded = gm.load()["goldens"]
    assert set(recorded) == {p.name for p in GOLDENS.glob("*.npz")}


def test_the_manifest_records_the_toolchain_the_fixtures_were_built_with():
    """ADR-005: goldens are not portable across compilers or platforms, so the
    committed set is worthless without a record of what produced it."""
    toolchain = gm.load()["toolchain"]
    assert "gfortran" in toolchain
    assert "-ffp-contract=off" in toolchain["flags"]
    assert toolchain["f64_flag"] == "-fdefault-real-8"


# --------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("variant", ["f32", "f64"])
def test_trajectory_shape_and_finiteness(case, variant):
    data = load(case, variant, "trajectory")
    values, columns = data["values"], data["columns"]
    assert values.shape == (49, len(columns)), "48 steps plus the initial state"
    assert np.isfinite(values).all()
    assert list(columns[:2]) == ["time_s", "time_h"]


@pytest.mark.parametrize("case", CASES)
def test_trajectory_quantities_are_physical(case):
    """Number, diameter and density are positive quantities. A fixture with a
    negative one is a broken reference, not a tight tolerance."""
    data = load(case, "f64", "trajectory")
    columns = list(data["columns"])
    for prefix in ("N_", "Ddry_", "Dwet_", "rhop_", "M_"):
        idx = [i for i, c in enumerate(columns) if c.startswith(prefix)]
        assert idx, f"{case}: no {prefix}* columns"
        assert (data["values"][:, idx] >= 0).all(), f"{case}: negative {prefix}*"


@pytest.mark.parametrize("case", CASES)
def test_time_advances_by_the_namelist_step(case):
    time_s = load(case, "f64", "trajectory")["values"][:, 0]
    assert time_s[0] == 0.0
    assert np.allclose(np.diff(time_s), 1800.0)


def _column_scaled_gap(case):
    """f32-vs-f64, per column, scaled by that column's own range.

    Scaling by the column rather than by each element is what conftest's
    `atol_scale` exists for: `M_*` entries are legitimately zero for absent
    components, and a pure elementwise relative error would be dominated by
    0-vs-1e-30 rather than by anything physical.
    """
    f32 = load(case, "f32", "trajectory")["values"]
    d64 = load(case, "f64", "trajectory")
    f64 = d64["values"]
    assert f32.shape == f64.shape
    colmax = np.maximum(np.abs(f64).max(axis=0), 1e-300)
    return list(d64["columns"]), (np.abs(f32 - f64) / colmax).max(axis=0)


@pytest.mark.parametrize("case", ["boundary_layer", "free_troposphere", "bl_nmts3"])
def test_the_f32_f64_gap_is_the_measured_precision_floor(case):
    """Task 13's 3.7e-4, re-derived from the committed fixtures.

    Below ~1e-6 the two variants would not really be different precisions;
    above ~1e-2 the difference is no longer precision at all. Note the case
    list: setup 2 is excluded and has its own test below, for reasons that are
    a finding rather than an inconvenience."""
    _, gaps = _column_scaled_gap(case)
    gap = float(gaps.max())
    assert 1e-6 < gap < 1e-2, f"{case}: f32/f64 gap {gap:.2e}, expected ~{PRECISION_FLOOR:.1e}"


def test_the_precision_floor_does_not_hold_for_a_depleting_insoluble_mode():
    """`marine_bcoc` is the only shipped case with an insoluble mode, and its
    f32 reference is quantitatively worthless by the end of the run.

    Ageing depletes the Aitken insoluble mode over seven orders of magnitude. In
    f64, number and mass leave in proportion and the mean dry diameter stays
    pinned at 30 nm. In f32 the residual loses significance, mass leaves faster
    than number, the diameter collapses to 5.8 nm and `N_aitins` stops decaying
    and turns back upward. Catastrophic cancellation in a depleting mode, not a
    threshold that flipped -- the divergence is continuous from around step 20,
    well before the first branch disagreement at step 45.

    Asserted rather than skipped so that (a) "the precision floor is 3.7e-4"
    cannot be quoted as a global fact, and (b) if a future change makes the f32
    run behave, that is noticed rather than silently absorbed. The f64
    reference is well behaved throughout, so nothing here threatens the port.
    See docs/porting-notes.md and issue #14."""
    columns, gaps = _column_scaled_gap("marine_bcoc")
    aitins = [i for i, c in enumerate(columns) if c.endswith("_aitins_nm")]
    others = [i for i in range(len(columns)) if i not in aitins]

    assert gaps[aitins].min() > 0.5, "the aitins diameters no longer diverge -- investigate"
    assert gaps[others].max() < 1e-2, (
        f"a column outside the insoluble diameters now diverges: "
        f"{columns[int(np.argmax(np.where(np.isin(np.arange(len(columns)), others), gaps, 0)))]}"
    )


# --------------------------------------------------------------------------
# Budgets, state, branches
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_budgets_are_one_row_per_step_and_never_negative(case):
    """bud_aer_mas accumulates mass fluxes and is zeroed per step, so a negative
    entry means a process removed more than it added."""
    data = load(case, "f64", "budgets")
    assert data["values"].shape[0] == 48
    assert np.isfinite(data["values"]).all()
    fluxes = data["values"][:, 2:]  # step, time_s, then bud0..budN
    assert (fluxes >= 0).all()


@pytest.mark.parametrize("case", CASES)
def test_budget_slot_zero_is_never_written(case):
    """All 344 writes in the vendored tree are wrapped in `IF (nmasxxx > 0)`,
    so slot 0 is a hole and not a null sink. A port that clamps unset indices
    to 0 and scatters into it changes the semantics.

    344, not the ~684 this said before: a write site spans two source lines, so
    a line-oriented grep double-counts it. The number is machine-checked by
    `tests/test_budget_indices.py::test_every_write_site_is_guarded_on_its_own_index`,
    which is also where "guarded" is verified — this test only shows the column
    stays zero in four committed runs."""
    data = load(case, "f64", "budgets")
    columns = list(data["columns"])
    assert (data["values"][:, columns.index("bud0")] == 0).all()


# The thirteen call sites, counted in `ukca_aero_step.F90`: `ukca_calc_drydiam`
# at :541, :561, :1146 and :1171; `ukca_volume_mode` at :564, :1149 and :1175;
# `ukca_remode` at :545 and :1156; and `ukca_conden` (:925),
# `ukca_calcnucrate` (:1030), `ukca_coagwithnucl` (:1099) and `ukca_ageing`
# (:1129) once each. The first pair of drydiam/volume_mode/remode calls runs
# before the `imts` loop and the rest inside it, which is how the two groups
# are told apart below. `tests/test_state_dump.py` asserts the same thirteen
# against a live instrumented run; this asserts them from the committed
# archives, which is the half that runs without gfortran.
CALLS_PER_SITE = {
    "drydiam": 4,
    "volume_mode": 3,
    "remode": 2,
    "conden": 1,
    "calcnucrate": 1,
    "coagwithnucl": 1,
    "ageing": 1,
}

# `ukca_calcnucrate` writes a second record at the next `seq` carrying only
# these four gas-phase fields (overlay 0005). It is the same call, so the gas
# record must not be counted as a fourteenth site.
GAS_FIELDS = ("h2so4", "delh2so4_nucl", "sec_org", "s_cond_s")

AERO_STEP_CALLS = {
    "drydiam": ("CALL ukca_calc_drydiam", (541, 561, 1146, 1171)),
    "volume_mode": ("CALL ukca_volume_mode", (564, 1149, 1175)),
    "remode": ("CALL ukca_remode", (545, 1156)),
    "conden": ("CALL ukca_conden", (925,)),
    "calcnucrate": ("CALL ukca_calcnucrate", (1030,)),
    "coagwithnucl": ("CALL ukca_coagwithnucl", (1099,)),
    "ageing": ("CALL ukca_ageing", (1129,)),
}

KEY_COLUMNS = ("step", "seq", "imts", "izts", "imode", "icp", "field")


@pytest.mark.parametrize("case", CASES)
def test_state_snapshots_cover_all_thirteen_call_sites(case):
    """Thirteen, checked — the name used to be aspirational.

    What this asserted before was that seven site *names* are present and that
    `int(data["_rows"]) == len(data["value"])`, which is true by construction:
    the capture writes `_rows = len(rows)`. Neither says anything about
    thirteen. The archives distinguish far more than seven — 84 distinct
    `(site, seq)` pairs per step, 94 for `bl_nmts3` — because `seq` counts
    calls, not sites.

    So the call sites are reconstructed: a call is one `(site, step, imts,
    izts, seq)` group, and a call *site* is a routine plus whether it runs
    inside the `imts` loop plus its rank among that routine's calls in one
    iteration. That gives 13 in all four cases, with the same per-routine
    breakdown as the Fortran source, and it stays 13 when `nmts` changes from
    1 to 3 — which is the point of counting sites rather than calls.
    """
    data = load(case, "f64", "state")
    sites = list(data["site_levels"])
    assert set(sites) == set(CALLS_PER_SITE)
    assert np.isfinite(data["value"]).all()

    fields = list(data["field_levels"])
    gas = [fields.index(f) for f in GAS_FIELDS if f in fields]
    aerosol_row = ~np.isin(data["field"], gas)
    site, seq, step, imts, izts = (
        data[k][aerosol_row] for k in ("site", "seq", "step", "imts", "izts")
    )

    calls = set(zip(site.tolist(), step.tolist(), imts.tolist(), izts.tolist(), seq.tolist()))
    per_iteration = {}
    for isite, istep, imt, izt, iseq in calls:
        per_iteration.setdefault((isite, istep, imt, izt), []).append(iseq)

    found = {}
    for (isite, _step, imt, _izts), seqs in per_iteration.items():
        for rank in range(len(seqs)):
            found.setdefault(sites[isite], set()).add((imt > 0, rank))

    assert {name: len(ranks) for name, ranks in found.items()} == CALLS_PER_SITE
    assert sum(CALLS_PER_SITE.values()) == 13


def test_the_thirteen_call_sites_are_where_the_comment_says():
    """The source half of the count above, so "thirteen" is not folklore.

    Reading `fortran/src/ukca/`, which is read-only and committed, needs no
    toolchain — so this belongs in this file rather than behind the `fortran`
    marker. Fails if a vendored update moves or adds a call, which is exactly
    the change that would silently make the archives cover twelve or fourteen.
    """
    lines = (
        (FORTRAN / "src" / "ukca" / "ukca_aero_step.F90").read_text(encoding="utf-8").splitlines()
    )
    for site, (needle, linenos) in AERO_STEP_CALLS.items():
        found = [n for n, line in enumerate(lines, 1) if line.lstrip().startswith(needle)]
        assert found == list(linenos), f"{site}: {needle} is now at {found}"
        assert len(linenos) == CALLS_PER_SITE[site], site
    assert sum(len(linenos) for _, linenos in AERO_STEP_CALLS.values()) == 13


@pytest.mark.parametrize("case", CASES)
def test_state_keys_are_unique(case):
    """`seq` is load-bearing and this is what makes it so.

    `calc_drydiam` and `volume_mode` each run twice per `imts`, so without
    `seq` the key `(step, imts, izts, imode, icp, field)` is not unique: a
    committed golden once carried 397 keys with two different values,
    separable only by file row order. Comparing `_rows` against
    `len(data["value"])` — which is what stood here — could not have caught
    that, because the capture writes `_rows = len(rows)`.
    """
    data = load(case, "f64", "state")
    rows = int(data["_rows"])
    assert {len(data[c]) for c in (*KEY_COLUMNS, "site", "value")} == {rows}

    key = np.stack([data[c].astype(np.int64) for c in KEY_COLUMNS], axis=1)
    assert len(np.unique(key, axis=0)) == rows, (
        f"{case}: {rows - len(np.unique(key, axis=0))} rows share a "
        f"{KEY_COLUMNS} key, so the dump cannot be joined against the port"
    )


@pytest.mark.parametrize("case", CASES)
def test_branch_records_are_masks_or_small_codes(case):
    """Gate 0's fixtures. Values are 0/1 masks, solvecoagnucl_v's 0-7 branch
    code, or nmodemax_merge's 2/3 -- nothing else."""
    data = load(case, "f64", "branches")
    assert data["value"].dtype == np.int8
    assert data["value"].min() >= 0
    assert data["value"].max() <= 7


@pytest.mark.parametrize("case", CASES)
def test_branch_fixtures_carry_the_up1_evidence(case):
    """The factor-3 branch (code 5) fires every substep for the top soluble
    mode, in every case. This is the fixture that the UP-1 fidelity flag's
    default is answerable to -- see docs/porting-notes.md."""
    data = load(case, "f64", "branches")
    tag = np.array(data["tag_levels"]) == "form"
    site = np.array(data["site_levels"]) == "coag_sol_solve"
    rows = tag[data["tag"]] & site[data["site"]]
    assert rows.any(), f"{case}: no solvecoagnucl_v branch codes recorded"
    assert (data["value"][rows] == 5).any(), f"{case}: the UP-1 branch never fires"


@pytest.mark.parametrize("case", CASES)
def test_nmts_structure_is_resolved_in_the_dumps(case):
    """bl_nmts3 is the only case with nmts > 1, and it exists precisely because
    no shipped namelist exercises the nested outer/inner scan."""
    imts = set(load(case, "f64", "branches")["imts"].tolist())
    expected = {1, 2, 3} if case == "bl_nmts3" else {1}
    assert expected <= imts, f"{case}: imts {sorted(imts)}"
