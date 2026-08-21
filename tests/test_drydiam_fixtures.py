"""Task 35d: `tests/goldens/drydiam.f64.leaf.npz` is a usable reference.

This file asks whether the fixture covers `ukca_calc_drydiam` and says what it
claims to say. It does **not** ask whether a port reproduces it -- there is no
port yet; task 36 writes it and task 37 the reset. Getting that split wrong is
how a fixture suite ends up asserting the port's own bugs.

No `fortran` marker and no gfortran skip: the archive is committed, so all of
this runs in CI.

The stated acceptance for task 35 was "shape+finiteness per fixture", which an
all-zeros archive of the right shape passes. What is here instead:

* every predicate in the routine has a **hit count committed inside the
  archive** and is asserted to be non-zero, or to be zero for a reason recorded
  at capture time and re-derived here from `modes.f64.tables.npz` rather than
  taken on trust;
* the closed forms of the two arms that are pure arithmetic (`:210`, `:225`)
  and of the member sum (`:218`) are recomputed from the **mode tables golden**
  and compared byte for byte. The mode tables come from the Fortran, not from
  `glomap_jax.physics.modes`, so nothing here is the port grading itself;
* the reset (`:250-258`) is checked on all four things it writes, including the
  one it must NOT write -- a non-member component's `md`;
* `mdt` is shown to be write-only by the pair of runs the capture made with
  disjoint garbage, and the disjointness is re-checked here so the
  demonstration cannot be vacuous;
* the deliberate abort is compared on the **ereport record** -- fatal count and
  message -- and not on the numbers it returned, which are meaningless.

Where a quantity has been through `cubrt_v` the comparison is
`assert_matches_reference`, which is byte equality on the capture platform and
a measured ulp window elsewhere. `x ** (1.0/3.0)` differs by up to 1 ulp
between Darwin arm64 and ubuntu x86_64 (see `tests/conftest.py`), and `drydp`
is exactly that expression.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import assert_matches_reference

GOLDENS = Path(__file__).parent / "goldens"
ARCHIVE = GOLDENS / "drydiam.f64.leaf.npz"
MODE_TABLES = GOLDENS / "modes.f64.tables.npz"

SETUPS = (1, 2, 3, 4, 5, 6, 8)
COMBOS = ("default", "nacl_off", "bc_mg_mix", "hygro_off", "dust_ageing")
CONTROLS = ("hygro_off", "dust_ageing")
CONFIGS = [(s, c) for s in SETUPS for c in COMBOS]
IDS = [f"s{s}-{c}" for s, c in CONFIGS]

# ukca_mode_setup.F90:86-88. The reset loop at :245 runs mode_nuc_sol to
# mode_acc_sol and no further, which is the whole reason slot 4 can abort.
RESET_MODES = (0, 1, 2)

# ukca_constants / ukca_config_constants. Re-parsed from the vendored Fortran by
# tests/test_constants.py, so importing them here is not a second source.
AVOGADRO = 6.022e23
MMSUL = 0.09808
RHO_SO4 = 1769.0
PI = 3.14159265358979323846

OUTPUTS = ("drydp", "dvol", "md_out", "mdt_out_a", "mdt_out_b")
INPUTS = ("nd", "md_in", "mdt_in_a", "mdt_in_b")


@pytest.fixture(scope="module")
def data():
    assert ARCHIVE.is_file(), "run `python validation/capture_drydiam_leaf.py`"
    with np.load(ARCHIVE, allow_pickle=False) as archive:
        yield {name: archive[name] for name in archive.files}


@pytest.fixture(scope="module")
def mode_tables():
    """The Fortran's own tables, from the phase-C golden.

    Deliberately not `glomap_jax.physics.modes.build`: every closed form below
    is checked against numbers the Fortran produced, so a shared mistake in the
    port cannot make both sides agree.
    """
    assert MODE_TABLES.is_file(), "run `python validation/capture_modes.py`"
    with np.load(MODE_TABLES, allow_pickle=False) as archive:
        yield {name: archive[name] for name in archive.files}


def table(mode_tables, setup, combo, field):
    prefix = f"s{setup}_" if combo == "default" else f"v_{combo}_s{setup}_"
    return mode_tables[prefix + field]


def bits(array):
    """Bit patterns, so -0.0 and a passed-through NaN compare as themselves."""
    return np.ascontiguousarray(array, dtype=np.float64).view(np.int64)


def get(data, setup, combo, key):
    return data[f"s{setup}_{combo}_{key}"]


def ratio1(mode_tables, setup, combo):
    """`:195`. `mm(:)/(avogadro*rhocomp(:))`, over every component."""
    mm = table(mode_tables, setup, combo, "mm")
    rhocomp = table(mode_tables, setup, combo, "rhocomp")
    return mm / (AVOGADRO * rhocomp)


def mmid_dvol(mode_tables, setup, combo):
    """`:210` and `:225`. Two spellings of one product; both are exact."""
    mmid = table(mode_tables, setup, combo, "mmid")
    return MMSUL * mmid / (AVOGADRO * RHO_SO4)


def masks(data, mode_tables, setup, combo):
    """`mode`, `mask` (`:206`) and `reset`, per (row, mode)."""
    mode = table(mode_tables, setup, combo, "mode").astype(bool)
    num_eps = table(mode_tables, setup, combo, "num_eps")
    nd = get(data, setup, combo, "nd")
    mask = (nd > num_eps[None, :]) & mode[None, :]
    reset = get(data, setup, combo, "reset").astype(bool)
    return mode, mask, reset


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def test_every_configuration_is_present(data):
    """Seven setups x five combinations. A configuration that quietly vanished
    would take its whole half of the coverage with it and nothing else here
    would notice, because every test below is parametrised over what exists."""
    assert list(data["_setups"]) == list(SETUPS)
    assert [str(c) for c in data["_combos"]] == list(COMBOS)
    assert str(data["_case"]) == "drydiam"
    assert str(data["_mode"]) == "leaf"
    assert str(data["_variant"]) == "f64"
    for setup, combo in CONFIGS:
        for key in INPUTS + OUTPUTS + ("reset", "tags", "hits"):
            assert f"s{setup}_{combo}_{key}" in data, f"s{setup}_{combo}_{key}"


@pytest.mark.parametrize("setup,combo", CONFIGS, ids=IDS)
def test_shapes_agree_across_every_array_of_a_configuration(data, setup, combo):
    rows, nmodes = get(data, setup, combo, "nd").shape
    ncp = get(data, setup, combo, "md_in").shape[2]
    assert nmodes == 8
    assert rows >= 3, "the nbox axis has to be wide enough to mix branches within a mode"
    for key in ("mdt_in_a", "mdt_in_b", "drydp", "dvol", "mdt_out_a", "mdt_out_b", "reset"):
        assert get(data, setup, combo, key).shape == (rows, nmodes), key
    for key in ("md_in", "md_out"):
        assert get(data, setup, combo, key).shape == (rows, nmodes, ncp), key
    assert get(data, setup, combo, "tags").shape == (rows, nmodes)


@pytest.mark.parametrize("setup,combo", CONFIGS, ids=IDS)
def test_only_the_mdt_columns_carry_anything_but_a_finite_number(data, setup, combo):
    """`drydp`, `dvol` and `md` are physical outputs and must be finite.

    `mdt_in`/`mdt_out` are the deliberate garbage -- NaN, +-1e300, signed zero
    -- and are the only arrays allowed to be otherwise. Stated as an exclusion
    rather than a global finiteness check, so a NaN leaking out of `dvol` is a
    failure rather than something a blanket `nan_to_num` would absorb.
    """
    for key in ("drydp", "dvol", "md_out", "md_in", "nd"):
        values = get(data, setup, combo, key)
        assert np.isfinite(values).all(), f"{key} is not finite"
    assert (get(data, setup, combo, "drydp") > 0).all(), "drydp <= 0 would have ereported"
    assert (get(data, setup, combo, "dvol") > 0).all(), "dvol <= 0 would have ereported"


# --------------------------------------------------------------------------
# Coverage: the hit counts, and the reasons for the zeros
# --------------------------------------------------------------------------


@pytest.mark.parametrize("setup,combo", CONFIGS, ids=IDS)
def test_every_predicate_is_reached_or_is_zero_for_a_recorded_reason(data, setup, combo):
    """The replacement for "shape+finiteness", which an all-zeros archive passes.

    Fourteen counts, one per arm of the seven predicates in the routine. The
    mutation this is answerable to: delete the reset straddle from
    `mode_cases`, and `reset_fired` falls to 0 for the six setups with active
    modes 1-3 -- the capture then refuses to write, and if it wrote anyway this
    would fail.
    """
    predicates = [str(p) for p in data["_predicates"]]
    hits = dict(zip(predicates, get(data, setup, combo, "hits").tolist()))
    reasons = json.loads(str(data["_expected_zero"])).get(f"s{setup}", {})
    for name, count in hits.items():
        if count == 0:
            assert name in reasons, f"{name} was never reached and no reason is recorded"
        else:
            assert name not in reasons, f"{name} is recorded as expected-zero but fired {count}"


@pytest.mark.parametrize("setup", SETUPS)
def test_the_recorded_zeros_are_the_ones_mode_choice_forces(data, mode_tables, setup):
    """The reasons are re-derived rather than trusted.

    Both arms of `:246` are decided by `mode_choice` and nothing else: a setup
    with all of modes 1-3 on never takes the FALSE arm, and a setup with none of
    them on never takes the TRUE arm or anything downstream of it. Between the
    seven setups both arms are covered, which is the claim, and this is what
    would notice if a vendored update turned one of them off.
    """
    mode = table(mode_tables, setup, "default", "mode").astype(bool)
    active = [i for i in RESET_MODES if mode[i]]
    reasons = json.loads(str(data["_expected_zero"])).get(f"s{setup}", {})
    if len(active) == len(RESET_MODES):
        assert set(reasons) == {"reset_mode_false"}
    elif not active:
        assert set(reasons) == {
            "reset_mode_true",
            "reset_fired",
            "reset_not_fired",
            "reset_component_true",
            "reset_component_false",
        }
    else:
        assert reasons == {}
    assert all("mode_choice" in text for text in reasons.values())


def test_the_reset_fires_where_no_committed_trajectory_reaches(data, mode_tables):
    """The headline, and the reason this archive exists.

    `test_the_reset_is_unreachable_from_every_shipped_namelist` below measures
    the other half from the branch archives: the `drydiam`/`undersize` predicate
    is recorded 3456 times across the four committed cases and is 0 every time.
    This archive is the only reference for the arm that fires, so if the count
    here ever returns to zero task 37 has nothing to port against.
    """
    predicates = [str(p) for p in data["_predicates"]]
    index = predicates.index("reset_fired")
    fired = {(s, c): int(get(data, s, c, "hits")[index]) for s, c in CONFIGS}
    dead = {k for k, n in fired.items() if n == 0}
    expected_dead = {
        (s, c)
        for s, c in CONFIGS
        if not table(mode_tables, s, "default", "mode").astype(bool)[list(RESET_MODES)].any()
    }
    assert dead == expected_dead, "the reset fired in a setup with modes 1-3 off, or not at all"
    assert sum(fired.values()) > 0
    assert min(n for k, n in fired.items() if k not in dead) >= 3


def test_the_reset_is_unreachable_from_every_shipped_namelist():
    """The motivating measurement, re-derived from the committed branch dumps.

    `undersize` is `dp < ddplim0*0.1` at `:250`, recorded per (call, mode, box)
    for modes 1-3. Across `bl_nmts3` (1296 records) and the other three cases
    (720 each) it is **3456 records and 0 hits**. The neighbouring `nd_gt_eps`
    predicate at `:206` is recorded 4848 times and goes both ways -- 3934 true,
    914 false -- so the archives are not simply missing `drydiam` coverage; it
    is this one predicate that no trajectory reaches.

    Asserted rather than written in prose so the claim this whole fixture rests
    on moves only when someone re-measures. If a future namelist did reach the
    reset, that is a finding about the trajectories and this is where it
    surfaces.
    """
    cases = ["bl_nmts3", "boundary_layer", "free_troposphere", "marine_bcoc"]
    totals = {"undersize": [0, 0], "nd_gt_eps": [0, 0]}
    for case in cases:
        with np.load(GOLDENS / f"{case}.f64.branches.npz", allow_pickle=False) as b:
            sites = [str(x) for x in b["site_levels"]]
            tags = [str(x) for x in b["tag_levels"]]
            drydiam = b["site"] == sites.index("drydiam")
            for tag, counts in totals.items():
                rows = drydiam & (b["tag"] == tags.index(tag))
                counts[0] += int(rows.sum())
                counts[1] += int((b["value"][rows] == 1).sum())
    assert totals["undersize"] == [3456, 0], (
        f"the undersize predicate is now {totals['undersize'][1]} of "
        f"{totals['undersize'][0]} -- a shipped namelist reaches the reset, which "
        "changes what this fixture is for"
    )
    assert totals["nd_gt_eps"] == [4848, 3934]


def test_setup_six_is_dead_because_of_mode_choice_and_nothing_else(data, mode_tables):
    """The control, and the reason it is the right control.

    `ddplim0`, `num_eps` and `mfrac_0` are byte-identical between setup 6 and
    setup 1, so a port that decided the reset was unreachable from any of those
    three would pass setup 6 for the wrong reason. What differs is
    `mode_choice`, and only that.
    """
    for field in ("ddplim0", "num_eps", "mfrac_0"):
        np.testing.assert_array_equal(
            table(mode_tables, 6, "default", field),
            table(mode_tables, 1, "default", field),
            err_msg=f"{field} now differs between setups 1 and 6",
        )
    mode6 = table(mode_tables, 6, "default", "mode").astype(bool)
    mode1 = table(mode_tables, 1, "default", "mode").astype(bool)
    assert not mode6[list(RESET_MODES)].any()
    assert mode1[list(RESET_MODES)].all()
    for combo in COMBOS:
        assert get(data, 6, combo, "reset").sum() == 0


# --------------------------------------------------------------------------
# The closed forms, recomputed from the Fortran's own tables
# --------------------------------------------------------------------------


@pytest.mark.parametrize("setup,combo", CONFIGS, ids=IDS)
def test_the_mmid_arm_is_exact_for_inactive_and_mask_false_slots(data, mode_tables, setup, combo):
    """`:210` and `:225`: `dvol = mmsul*mmid/(avogadro*rho_so4)`.

    Two multiplications and a division, no libm, so byte equality is the right
    comparison on every platform. Both arms are checked in one place because
    they compute the same number by different routes, and a port that used
    `mlo` or `mhi` for one of them would produce a plausible column.
    """
    mode, mask, _ = masks(data, mode_tables, setup, combo)
    dvol = get(data, setup, combo, "dvol")
    expected = mmid_dvol(mode_tables, setup, combo)
    rows = ~mask  # inactive slots, plus active slots below num_eps
    assert rows.any()
    np.testing.assert_array_equal(dvol[rows], np.broadcast_to(expected[None, :], dvol.shape)[rows])
    # Both populations non-empty, or the assertion above is half a test: a
    # setup with no mask-false row would satisfy it on the inactive slots alone.
    assert (~mode).any(), "no inactive mode, so the :225 arm is untested here"
    assert (mode[None, :] & ~mask).any(), "no mask-false row, so the :210 arm is untested"


@pytest.mark.parametrize("setup,combo", CONFIGS, ids=IDS)
def test_the_member_sum_is_exact_where_the_mask_is_on(data, mode_tables, setup, combo):
    """`:214-222`: `dvol = sum over members of ratio1(icp)*md(icp)`, from 0.0.

    Accumulated in ascending `icp` from a zeroed `dvol`, which is what the
    Fortran does. Exact on any IEEE 754 host -- the reference was built with
    `-ffp-contract=off`, so there is no FMA to disagree about -- hence byte
    equality rather than a tolerance.

    Restricted to slots the reset did not overwrite, and the non-member terms
    are excluded from the sum on purpose: if `:215` were ignored, the polluted
    rows would come out larger and this is what would say so.
    """
    _, mask, reset = masks(data, mode_tables, setup, combo)
    component = table(mode_tables, setup, combo, "component").astype(bool)
    md = get(data, setup, combo, "md_in")
    r1 = ratio1(mode_tables, setup, combo)
    dvol = get(data, setup, combo, "dvol")

    expected = np.zeros_like(dvol)
    for imode in range(dvol.shape[1]):
        for icp in range(md.shape[2]):
            if component[imode, icp]:
                expected[:, imode] = expected[:, imode] + r1[icp] * md[:, imode, icp]

    rows = mask & ~reset
    assert rows.any()
    np.testing.assert_array_equal(dvol[rows], expected[rows])


@pytest.mark.parametrize("setup,combo", CONFIGS, ids=IDS)
def test_a_non_member_component_never_reaches_dvol(data, mode_tables, setup, combo):
    """`:215` again, stated as the thing that would break if it were dropped.

    The grid carries rows with a large `md` in a component that
    `component(imode,icp)` is FALSE for. `dvol` on those rows must equal the
    same row with that component zeroed -- which is what the member sum above
    computes, so this only has to show such rows exist and are mask-true.
    """
    tags = get(data, setup, combo, "tags")
    _, mask, _ = masks(data, mode_tables, setup, combo)
    polluted = np.char.startswith(tags, "nonmember/") & mask
    assert polluted.any(), "no mask-true row carries non-member mass"
    component = table(mode_tables, setup, combo, "component").astype(bool)
    md = get(data, setup, combo, "md_in")
    for row, imode in np.argwhere(polluted):
        carried = [c for c in range(md.shape[2]) if md[row, imode, c] != 0.0]
        assert any(not component[imode, c] for c in carried), (row, imode, carried)


@pytest.mark.parametrize("setup,combo", CONFIGS, ids=IDS)
def test_drydp_is_the_cube_root_of_sixovrpix_times_dvol(data, mode_tables, setup, combo):
    """`:230-237` and `:258`, the only place in the routine that reaches libm.

    `cubrt_v` is literally `x ** (1.0/3.0)`, not a cube root function, so that
    is the expression compared. Byte equality on the capture platform and a
    measured ulp window elsewhere -- `x ** (1.0/3.0)` differs by up to 1 ulp
    between Darwin arm64 and ubuntu x86_64.
    """
    x = table(mode_tables, setup, combo, "x")
    sixovrpix = 6.0 / (PI * x)
    dvol = get(data, setup, combo, "dvol")
    expected = (sixovrpix[None, :] * dvol) ** (1.0 / 3.0)
    assert_matches_reference(
        get(data, setup, combo, "drydp"), expected, f"drydp s{setup} {combo}", ulp=1
    )


# --------------------------------------------------------------------------
# The reset: everything it writes, and the one thing it must not
# --------------------------------------------------------------------------


@pytest.mark.parametrize("setup,combo", CONFIGS, ids=IDS)
def test_the_reset_rewrites_md_mdt_and_dvol_to_the_mlo_forms(data, mode_tables, setup, combo):
    """`:251-257`, all three writes, on the slots where the reset fired."""
    _, _, reset = masks(data, mode_tables, setup, combo)
    if not reset.any():
        pytest.skip("modes 1-3 are inactive in this setup")
    mlo = table(mode_tables, setup, combo, "mlo")
    mfrac_0 = table(mode_tables, setup, combo, "mfrac_0")
    component = table(mode_tables, setup, combo, "component").astype(bool)
    md_out = get(data, setup, combo, "md_out")
    dvol = get(data, setup, combo, "dvol")

    for row, imode in np.argwhere(reset):
        assert imode in RESET_MODES
        assert get(data, setup, combo, "mdt_out_a")[row, imode] == mlo[imode]
        assert get(data, setup, combo, "mdt_out_b")[row, imode] == mlo[imode]
        assert dvol[row, imode] == MMSUL * mlo[imode] / (AVOGADRO * RHO_SO4)
        for icp in range(md_out.shape[2]):
            if component[imode, icp]:
                assert md_out[row, imode, icp] == mlo[imode] * mfrac_0[imode, icp]


@pytest.mark.parametrize("setup,combo", CONFIGS, ids=IDS)
def test_the_reset_leaves_non_member_components_alone(data, mode_tables, setup, combo):
    """`:252` guards the rewrite on `component`, so a non-member keeps its `md`.

    The grid carries a `reset/nonmember/*` row precisely so this is not vacuous:
    on every other reset row the non-member slots are zero, and "unchanged" and
    "overwritten with `mlo*mfrac_0` = 0" are then the same number.
    """
    _, _, reset = masks(data, mode_tables, setup, combo)
    if not reset.any():
        pytest.skip("modes 1-3 are inactive in this setup")
    component = table(mode_tables, setup, combo, "component").astype(bool)
    md_in = get(data, setup, combo, "md_in")
    md_out = get(data, setup, combo, "md_out")

    witnessed = 0
    for row, imode in np.argwhere(reset):
        for icp in range(md_in.shape[2]):
            if component[imode, icp]:
                continue
            assert bits(md_out[row, imode, icp]) == bits(md_in[row, imode, icp])
            witnessed += md_in[row, imode, icp] != 0.0
    assert witnessed, (
        "no reset row carries a non-zero non-member md, so 'not overwritten' and "
        "'overwritten with mlo*mfrac_0 = 0' are indistinguishable here"
    )


@pytest.mark.parametrize("setup,combo", CONFIGS, ids=IDS)
def test_md_and_mdt_are_untouched_wherever_the_reset_did_not_fire(data, mode_tables, setup, combo):
    """The other half of `INTENT(IN OUT)`: outside `:250` nothing is written."""
    _, _, reset = masks(data, mode_tables, setup, combo)
    md_in, md_out = get(data, setup, combo, "md_in"), get(data, setup, combo, "md_out")
    quiet = ~reset
    assert quiet.any()
    np.testing.assert_array_equal(bits(md_out)[quiet], bits(md_in)[quiet])
    for suffix in ("a", "b"):
        got = get(data, setup, combo, f"mdt_out_{suffix}")
        want = get(data, setup, combo, f"mdt_in_{suffix}")
        np.testing.assert_array_equal(bits(got)[quiet], bits(want)[quiet])


@pytest.mark.parametrize("setup,combo", CONFIGS, ids=IDS)
def test_mdt_is_written_and_never_read(data, setup, combo):
    """`mdt` appears at `:40`, `:135`, `:243` and `:256` and nowhere else.

    The capture ran the whole grid twice with disjoint `mdt` and counted the
    output elements that moved; zero is the claim. The disjointness is
    re-checked here, because a demonstration run with two identical garbage
    arrays would report zero too and prove nothing.
    """
    a = get(data, setup, combo, "mdt_in_a")
    b = get(data, setup, combo, "mdt_in_b")
    assert not (bits(a) == bits(b)).any(), "the two mdt garbage sets overlap somewhere"
    assert not (np.isnan(a) & np.isnan(b)).any(), "both garbage sets are NaN at the same slot"
    assert int(get(data, setup, combo, "mdt_mismatch")) == 0


@pytest.mark.parametrize("setup,combo", CONFIGS, ids=IDS)
def test_the_batched_call_equals_the_row_by_row_calls(data, setup, combo):
    """`:265-267` reduces over the whole `nbox` extent, so a batched call and a
    stack of one-row calls are not obviously the same computation. The capture
    ran both and compared bit patterns; this is where the answer is asserted."""
    assert int(get(data, setup, combo, "nbox1_mismatch")) == 0


@pytest.mark.parametrize("setup,combo", CONFIGS, ids=IDS)
def test_no_mask_false_row_ever_reaches_the_reset(data, mode_tables, setup, combo):
    """Mask-false rows get `drydp` from `mmid`, which is 22x to 32x above
    `0.1*ddplim0` for modes 1-3 -- so the reset is unreachable from that arm and
    only a mask-true row with a tiny `md` can get there. Measured, not assumed:
    the margin is recomputed here from the tables."""
    mode, mask, reset = masks(data, mode_tables, setup, combo)
    ddplim0 = table(mode_tables, setup, combo, "ddplim0")
    drydp = get(data, setup, combo, "drydp")
    for imode in RESET_MODES:
        if not mode[imode]:
            continue
        rows = ~mask[:, imode]
        assert rows.any()
        assert not reset[rows, imode].any()
        assert (drydp[rows, imode] / (0.1 * ddplim0[imode]) > 20.0).all()


@pytest.mark.parametrize("setup,combo", CONFIGS, ids=IDS)
def test_the_grid_mixes_branches_within_one_mode(data, mode_tables, setup, combo):
    """A port that decided a whole `nbox` column at once would still reproduce a
    fixture in which every row of a mode took the same branch. This is what
    makes the fixture able to tell the difference."""
    mode, mask, reset = masks(data, mode_tables, setup, combo)
    for imode in range(mask.shape[1]):
        if not mode[imode]:
            continue
        assert mask[:, imode].any() and not mask[:, imode].all(), f"mode {imode + 1} mask"
    for imode in RESET_MODES:
        if not mode[imode]:
            continue
        assert reset[:, imode].any() and not reset[:, imode].all(), f"mode {imode + 1} reset"


@pytest.mark.parametrize("setup,combo", CONFIGS, ids=IDS)
def test_nd_exactly_equal_to_num_eps_takes_the_mmid_arm(data, mode_tables, setup, combo):
    """`:206` is `nd > num_eps`, strictly.

    `nd == num_eps` must go the `mmid` way and the next representable value
    above it the other way. Both rows are in the grid, so a port written with
    `>=` fails here rather than passing on a grid that only brackets loosely.
    """
    mode = table(mode_tables, setup, combo, "mode").astype(bool)
    num_eps = table(mode_tables, setup, combo, "num_eps")
    nd = get(data, setup, combo, "nd")
    dvol = get(data, setup, combo, "dvol")
    expected = mmid_dvol(mode_tables, setup, combo)
    seen = 0
    for imode in range(nd.shape[1]):
        if not mode[imode]:
            continue
        at = nd[:, imode] == num_eps[imode]
        above = nd[:, imode] == np.nextafter(num_eps[imode], np.inf)
        assert at.any() and above.any(), f"mode {imode + 1} does not straddle num_eps"
        np.testing.assert_array_equal(dvol[at, imode], np.full(int(at.sum()), expected[imode]))
        assert (dvol[above, imode] != expected[imode]).all()
        seen += 1
    assert seen


# --------------------------------------------------------------------------
# The deliberate abort
# --------------------------------------------------------------------------


@pytest.mark.parametrize("setup,combo", CONFIGS, ids=IDS)
def test_the_abort_block_moved_the_fatal_counter_by_exactly_one_per_row(data, setup, combo):
    """`:313-315`, compared on the ereport record and not on the numbers.

    Exactly one, not "at least one": every other mode in the row is left
    mask-false with `dvol = mmsul*mmid/(avogadro*rho_so4) > 0`, so a second
    fatal would mean a mode aborted that was not meant to. Two rows, because
    `:268` is a disjunction and `dvol = 0` only reaches its first arm.
    """
    fatal = get(data, setup, combo, "abort_fatal")
    tags = [str(t) for t in get(data, setup, combo, "abort_tags")]
    messages = [str(m) for m in get(data, setup, combo, "abort_message")]
    assert len(fatal) == 2
    assert list(fatal) == [1, 1]
    assert all("dvol or drydp <= 0" in m for m in messages)
    assert [t.split("/")[0] for t in tags] == ["dvol_zero", "dvol_negative"]
    for tag in tags:
        imode = int(tag.split("/")[1].removeprefix("mode"))
        assert imode > max(RESET_MODES) + 1, (
            f"{tag}: modes 1-3 cannot abort -- the reset restores dvol before :268"
        )


@pytest.mark.parametrize("setup,combo", CONFIGS, ids=IDS)
def test_modes_one_to_three_escape_the_abort_the_others_cannot(data, mode_tables, setup, combo):
    """The `dvol = 0` corner, which is a reset for modes 1-3 and a fatal above.

    The grid carries a mask-true row with every member `md = 0` for each active
    mode 1-3. That row has `dvol = 0` and `drydp = 0` going into `:250`, and the
    only reason the whole call did not ereport is that `:257` put `dvol` back.
    The archive exists, so the call returned; what this checks is that the row
    is there and that it is the reset that saved it.
    """
    mode, _, reset = masks(data, mode_tables, setup, combo)
    tags = get(data, setup, combo, "tags")
    mlo = table(mode_tables, setup, combo, "mlo")
    dvol = get(data, setup, combo, "dvol")
    seen = 0
    for imode in RESET_MODES:
        if not mode[imode]:
            continue
        rows = np.flatnonzero(tags[:, imode] == "reset/md0")
        assert rows.size == 1, f"mode {imode + 1} has no zero-md mask-true row"
        row = int(rows[0])
        assert get(data, setup, combo, "md_in")[row, imode].max() == 0.0
        assert reset[row, imode]
        assert dvol[row, imode] == MMSUL * mlo[imode] / (AVOGADRO * RHO_SO4) > 0.0
        seen += 1
    if not seen:
        pytest.skip("modes 1-3 are inactive in this setup")


# --------------------------------------------------------------------------
# Anti-collapse: the switches, and the setups
# --------------------------------------------------------------------------


@pytest.mark.parametrize("setup", SETUPS)
def test_the_controls_are_byte_identical_and_the_density_switches_are_not(data, setup):
    """`hygro_off` moves `no_ions` and `dust_ageing` moves `topmode`, and neither
    name occurs anywhere in `ukca_calc_drydiam.F90` -- so both must reproduce
    `default` exactly. `nacl_off` and `bc_mg_mix` move `rhocomp`, hence `ratio1`
    and `mmid`, and must not.

    Recorded as an expectation in both directions. An unexpected collision means
    a switch never reached the Fortran; a missing one means the routine grew a
    dependency it did not have.
    """
    for control in CONTROLS:
        for key in INPUTS + OUTPUTS + ("reset",):
            np.testing.assert_array_equal(
                bits(get(data, setup, control, key))
                if get(data, setup, control, key).dtype == np.float64
                else get(data, setup, control, key),
                bits(get(data, setup, "default", key))
                if get(data, setup, "default", key).dtype == np.float64
                else get(data, setup, "default", key),
                err_msg=f"setup {setup}: {control} moved {key}",
            )
    for switch in ("nacl_off", "bc_mg_mix"):
        moved = [
            key
            for key in OUTPUTS
            if not np.array_equal(
                bits(get(data, setup, switch, key)), bits(get(data, setup, "default", key))
            )
        ]
        assert moved, f"setup {setup}: {switch} changed nothing, so it never reached the Fortran"


def test_the_only_identical_records_are_the_twenty_one_recorded_controls(data):
    """`leaf_common.check_varied`, re-run against the committed archive.

    The capture refuses to write a collapsed golden; this is what would notice
    if one were committed anyway. Exactly 21 collisions are expected -- the
    three pairs among `default`, `hygro_off` and `dust_ageing`, for each of the
    seven setups -- and nothing else, in either direction.
    """
    prints = {
        (s, c): hashlib.sha256(
            b"".join(bits(get(data, s, c, key)).tobytes() for key in OUTPUTS)
        ).hexdigest()
        for s, c in CONFIGS
    }
    collided = {
        frozenset({a, b})
        for i, a in enumerate(CONFIGS)
        for b in CONFIGS[i + 1 :]
        if prints[a] == prints[b]
    }
    identical = ("default", *CONTROLS)
    expected = {
        frozenset({(s, a), (s, b)})
        for s in SETUPS
        for i, a in enumerate(identical)
        for b in identical[i + 1 :]
    }
    assert len(expected) == 21
    assert collided == expected, (
        f"unexpected {sorted(map(sorted, collided - expected))}, "
        f"missing {sorted(map(sorted, expected - collided))}"
    )


@pytest.mark.parametrize("setup,combo", CONFIGS, ids=IDS)
def test_the_archive_describes_the_setup_it_is_filed_under(data, mode_tables, setup, combo):
    """Cross-checked against a golden the Fortran produced in a different run.

    The load-bearing guard against "the setup never took" is in the capture: the
    child reads `mode`, `component`, `mmid`, `mlo`, `x`, `mm`, `rhocomp`,
    `num_eps`, `ddplim0` and `mfrac_0` back out of the Fortran and refuses to
    return a record unless they are byte-identical to the tables the grid was
    derived from. That cannot run in CI, which has no toolchain, so this is the
    half that can: the modes the archive treats as active, and the components it
    treats as members, must be the ones `modes.f64.tables.npz` records for this
    exact (setup, combination).

    `component` differs for all 21 setup pairs (see `capture_modes`), so a
    record filed under the wrong setup fails here.
    """
    mode = table(mode_tables, setup, combo, "mode").astype(bool)
    component = table(mode_tables, setup, combo, "component").astype(bool)
    tags = get(data, setup, combo, "tags")
    dvol = get(data, setup, combo, "dvol")

    inactive = np.flatnonzero(~mode)
    for imode in inactive:
        assert len(set(dvol[:, imode].tolist())) == 1, (
            f"mode {imode + 1} is inactive but its dvol column varies with the row"
        )
        assert (np.char.startswith(tags[:, imode], "inactive/")).all()
    for imode in np.flatnonzero(mode):
        members = {
            int(t.split("/")[1].removeprefix("cp")) for t in tags[:, imode] if t.startswith("pure/")
        }
        assert members == set(np.flatnonzero(component[imode]).tolist()), (
            f"mode {imode + 1}: the grid treats {sorted(members)} as members, the mode "
            f"tables say {np.flatnonzero(component[imode]).tolist()}"
        )
