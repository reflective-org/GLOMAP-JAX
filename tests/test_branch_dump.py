"""Task 15b: the predicates the science branches on (validation gate 0).

This code's dominant divergence mode is not precision drift. It is discrete
branches on computed floats: about ten sites compare a float against a
threshold and then select a *different closed form*. Two individually correct
float64 implementations that disagree at one of those comparisons produce an
O(1) trajectory difference, and no tolerance on the trajectory can say which
comparison it was. So the reference dumps the predicates themselves.

The fixture is `marine_bcoc` (i_mode_setup=2), the smallest shipped case that
has both soluble and insoluble modes and therefore reaches every instrumented
site. `boundary_layer` and `free_troposphere` are setup 1, four soluble modes,
and never enter `ukca_coagwithnucl`'s insoluble blocks at all.

Counts below reconcile structurally. With nsteps=3, nmts=1, nzts=15 and nbox=1
there are 45 substeps; setup 2 has 5 active modes (4 soluble + Aitken
insoluble), 2 condensable gases, and 15 (mode, component) pairs.

One count is not what the splitting diagram predicts and is worth knowing:
`ukca_calc_drydiam` runs FIVE times per chemistry step, not four. The fifth is
`glomap_box_state_mod`'s `update_size`, outside `ukca_aero_step` entirely. Those
records carry imts = izts = -1 so they cannot be mistaken for the tail of the
step that just ran.

What the shipped fixtures do NOT reach, which is itself a finding: of
`ukca_solvecoagnucl_v`'s eight branch codes only 0, 1 and 5 occur; the TAN
branch, both A == 0 branches and the error branch are unexercised, as are the
MDCPNEW < 0 reset, the undersize diameter reset and every mode merge. Those need
the synthetic branch sweep of task 64 -- they cannot be validated from a
trajectory fixture, because a trajectory fixture never visits them.
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
SUBSTEPS = NSTEPS * NZTS  # nmts = 1 in every shipped namelist

# (site, tag) -> record count, for i_mode_setup = 2.
EXPECTED = {
    # --- ukca_solvecoagnucl_v, via its two call sites -----------------------
    ("coag_sol_solve", "form"): 4 * SUBSTEPS,  # 4 soluble modes
    ("coag_sol_solve", "mask"): 4 * SUBSTEPS,
    ("coag_sol_solve", "logic3"): 4 * SUBSTEPS,
    ("coag_sol_solve", "sqd_clamp"): 4 * SUBSTEPS,
    ("coag_sol_solve", "tan_pole"): 4 * SUBSTEPS,
    ("coag_sol_solve", "mask1a"): 4 * SUBSTEPS,
    ("coag_insol_solve", "form"): 1 * SUBSTEPS,  # 1 insoluble mode
    ("coag_insol_solve", "mask"): 1 * SUBSTEPS,
    ("coag_insol_solve", "logic3"): 1 * SUBSTEPS,
    ("coag_insol_solve", "sqd_clamp"): 1 * SUBSTEPS,
    ("coag_insol_solve", "tan_pole"): 1 * SUBSTEPS,
    # --- ukca_coagwithnucl --------------------------------------------------
    ("coag_sol", "mask1"): 4 * SUBSTEPS,
    ("coag_insol", "mask1"): 1 * SUBSTEPS,
    ("coag_sol_sol", "mask2"): 6 * SUBSTEPS,  # unordered pairs of 4 modes
    ("coag_sol_sol", "mask4"): 6 * SUBSTEPS,
    ("coag_sol_insol", "mask2"): 1 * SUBSTEPS,
    ("coag_sol_insol", "mask4"): 1 * SUBSTEPS,
    ("coag_insol_insol", "mask2"): 2 * SUBSTEPS,
    ("coag_insol_insol", "mask4"): 2 * SUBSTEPS,
    ("coag_reset", "mask1_entry"): 5 * SUBSTEPS,  # every active mode
    ("coag_reset", "mdcp_neg"): 15 * SUBSTEPS,  # (mode, component) pairs
    ("coag_reset", "mask1_after"): 15 * SUBSTEPS,
    ("coag_reset", "mask3"): 15 * SUBSTEPS,
    # --- ukca_conden --------------------------------------------------------
    ("conden", "mask1"): 2 * SUBSTEPS,  # 2 condensable gases
    ("conden_mode", "mask2"): 2 * 5 * SUBSTEPS,
    ("conden_uptake", "mask2"): 2 * SUBSTEPS,
    ("conden_uptake", "up4_guard"): 2 * SUBSTEPS,
    ("conden_dist", "mask3"): 2 * 4 * SUBSTEPS,  # soluble modes only
    ("conden_dist", "mask3i"): 2 * 4 * SUBSTEPS,
    ("conden_dist", "mask4i"): 2 * 4 * SUBSTEPS,
    # --- ukca_calcnucrate / ukca_binapara -----------------------------------
    ("nucrate", "l1"): SUBSTEPS,
    ("nucrate", "l2"): SUBSTEPS,
    ("nucrate", "veh_guard"): SUBSTEPS,
    ("binapara", "ntot_lt_4"): SUBSTEPS,
    ("binapara", "t_lt_195"): SUBSTEPS,
    # --- ukca_calc_drydiam: 4 calls in aero_step + 1 in the box driver ------
    ("drydiam", "nd_gt_eps"): 5 * 5 * NSTEPS,
    ("drydiam", "undersize"): 3 * 5 * NSTEPS,  # modes 1-3 only
    # --- ukca_remode: 2 calls per step, 3 mergeable modes -------------------
    ("remode", "nmodemax_merge"): 2 * NSTEPS,
    ("remode", "merge_trigger"): 3 * 2 * NSTEPS,
    ("remode", "nd_gt_eps"): 3 * 2 * NSTEPS,
}

# Predicates that only fire deeper inside a guard the shipped fixtures may or
# may not open. Presence is case-dependent, so they are excluded from the
# exhaustiveness check rather than counted.
CONDITIONAL = {
    ("nucrate", "jkul_gt"),
    ("nucrate", "japp_gt"),
    ("nucrate", "japp_bln_gt"),
    ("remode", "frac_n_clamp"),
    ("remode", "frac_m_clamp"),
    ("remode", "newn_gt_eps"),
}


def _run(tmp_path, namelist, branch=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    exe = FORTRAN / "bin-ref-f64" / "glomap_box"
    if not exe.is_file():
        pytest.skip("reference not built; run validation/build_reference.sh")
    out, brn, nml = tmp_path / "o.csv", tmp_path / "b.csv", tmp_path / "o.nml"
    text = (FORTRAN / "namelists" / f"{namelist}.nml").read_text(encoding="utf-8")
    text = text.replace(f"out/{namelist}.csv", str(out))
    text = text.replace("  nsteps       = 48", f"  nsteps       = {NSTEPS}")
    if branch:
        text = text.replace("  verbose      = 0", f"  verbose      = 0\n  branch_file  = '{brn}'")
    nml.write_text(text, encoding="utf-8")
    subprocess.run([str(exe), str(nml)], check=True, capture_output=True)
    return out, brn


@pytest.fixture(scope="module")
def dump(tmp_path_factory):
    _, brn = _run(tmp_path_factory.mktemp("bd"), "marine_bcoc")
    return list(csv.DictReader(brn.open()))


@needs_gfortran
def test_every_instrumented_predicate_is_present(dump):
    seen = {(r["site"], r["tag"]) for r in dump}
    assert seen - CONDITIONAL == set(EXPECTED)


@needs_gfortran
@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_each_predicate_fires_the_expected_number_of_times(dump, key):
    """A count that moves means a loop trip count or call site changed, which is
    a splitting-order change and must not pass silently."""
    site, tag = key
    actual = sum(1 for r in dump if r["site"] == site and r["tag"] == tag)
    assert actual == EXPECTED[key], f"{site}/{tag}: {actual} records, expected {EXPECTED[key]}"


@needs_gfortran
def test_masks_are_boolean_and_forms_are_in_range(dump):
    for r in dump:
        v = int(r["value"])
        if r["tag"] == "form":
            assert 0 <= v <= 7, r
        elif r["tag"] == "nmodemax_merge":
            assert v in (2, 3), r
        else:
            assert v in (0, 1), r


@needs_gfortran
def test_every_substep_is_individually_resolved(dump):
    izts = {int(r["izts"]) for r in dump if r["site"] == "conden"}
    assert izts == set(range(1, NZTS + 1))


@needs_gfortran
def test_driver_side_drydiam_is_distinguishable_from_the_aero_step_calls(dump):
    """update_size calls calc_drydiam outside ukca_aero_step. Without a marker
    those records would be attributed to the last substep of the previous step."""
    imts = {r["imts"] for r in dump if r["site"] == "drydiam"}
    assert "-1" in imts, "the box driver's calc_drydiam call is not marked"


@needs_gfortran
def test_up1_factor_three_branch_runs_every_substep(dump):
    """UP-1: `1/(1/N - 3*A*dt)` where the exact integral of dN/dt = A*N^2 has no
    factor 3. Branch code 5. For the top soluble mode there is no larger soluble
    mode to coagulate with and no nucleation source, so B and C are exactly zero
    and D is exactly zero -- every substep, in a default configuration.

    This is stronger than the plan's reachability claim, which named only the
    top *insoluble* modes. The fidelity flag for UP-1 must therefore default to
    reproducing the defect, or gate C fails on every fixture."""
    forms = Counter(
        (r["i1"], int(r["value"]))
        for r in dump
        if r["site"] == "coag_sol_solve" and r["tag"] == "form"
    )
    assert forms[("4", 5)] == SUBSTEPS, f"coarse soluble mode not always on branch 5: {forms}"


@needs_gfortran
def test_up4_guard_never_fires(dump):
    """UP-4 is argued unreachable: DELGC_COND = GC*(1 - EXP(-x)) with x >= 0 is
    bounded in [0, GC], so the `> GC` correction cannot trigger. Dumping the
    guard turns that argument into an observation, which is why no fidelity flag
    is warranted for it -- an invariant test is the right shape."""
    assert all(int(r["value"]) == 0 for r in dump if r["tag"] == "up4_guard")


@needs_gfortran
def test_coag_reset_mask_only_ever_narrows(dump):
    """ukca_coagwithnucl mutates mask1 inside a WHERE and carries the narrowed
    result into the next component. A port that computes all components from the
    entry mask gets a different MDT. The dump has to make that visible, so the
    after-mask must be a subset of the entry mask for every (mode, substep)."""
    entry = {
        (r["step"], r["imts"], r["izts"], r["i1"]): int(r["value"])
        for r in dump
        if r["site"] == "coag_reset" and r["tag"] == "mask1_entry"
    }
    checked = 0
    for r in dump:
        if r["site"] == "coag_reset" and r["tag"] == "mask1_after":
            key = (r["step"], r["imts"], r["izts"], r["i1"])
            assert int(r["value"]) <= entry[key], f"mask1 widened at {key}"
            checked += 1
    assert checked == EXPECTED[("coag_reset", "mask1_after")]


@needs_gfortran
def test_dumping_branches_does_not_change_the_science(tmp_path):
    """The whole overlay is worthless if it perturbs the run it observes. Every
    hunk under src/ukca/ only reads variables that are already live, so the
    trajectory must be byte-identical with the dump on and off."""
    with_dump, _ = _run(tmp_path / "on", "marine_bcoc", branch=True)
    without, _ = _run(tmp_path / "off", "marine_bcoc", branch=False)
    assert with_dump.read_bytes() == without.read_bytes()


@pytest.fixture(scope="module")
def dump_setup1(tmp_path_factory):
    _, brn = _run(tmp_path_factory.mktemp("bd1"), "boundary_layer")
    return list(csv.DictReader(brn.open()))


@needs_gfortran
def test_setup_one_reaches_no_insoluble_sites(dump_setup1):
    """Recorded so the fixture choice above is not mistaken for arbitrary: the
    two setup-1 namelists cannot exercise the insoluble coagulation blocks."""
    sites = {r["site"] for r in dump_setup1}
    assert not {s for s in sites if "insol" in s}
