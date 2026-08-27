"""Task 55: `ukca_solvecoagnucl_v`, and a ninth branch nobody wrote down.

Against `tests/goldens/solvecoagnucl.f64.leaf.npz`: 114 rows x 4 timesteps,
plus a separate 12-row probe for the fatal error case.

**Byte-equal on 453 of the 456 live (row, timestep) pairs.** The three that
differ are all in branch `2a`, the one that computes `EXP(B·dtz)`, and
substituting libm's `exp` removes all three. Issue #28's fifth routine.

**`ATAN` and `TAN` are bit-identical to gfortran** on all 60 rows of branch
`1b` — two primitives the numerics table never measured, and the reason the
`1b` arm needs no allowance at all.

Issue #13 is what the fixture is for
------------------------------------

Four of the eight documented branch codes cannot be reached from a trajectory,
because a trajectory does not choose `A`, `B` and `C`. `1a_term4` is the
extreme case: it needs `EXP(sqd·dtz)/term3` to round to exactly `1.0`, a
tolerance four orders below the representable spacing, so the capture **solves**
for it rather than sampling.

The ninth branch
----------------

`:198-199` writes both `A` tests strictly against the same `eps_ab`, so
`|A| == eps_ab` exactly satisfies neither `logic1` nor `logic2`. Those rows take
no closed form: `ndnew` keeps its initialised `nd` and `deln` is exactly zero,
with no `ierr` and no diagnostic. `logic3` is inclusive, so `B` has no such
gap. Issue #31 — found by the capture, when the `2b` invariant fired on a row
the classifier called `2b` and the compiled routine had left alone.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import glomap_jax.physics.solvecoagnucl as sv
from glomap_jax.config.fidelity import FidelityConfig
from glomap_jax.core.constants import EPS_AB, SQD_CLAMP

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "solvecoagnucl.f64.leaf.npz"

sys.path.insert(0, str(REPO / "validation"))

import capture_solvecoagnucl_leaf as cap  # noqa: E402

#: Measured. Three rows, all in branch 2a, all through `exp`.
EXP_ULP = 2
EXP_AFFECTED = 3


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run validation/capture_solvecoagnucl_leaf.py (or `make goldens`)"
    return np.load(GOLDEN, allow_pickle=False)


@pytest.fixture(scope="module")
def inputs(sweep):
    return {
        "mask": jnp.asarray(sweep["mask"].astype(bool)),
        **{k: jnp.asarray(sweep[k]) for k in ("a", "b", "c", "nd")},
    }


def _run(inputs, dtz, **kw):
    return sv.solvecoagnucl(
        inputs["mask"], inputs["a"], inputs["b"], inputs["c"], inputs["nd"], dtz=dtz, **kw
    )


def _code(sweep, i, dtz):
    return cap.classify(
        float(sweep["a"][i]),
        float(sweep["b"][i]),
        float(sweep["c"][i]),
        float(sweep["nd"][i]),
        dtz,
    )


def test_the_port_matches_the_compiled_routine(sweep, inputs):
    differing = []
    for k, dtz in enumerate(sweep["dtz"].tolist()):
        got = np.asarray(_run(inputs, dtz).deln)
        want = sweep["deln"][0, k]
        for i in np.flatnonzero(got != want):
            differing.append((dtz, int(i), _code(sweep, i, dtz)))
            ulp = abs(got[i] - want[i]) / np.spacing(abs(want[i]))
            assert ulp <= EXP_ULP, f"row {i} at dtz={dtz}: {ulp:.1f} ulp"
    assert len(differing) == EXP_AFFECTED
    assert {c for _d, _i, c in differing} == {"2a"}, differing


def test_the_whole_gap_is_the_exponential(sweep, inputs, monkeypatch):
    class _Libm:
        def __getattr__(self, name):
            return getattr(jnp, name)

        @staticmethod
        def exp(x):
            return jnp.asarray(np.exp(np.asarray(x)))

    monkeypatch.setattr(sv, "jnp", _Libm())
    for k, dtz in enumerate(sweep["dtz"].tolist()):
        np.testing.assert_array_equal(np.asarray(_run(inputs, dtz).deln), sweep["deln"][0, k])


def test_atan_and_tan_are_bit_identical_to_gfortran(sweep, inputs):
    """Branch `1b` is the only consumer of either, and it is byte-equal on all
    60 of its rows with no substitution. Two primitives the numerics table
    never measured, and evidence that issue #28 is about `exp` specifically
    rather than about transcendentals in general."""
    rows = [
        (k, i)
        for k, dtz in enumerate(sweep["dtz"].tolist())
        for i in range(sweep["a"].size)
        if sweep["mask"][i] and _code(sweep, i, dtz) == "1b"
    ]
    assert len(rows) == 60
    for k, dtz in enumerate(sweep["dtz"].tolist()):
        got = np.asarray(_run(inputs, dtz).deln)
        idx = [i for kk, i in rows if kk == k]
        np.testing.assert_array_equal(got[idx], sweep["deln"][0, k][idx])


@pytest.mark.parametrize(
    "code", ("1a_ok", "1a_term3", "1a_term4", "1b", "1cb", "2a", "2b", "gap_a")
)
def test_every_branch_code_is_reached(sweep, code):
    """Issue #13: the shipped fixtures reach four of these. A constructed grid
    reaches all eight, and a port that got any of them wrong would otherwise
    pass every comparison."""
    hits = int(sweep[f"_hits_{code}"])
    assert hits > 0, f"{code} is never taken"


def test_the_ninth_branch_returns_exactly_zero(sweep, inputs):
    """Issue #31. `|A| == eps_ab` satisfies neither strict test, so no closed
    form runs and `ndnew` keeps its initialised `nd`."""
    assert EPS_AB == 1.0e-20
    gap = [i for i in range(sweep["a"].size) if abs(float(sweep["a"][i])) == EPS_AB]
    assert gap, "the grid no longer carries |a| == eps_ab exactly"
    for k, dtz in enumerate(sweep["dtz"].tolist()):
        got = np.asarray(_run(inputs, dtz).deln)
        assert np.all(got[gap] == 0.0)
        np.testing.assert_array_equal(got[gap], sweep["deln"][0, k][gap])
    # And the gap is a gap, not "a is zero": a nearby value must take a branch.
    near = float(np.nextafter(EPS_AB, np.inf))
    assert cap.classify(near, 0.0, 0.0, 1.0e3, 60.0) == "1cb"
    assert cap.classify(EPS_AB, 0.0, 0.0, 1.0e3, 60.0) == "gap_a"


def test_masked_rows_come_back_zero_through_poison(sweep, inputs):
    off = np.asarray(sweep["mask"]) == 0
    assert off.sum() == 5
    poisoned = np.asarray(sweep["a"])[off]
    assert np.isnan(poisoned).any() and np.isinf(poisoned).any()
    for dtz in sweep["dtz"].tolist():
        assert np.all(np.asarray(_run(inputs, dtz).deln)[off] == 0.0)


def test_the_error_branch_is_returned_rather_than_raised(sweep):
    """`logic1ca` calls a fatal `ereport` upstream -- the capture's probe fired
    exactly one. This is array code, so one poisoned row must not take the
    column with it: the port returns a mask and the caller decides."""
    assert int(sweep["probe_ereports"]) >= 1
    probe = {k: jnp.asarray(sweep[f"probe_{k}"]) for k in ("a", "b", "c", "nd")}
    n = probe["a"].shape[0]
    result = sv.solvecoagnucl(
        jnp.ones(n, dtype=bool), probe["a"], probe["b"], probe["c"], probe["nd"], dtz=60.0
    )
    assert bool(np.all(np.asarray(result.failed))), "the probe rows are not classified as failed"
    # The Fortran leaves ndnew at nd there, so deln is zero -- which is exactly
    # why the failure has to be reported separately from the value.
    np.testing.assert_array_equal(np.asarray(result.deln), np.zeros(n))
    # `probe_deln` is stacked over the two setups; both must agree with it.
    assert sweep["probe_deln"].shape == (2, n)
    np.testing.assert_array_equal(sweep["probe_deln"][0], sweep["probe_deln"][1])
    np.testing.assert_array_equal(np.asarray(result.deln), sweep["probe_deln"][0])


def test_no_ordinary_row_is_flagged_failed(sweep, inputs):
    for dtz in sweep["dtz"].tolist():
        assert not np.any(np.asarray(_run(inputs, dtz).failed))


def test_the_clamp_is_surfaced(sweep, inputs):
    """`:225-228` caps `sqd*dtz` at 50 with no diagnostic. CLAUDE.md requires a
    cap to be reported, and the Fortran has nowhere to report it -- so the port
    returns the mask."""
    assert SQD_CLAMP == 50.0
    total = 0
    for dtz in sweep["dtz"].tolist():
        clamped = np.asarray(_run(inputs, dtz).clamped)
        total += int(clamped.sum())
        for i in np.flatnonzero(clamped):
            assert _code(sweep, i, dtz).startswith("1a")
            assert np.sqrt(-(4.0 * sweep["a"][i] * sweep["c"][i] - sweep["b"][i] ** 2)) * dtz > 50.0
    assert total > 0, "no row reaches the clamp; the grid no longer straddles it"


def test_up1_changes_the_answer_at_both_settings(sweep, inputs):
    """The spurious factor 3 at `:259`. Gate 0 established this is the branch
    the top soluble mode takes every substep of every shipped namelist, so a
    flag that did nothing would be a serious mis-record."""
    for k, dtz in enumerate(sweep["dtz"].tolist()):
        on = np.asarray(_run(inputs, dtz, fidelity=FidelityConfig(coag_intra_factor3=True)).deln)
        off = np.asarray(_run(inputs, dtz, fidelity=FidelityConfig(coag_intra_factor3=False)).deln)
        rows = [
            i for i in range(sweep["a"].size) if sweep["mask"][i] and _code(sweep, i, dtz) == "1cb"
        ]
        assert rows
        assert not np.array_equal(on[rows], off[rows]), f"the flag changed nothing at dtz={dtz}"
        # Only the 1cb rows may move.
        others = [i for i in range(sweep["a"].size) if i not in rows]
        np.testing.assert_array_equal(on[others], off[others])
        # The default arm is the one the reference agrees with.
        np.testing.assert_array_equal(on[rows], sweep["deln"][0, k][rows])


def test_the_2b_arm_keeps_the_cancellation(sweep, inputs):
    """`deln = (nd + c*dtz) - nd`, which is NOT `c*dtz`: at `nd = 1e3` and
    `c*dtz = 1e-4` it comes back `9.999999997489795e-05`. A port that returned
    the increment directly would disagree on every row where `nd` dominates --
    the capture's own first draft asserted `c*dtz` and was refused."""
    for k, dtz in enumerate(sweep["dtz"].tolist()):
        got = np.asarray(_run(inputs, dtz).deln)
        for i in range(sweep["a"].size):
            if not sweep["mask"][i] or _code(sweep, i, dtz) != "2b":
                continue
            nd_i, c_i = float(sweep["nd"][i]), float(sweep["c"][i])
            assert got[i] == (nd_i + c_i * dtz) - nd_i
            np.testing.assert_array_equal(got[i], sweep["deln"][0, k][i])


def test_the_source_still_says_what_this_port_reproduces():
    assert cap.verify_source_literals() == {"checked": 8}
