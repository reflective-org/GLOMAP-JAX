"""Task 50: `ukca_calc_coag_kernel`, and the first routine where `EXP` is live.

Against `tests/goldens/coag_kernel.f64.leaf.npz`: 7 setups x 4 calls, `kii_arr`
and `kij_arr`, 3,690 non-zero elements in total.

**Byte-equal at `icoag = 3` and at `coag_on = 0`. Not byte-equal at `icoag = 1`
or `2`, and the reason is not a porting error.**

20 elements at method 1 and 22 at method 2 differ from the compiled routine, by
at most 1 and 3 ulp. Substituting `numpy.exp` for `jnp.exp` inside `coag_coff`
makes **all 42 disappear** -- `test_the_whole_gap_is_the_exponential` performs
exactly that substitution and requires zero differences. So the entire
discrepancy is one call: `EXP(-1.1/kn)` in the Cunningham slip correction at
`ukca_coag_coff_v.F90:266`.

`jnp.exp` is XLA's own evaluation; `numpy.exp` and gfortran's `EXP` both go to
the platform libm, and on the arguments this grid produces they agree bit for
bit while XLA does not -- on 5 of the 25 distinct arguments method 1 evaluates
and 22 of the 95 method 2 does.

**This is the first ported routine with `EXP` in a live path.** `vapour` has
only `LOG` and `SQRT` there, `water_content` no transcendental at all, and
`drydiam` and `volume_mode` reach `EXP` nowhere that a golden compares. So the
project has been able to gate on byte equality for four phases without ever
testing what CLAUDE.md's own numerics table already records: `exp` differs from
gfortran on 14% of its sweep. The table calls that "inside tolerance", which is
true of `RTOL_TRANSCENDENTAL` and false of `array_equal`.

**Task 49's byte-equality claim for `coag_coff` was grid-luck, not a property.**
Its 280-row fixture evaluates `EXP` at arguments where the 1-ulp gap is
absorbed by the `1.257 + 0.4*EXP(...)` sum -- 0.4 times an ulp of 0.37 is a
tenth of an ulp of 1.4, so it usually vanishes in the addition and occasionally
does not. `test_coag_coff.py` is byte-equal on every one of its rows and would
have stayed so however carefully it was reviewed; only a second grid found it.
That is the same shape as the two phase-D failures, one level up: not a grid
too coarse for the defect, but a grid on which the defect cancels.

Issue #28. What is gated here is a **measured** bound -- exactly 20 and 22
elements, at most 4 ulp -- so the number cannot grow silently, plus the
substitution test that says what causes it.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from glomap_jax.physics import coag_coff as cg
from glomap_jax.physics.coag_kernel import calc_coag_kernel, slot_map

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "coag_kernel.f64.leaf.npz"

sys.path.insert(0, str(REPO / "validation"))

from conftest import assert_matches_reference  # noqa: E402

GRID_KEYS = ("drydp", "dvol", "wetdp", "wvol", "rhopar", "mfpa", "dvisc", "t")

#: Measured, not chosen. 1 ulp at method 1, 3 at method 2; the bound is the
#: next power of two up so a re-measure on another libm has room, and the
#: element COUNTS below are what actually stop it drifting.
EXP_ULP = 4

#: Elements that differ, by (icoag, coag_on), over the whole archive.
EXP_AFFECTED = {(1, 1): 20, (2, 1): 22, (3, 1): 0, (1, 0): 0}


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run validation/capture_coag_kernel_leaf.py (or `make goldens`)"
    return np.load(GOLDEN, allow_pickle=False)


@pytest.fixture(scope="module")
def grid(sweep):
    return {k: jnp.asarray(sweep[k]) for k in GRID_KEYS}


def _run(sweep, grid, s_i, c_i):
    return calc_coag_kernel(
        sweep["mode"][s_i],
        sweep["modesol"][s_i],
        *(grid[k] for k in GRID_KEYS),
        coag_on=int(sweep["call_coag_on"][c_i]),
        icoag=int(sweep["call_icoag"][c_i]),
    )


with np.load(GOLDEN, allow_pickle=False) as _g:
    SETUPS = _g["setups"].tolist()
    CALLS = list(zip(_g["call_icoag"].tolist(), _g["call_coag_on"].tolist()))
CASES = [(s, c) for s in range(len(SETUPS)) for c in range(len(CALLS))]
CASE_IDS = [f"setup{SETUPS[s]}-icoag{CALLS[c][0]}-on{CALLS[c][1]}" for s, c in CASES]


def _assert_within_ulp(actual, expected, what, ulp):
    """A same-platform ulp window, which `assert_matches_reference` is not.

    That helper takes the exact branch whenever it is running where the goldens
    were captured, because its allowance exists for a *different* libm. The gap
    here is on the capture platform itself and has a named cause, so it needs
    its own assertion rather than a widened reading of that one -- and it must
    not silently become a cross-platform allowance on top.
    """
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    lo = hi = expected
    for _ in range(ulp):
        lo, hi = np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf)
    off = ~((actual >= lo) & (actual <= hi))
    if off.any():
        gap = np.abs(actual[off] - expected[off]) / np.spacing(np.abs(expected[off]))
        raise AssertionError(
            f"{what}: {off.sum()} elements past {ulp} ulp, worst {gap.max():.1f} -- "
            "the exponential accounts for at most 3; this is something else"
        )


@pytest.mark.parametrize(("s_i", "c_i"), CASES, ids=CASE_IDS)
def test_the_port_matches_the_compiled_routine(sweep, grid, s_i, c_i):
    """Exact where no `EXP` reaches the answer, within a measured ulp window
    where it does. Exact means `ulp=0` through the shared helper, so the
    allowance cannot quietly spread to the methods that do not need it."""
    icoag, coag_on = CALLS[c_i]
    exact = coag_on == 0 or icoag == 3
    kii, kij = _run(sweep, grid, s_i, c_i)
    tag = CASE_IDS[CASES.index((s_i, c_i))]
    if exact:
        assert_matches_reference(np.asarray(kii), sweep["kii"][s_i, c_i], f"kii[{tag}]", ulp=0)
        assert_matches_reference(np.asarray(kij), sweep["kij"][s_i, c_i], f"kij[{tag}]", ulp=0)
    else:
        _assert_within_ulp(kii, sweep["kii"][s_i, c_i], f"kii[{tag}]", EXP_ULP)
        _assert_within_ulp(kij, sweep["kij"][s_i, c_i], f"kij[{tag}]", EXP_ULP)


def test_exactly_which_elements_the_exponential_moves(sweep, grid):
    """A measured bound with no measured count is a knob. These are the counts;
    if one grows, something other than `exp` has changed."""
    counts = dict.fromkeys(EXP_AFFECTED, 0)
    for s_i in range(len(SETUPS)):
        for c_i, key in enumerate(CALLS):
            kii, kij = _run(sweep, grid, s_i, c_i)
            counts[key] += int((np.asarray(kii) != sweep["kii"][s_i, c_i]).sum())
            counts[key] += int((np.asarray(kij) != sweep["kij"][s_i, c_i]).sum())
    assert counts == EXP_AFFECTED


def test_the_whole_gap_is_the_exponential(sweep, grid, monkeypatch):
    """The decisive experiment. Route `coag_coff`'s only transcendental through
    numpy -- which reaches the same platform libm gfortran does -- and every
    difference disappears. Nothing else in the port is contributing."""

    class _LibmExp:
        def __getattr__(self, name):
            return getattr(jnp, name)

        @staticmethod
        def exp(x):
            return jnp.asarray(np.exp(np.asarray(x)))

    monkeypatch.setattr(cg, "jnp", _LibmExp())
    differing = 0
    for s_i in range(len(SETUPS)):
        for c_i in range(len(CALLS)):
            kii, kij = _run(sweep, grid, s_i, c_i)
            differing += int((np.asarray(kii) != sweep["kii"][s_i, c_i]).sum())
            differing += int((np.asarray(kij) != sweep["kij"][s_i, c_i]).sum())
    assert differing == 0, (
        f"{differing} elements still differ with libm's exp, so something other "
        "than the exponential is wrong and the docstring above is misleading"
    )


def test_method_three_has_no_exponential_and_is_exact(sweep, grid):
    """The control. `icoag = 3` replaces the Cunningham correction with the
    constant 1.591 (`:317`), so it reaches no `EXP` -- and it is byte-equal on
    every setup. If it ever stops being, the cause is not `exp`."""
    for s_i in range(len(SETUPS)):
        c_i = CALLS.index((3, 1))
        kii, kij = _run(sweep, grid, s_i, c_i)
        np.testing.assert_array_equal(np.asarray(kii), sweep["kii"][s_i, c_i])
        np.testing.assert_array_equal(np.asarray(kij), sweep["kij"][s_i, c_i])


@pytest.mark.parametrize("s_i", range(7))
def test_the_port_fills_the_slots_the_fortran_filled(sweep, grid, s_i):
    """The transposition gate, on the port rather than on the archive. Values
    cannot check this: the kernel is symmetric, so a transposed convention
    would put the right number in the wrong slot and every comparison above
    would still pass on a symmetric golden."""
    c_i = CALLS.index((1, 1))
    kii, kij = _run(sweep, grid, s_i, c_i)
    got = {(i, j) for i in range(8) for j in range(8) if np.any(np.asarray(kij)[:, i, j] != 0.0)}
    want = {
        (i, j) for i in range(8) for j in range(8) if np.any(sweep["kij"][s_i, c_i][:, i, j] != 0.0)
    }
    assert got == want
    got_kii = {m for m in range(8) if np.any(np.asarray(kii)[:, m] != 0.0)}
    want_kii = {m for m in range(8) if np.any(sweep["kii"][s_i, c_i][:, m] != 0.0)}
    assert got_kii == want_kii


def test_a_transposed_convention_would_not_be_caught_by_values(sweep, grid):
    """Stated as a test because it is the reason the slot gate exists.

    Transposing the port's `kij_arr` gives an array whose *non-zero values* are
    exactly the golden's non-zero values -- the multiset is identical -- and
    only the positions differ. Any check that compared sorted values, or
    summed, or compared `kij + kij.T`, would pass a transposed port.
    """
    c_i = CALLS.index((1, 1))
    _, kij = _run(sweep, grid, 6, c_i)
    kij = np.asarray(kij)
    transposed = np.transpose(kij, (0, 2, 1))
    assert not np.array_equal(kij, transposed)
    np.testing.assert_array_equal(np.sort(kij, axis=None), np.sort(transposed, axis=None))
    np.testing.assert_array_equal(kij + transposed, transposed + kij)


def test_the_slot_map_is_derived_from_mode_and_not_from_the_golden(sweep):
    """`slot_map` must depend on `mode`. If it returned a fixed set, the
    setup-dependence of the whole driver would be fictional and setup 6 -- two
    active modes -- would claim the slots of setup 8."""
    sets = []
    for s_i in range(len(SETUPS)):
        _, kij = slot_map(sweep["mode"][s_i], sweep["modesol"][s_i])
        sets.append(frozenset(kij))
    assert len(set(sets)) > 1, "slot_map returns the same slots for every setup"
    fewest = sets[SETUPS.index(6)]
    most = sets[SETUPS.index(8)]
    assert fewest < most, "setup 6 should be a strict subset of setup 8"


def test_coag_on_zero_returns_two_zero_arrays(sweep, grid):
    for s_i in range(len(SETUPS)):
        kii, kij = _run(sweep, grid, s_i, CALLS.index((1, 0)))
        assert np.all(np.asarray(kii) == 0.0) and np.all(np.asarray(kij) == 0.0)


def test_the_shapes_are_full_width_whatever_is_active(sweep, grid):
    """`nmodes` is a PARAMETER, so inactive slots exist and stay zero. A port
    that stored only the active modes would change every index downstream."""
    kii, kij = _run(sweep, grid, SETUPS.index(6), CALLS.index((1, 1)))
    nbox = sweep["wetdp"].shape[0]
    assert np.asarray(kii).shape == (nbox, 8)
    assert np.asarray(kij).shape == (nbox, 8, 8)
