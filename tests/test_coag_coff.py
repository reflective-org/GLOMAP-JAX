"""Task 49: `ukca_coag_coff_v`, byte-equal to the compiled routine.

Against `tests/goldens/coag_coff.f64.leaf.npz`: 6 calls -- three methods times
both `coag_on` settings -- times 280 rows. `array_equal` where the goldens were
captured, a measured ulp bound elsewhere.

**The fixture earned its keep on the first comparison.** The port's `icoag = 2`
branch wrote `(PI/0.75) * rmid * rmid * rmid`, which associates as
`(((c*r)*r)*r)`. The Fortran writes `(pi/0.75)*rmid(:)**3`, where the cube is a
separate `powi` operand and the prefactor multiplies the *finished* cube. Those
differ on 14 of the 280 rows, by one ulp. Nothing in the physics, the shape of
the code or a plausible tolerance would have shown it; a byte-equality gate
over a grid wide enough to contain the disagreement did.

`test_the_prefactor_multiplies_the_finished_cube` reapplies that exact mutation
and requires it to fail, so the fix cannot be undone silently.

Two further mutations are pinned, and one precaution is honestly labelled as
currently free:

* **`termv3` divides by `3.0`.** Removing `numerics.true_divide` changes 3 rows
  at `icoag = 3` and none at 1 or 2, which read no such expression.

* **The mask is a select.** The masked-off rows carry `inf`, `-inf` and `NaN`.

* **The cubes are written as multiplications, which JAX does not need.** `jnp`
  lowers `x**3` to `lax.integer_pow`, i.e. repeated multiplication, and the two
  spellings agree on all 273 live rows and on 400,000 random values. **numpy**
  does not: `x**3` calls `pow` and differs from `x*x*x` on 25.7% of a random
  sample, which is phase C's `d**3` finding. The explicit form is kept because
  this file has already been bitten once by an expression whose result depended
  on whether the array was numpy or jnp.

**The byte equality below is narrower than it looks, and task 50 found out how.**
`ukca_coag_coff_v.F90:266` calls `EXP`, and `jnp.exp` is XLA's own evaluation
while gfortran's `EXP` goes to the platform libm. On this grid they disagree on
5 of the 25 distinct arguments method 1 evaluates and 22 of the 95 method 2
does -- and every row here is still byte-equal, because the 1-ulp gap is
absorbed by the `1.257 + 0.4*EXP(...)` sum. On `ukca_calc_coag_kernel`'s grid it
is not absorbed on 42 elements. So what this file demonstrates is that the port
agrees with the Fortran *on these inputs*, not that it agrees. See
`tests/test_coag_kernel.py` and issue #28.

And one property that is a warning rather than a reassurance: the port is
byte-symmetric under a simultaneous swap of the three `(i, j)` pairs. So is the
Fortran. Neither can distinguish `kij` from `kji`, and neither can `coag_mode`,
which phase C measured symmetric on all 64 entries -- `ukca_calc_coag_kernel`'s
subscripts are the only thing that fixes the convention, and task 50 has
nothing downstream to catch a transposition.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glomap_jax.core import numerics
from glomap_jax.physics import coag_coff as cg

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "coag_coff.f64.leaf.npz"
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_coag_coff_v.F90"

sys.path.insert(0, str(REPO / "validation"))

from conftest import assert_matches_reference  # noqa: E402

ARRAY_ARGS = ("ri", "rj", "vi", "vj", "rhoi", "rhoj", "mfpa", "dvisc", "t")


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run validation/capture_coag_coff_leaf.py (or `make goldens`)"
    return np.load(GOLDEN, allow_pickle=False)


@pytest.fixture(scope="module")
def inputs(sweep):
    out = {"mask": jnp.asarray(sweep["mask"].astype(bool))}
    out.update({k: jnp.asarray(sweep[k]) for k in ARRAY_ARGS})
    return out


def _run(inputs, icoag, coag_on):
    return cg.coag_coff(
        inputs["mask"], *(inputs[k] for k in ARRAY_ARGS), coag_on=coag_on, icoag=icoag
    )


with np.load(GOLDEN, allow_pickle=False) as _g:
    CALLS = [
        (int(_g["call_icoag"][i]), int(_g["call_coag_on"][i])) for i in range(_g["call_icoag"].size)
    ]
CALL_IDS = [f"icoag{a}-on{b}" for a, b in CALLS]


@pytest.mark.parametrize(("icoag", "coag_on"), CALLS, ids=CALL_IDS)
def test_the_port_is_byte_equal_to_the_compiled_routine(sweep, inputs, icoag, coag_on):
    i = CALLS.index((icoag, coag_on))
    got = np.asarray(_run(inputs, icoag, coag_on))
    assert_matches_reference(got, sweep["kij"][0, i], f"kij[{CALL_IDS[i]}]")


def test_every_live_call_actually_produced_something(sweep, inputs):
    """The parametrised test passes trivially against two all-zero arrays. This
    is the guard that says the `coag_on = 1` calls did work."""
    live = np.asarray(sweep["mask"]) == 1
    for icoag in (1, 2, 3):
        k = np.asarray(_run(inputs, icoag, 1))
        assert np.all(k[live] > 0.0) and np.all(np.isfinite(k[live])), icoag


def test_masked_rows_are_written_zero(sweep, inputs):
    off = np.asarray(sweep["mask"]) == 0
    for icoag in (1, 2, 3):
        assert np.all(np.asarray(_run(inputs, icoag, 1))[off] == 0.0), icoag


def test_coag_on_zero_zeroes_rows_the_mask_selected(sweep, inputs):
    """`:239-242` returns before `WHERE (mask(:))`, so this is a different
    statement from the test above and the only path on which a selected row is
    zero."""
    for icoag in (1, 2, 3):
        assert np.all(np.asarray(_run(inputs, icoag, 0)) == 0.0), icoag


def test_the_prefactor_multiplies_the_finished_cube(sweep, inputs, monkeypatch):
    """The mutation the fixture caught, reapplied.

    `(pi/0.75)*rmid(:)**3` at `:294` is a prefactor times a `powi` operand.
    Writing it `(PI/0.75) * rmid * rmid * rmid` associates as `(((c*r)*r)*r)`
    and differs by one ulp -- on 14 of the 280 rows, which a narrower grid
    would have missed entirely.
    """
    rmid = 0.5 * (inputs["ri"] + inputs["rj"])
    live = np.asarray(sweep["mask"]) == 1
    right = np.asarray(cg._vmid(rmid))
    wrong = np.asarray((cg.PI / 0.75) * rmid * rmid * rmid)
    in_vmid = int((right[live] != wrong[live]).sum())

    i2 = CALLS.index((2, 1))
    reference = sweep["kij"][0, i2]
    np.testing.assert_array_equal(np.asarray(_run(inputs, 2, 1)), reference)
    monkeypatch.setattr(cg, "_vmid", lambda r: (cg.PI / 0.75) * r * r * r)
    in_kij = int((np.asarray(_run(inputs, 2, 1))[live] != reference[live]).sum())

    assert (in_vmid, in_kij) == (84, 14), (
        f"the two associations differ on {in_vmid} vmid rows and {in_kij} kij rows, "
        "measured 84 and 14; the grid has moved and this no longer measures what "
        "it says. The gap between them is the point: most of the ulp washes out "
        "in the rounding that follows, and a narrower grid sees neither."
    )


def test_the_termv3_divide_must_not_become_a_reciprocal_multiply(sweep, inputs, monkeypatch):
    """`2.0e6*boltzmann*t/3.0/dvisc` at `:316` divides an array by a scalar
    constant. Only `icoag = 3` contains such an expression, so the mutation
    must break that method and leave the other two alone."""
    monkeypatch.setattr(cg.numerics, "true_divide", lambda x, c: jnp.asarray(x) / c)
    i3 = CALLS.index((3, 1))
    assert not np.array_equal(np.asarray(_run(inputs, 3, 1)), sweep["kij"][0, i3])
    for icoag in (1, 2):
        i = CALLS.index((icoag, 1))
        np.testing.assert_array_equal(np.asarray(_run(inputs, icoag, 1)), sweep["kij"][0, i])


def test_the_cube_spelling_is_free_in_jax_and_not_in_numpy():
    """Honest about its own value. `jnp` lowers `x**3` to `lax.integer_pow`, so
    the explicit multiplication changes nothing there; numpy's `x**3` calls
    `pow` and differs on a quarter of a random sample, which is phase C's
    `d**3` finding. The form is kept because `coag_coff` converts its inputs
    with `jnp.asarray` precisely so that this cannot depend on the caller."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0.5, 4.0, 20000)
    assert (x**3 != x * x * x).sum() > 1000, "numpy's pow no longer differs; re-derive"
    j = jnp.asarray(x)
    np.testing.assert_array_equal(np.asarray(j**3), np.asarray(j * j * j))


def test_the_byte_equality_here_is_cancellation_not_agreement(sweep):
    """What task 50 found, asserted where the claim is made.

    `jnp.exp` and the platform libm disagree on a fifth of this grid's distinct
    Cunningham arguments. Every row above is nonetheless byte-equal, because
    `0.4 * ulp(0.37)` is a tenth of `ulp(1.4)` and usually vanishes in the
    addition at `:266`. `numpy.exp` stands in for gfortran's here -- task 50
    checked the two against `leaf_exp` on the same arguments and found them
    identical, which is the whole point: it is XLA that is the odd one out.

    If this ever reports zero disagreements, the grid has moved somewhere the
    hazard is invisible and the byte equality above stops meaning anything.
    """
    live = np.asarray(sweep["mask"]) == 1
    mfpa, ri, rj = (np.asarray(sweep[k])[live] for k in ("mfpa", "ri", "rj"))

    def disagreements(kn):
        x = jnp.asarray(np.unique(-1.1 / kn))
        return int((np.asarray(jnp.exp(x)) != np.exp(np.asarray(x))).sum()), x.size

    m1 = disagreements(np.concatenate([mfpa / ri, mfpa / rj]))
    m2 = disagreements(mfpa / (0.5 * (ri + rj)))
    assert m1 == (5, 25), f"method 1 exp disagreements {m1}, measured (5, 25)"
    assert m2 == (22, 95), f"method 2 exp disagreements {m2}, measured (22, 95)"


@pytest.mark.parametrize("icoag", (1, 2, 3))
def test_the_kernel_cannot_tell_i_from_j(inputs, icoag):
    """Byte-symmetric under a simultaneous swap of `(ri,rj)`, `(vi,vj)` and
    `(rhoi,rhoj)`, on the whole grid rather than the fixture's six pairs.

    Recorded as a warning: `coag_mode` is symmetric on all 64 entries too
    (phase C), so a transposed `(imode, jmode)` would be undetectable both in
    the table and in the kernel. Only `ukca_calc_coag_kernel`'s subscripts fix
    the convention."""
    straight = np.asarray(_run(inputs, icoag, 1))
    swapped = np.asarray(
        cg.coag_coff(
            inputs["mask"],
            inputs["rj"],
            inputs["ri"],
            inputs["vj"],
            inputs["vi"],
            inputs["rhoj"],
            inputs["rhoi"],
            inputs["mfpa"],
            inputs["dvisc"],
            inputs["t"],
            coag_on=1,
            icoag=icoag,
        )
    )
    np.testing.assert_array_equal(straight, swapped)
    # And the grid really is asymmetric, or this compares two identical calls.
    assert not np.array_equal(np.asarray(inputs["ri"]), np.asarray(inputs["rj"]))


@pytest.mark.parametrize(
    ("arg", "dead_at"),
    [("vi", (2, 3)), ("vj", (2, 3)), ("rhoi", (3,)), ("rhoj", (3,)), ("mfpa", (3,))],
)
def test_an_argument_is_read_at_exactly_the_methods_that_read_it(inputs, arg, dead_at):
    """The port must share the Fortran's argument map. Replacing one argument
    with nonsense must change nothing at the methods that do not read it, and
    something at the methods that do.

    `mfpa` is the notable one: `icoag = 3` uses `UKCA_MFP_REF` instead, so the
    caller's mean free path -- and the pressure-temperature scaling the
    routine's header describes -- never enters."""
    for icoag in (1, 2, 3):
        base = np.asarray(_run(inputs, icoag, 1))
        poked = dict(inputs)
        poked[arg] = inputs[arg] * 3.0
        got = np.asarray(
            cg.coag_coff(poked["mask"], *(poked[k] for k in ARRAY_ARGS), coag_on=1, icoag=icoag)
        )
        same = np.array_equal(base, got)
        assert same == (icoag in dead_at), f"{arg} at icoag={icoag}"


def test_icoag_four_raises_and_names_the_defect(inputs):
    """UP-5. `:339-340` reads `mfppi`/`mfppj`, assigned only inside the
    `icoag == 1` block, so there is no correct reference. `docs/unsupported.md`
    says this raises rather than producing plausible garbage."""
    with pytest.raises(ValueError, match="UP-5"):
        _run(inputs, 4, 1)
    text = SOURCE.read_text(encoding="utf-8").replace(" ", "")
    assert "mfppi(:)*cci(:)/ri(:)/ri(:)" in text, "the icoag=4 read has moved; recheck UP-5"


@pytest.mark.parametrize("icoag", (0, 5, -1))
def test_an_unknown_method_raises_rather_than_returning_zero(inputs, icoag):
    """The four blocks are the only writers of `kij` after `:238` zeroes it, so
    upstream an unknown `icoag` silently returns no coagulation at all."""
    with pytest.raises(ValueError):
        _run(inputs, icoag, 1)


def test_the_mean_free_path_parameter_still_appears_in_the_routine():
    text = SOURCE.read_text(encoding="utf-8").replace(" ", "")
    assert cg.UKCA_MFP_REF == 6.6e-8
    assert "ukca_mfp_ref=6.6e-8" in text
    # The Cunningham constant the code uses, which its own header calls 1.59.
    assert "1.591*ukca_mfp_ref" in text


@pytest.mark.parametrize("icoag", (1, 2, 3))
def test_the_gradient_is_finite_on_every_live_row(sweep, inputs, icoag):
    live = np.asarray(sweep["mask"]) == 1

    def total(ri):
        poked = dict(inputs)
        poked["ri"] = ri
        return cg.coag_coff(
            poked["mask"], *(poked[k] for k in ARRAY_ARGS), coag_on=1, icoag=icoag
        ).sum()

    g = np.asarray(jax.grad(total)(inputs["ri"]))
    assert np.isfinite(g[live]).all()
    assert np.abs(g[live]).max() > 0.0


def test_the_poisoned_masked_rows_behave_as_cond_coff_measured(sweep, inputs):
    """The same limitation, measured for this routine rather than assumed from
    the other one. `safe_divide` guards denominators; a numerator that is
    already `inf` or `NaN` still poisons the cotangent, and how many rows that
    reaches depends on how many products the method forms around a division.
    Method 3 forms none and is clean."""
    off = np.asarray(sweep["mask"]) == 0

    def bad(icoag):
        def total(ri):
            poked = dict(inputs)
            poked["ri"] = ri
            return cg.coag_coff(
                poked["mask"], *(poked[k] for k in ARRAY_ARGS), coag_on=1, icoag=icoag
            ).sum()

        return int((~np.isfinite(np.asarray(jax.grad(total)(inputs["ri"]))[off])).sum())

    assert (bad(1), bad(2), bad(3)) == (6, 4, 0)


def test_safe_divide_is_what_guards_the_cunningham_exponent(inputs):
    """`EXP(-1.1/kn)` at `:266`. In Fortran the unary minus binds outside the
    divide; here it binds to the literal. Negation is exact, so the two agree
    bit for bit -- and only the second spelling has a denominator
    `safe_divide` can substitute."""
    kn = jnp.asarray([1.0e-6, 0.5, 1.0, 2.0, 1.0e6])
    mask = jnp.ones_like(kn, dtype=bool)
    np.testing.assert_array_equal(
        np.asarray(numerics.safe_divide(-1.1, kn, mask)),
        np.asarray(-(numerics.safe_divide(1.1, kn, mask))),
    )
    assert np.all(np.asarray(cg._cunningham(kn, mask)) >= 1.0)
