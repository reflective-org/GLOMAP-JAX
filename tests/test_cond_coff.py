"""Task 47: `ukca_cond_coff_v`, byte-equal to the compiled routine.

Against `tests/goldens/cond_coff.f64.leaf.npz`: 24 calls -- six scalar sets
across all four `(ifuchs, idcmfp)` settings -- times 259 rows, on both outputs.
`array_equal`, not `allclose`, wherever the goldens were captured; a measured
ulp bound elsewhere, per `conftest.assert_matches_reference`.

Byte equality is the acceptance criterion because it is achievable here and
because a tolerance would hide exactly the two defects this phase is most
likely to reintroduce. Both are pinned by mutation:

* **the divide-by-a-constant rewrite.** `:179` divides by the literal
  `101325.0`. Written as a plain `/`, XLA turns it into a multiply by the
  reciprocal and 8 of the sweep's 259 rows change. That is the defect that cost
  phase D 73 tests, and `test_the_pressure_divide_must_not_become_a_reciprocal`
  applies it and watches it fail.

* **the mask as a multiply.** The fixture's masked-off rows carry `inf`, `-inf`
  and `NaN` on purpose, so `mask * term` gives `NaN` where `WHERE` gives the
  `0.0` the routine opens with at `:167`.

One measurement recorded here rather than assumed: `term8` writes
`base * base` because `(...)**2` in the Fortran is an integer literal exponent,
which gfortran expands to a multiplication. In CPython the two happen to agree
for both live `difvol` values -- so the precaution is currently free, and it is
kept because the equality is a property of libm's `pow` on this platform rather
than of the algebra, and because moving the expression into `jnp` would end it.

And one that is a limitation rather than a guarantee: the double-`where` idiom
makes every *division* safe under reverse-mode AD, and every live row has a
finite cotangent. It does not make a poisoned *numerator* safe.
`test_a_poisoned_masked_row_can_still_poison_the_cotangent` pins where that
bites, so the claim in this file is the one that is true.
"""

import re
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glomap_jax.core import constants, numerics
from glomap_jax.physics import cond_coff as cc_mod

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "cond_coff.f64.leaf.npz"
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_cond_coff_v.F90"

sys.path.insert(0, str(REPO / "validation"))

from conftest import assert_matches_reference  # noqa: E402


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run validation/capture_cond_coff_leaf.py (or `make goldens`)"
    return np.load(GOLDEN, allow_pickle=False)


@pytest.fixture(scope="module")
def inputs(sweep):
    return {
        "mask": jnp.asarray(sweep["mask"].astype(bool)),
        "rp": jnp.asarray(sweep["rp"]),
        "tsqrt": jnp.asarray(sweep["tsqrt"]),
        "airdm3": jnp.asarray(sweep["airdm3"]),
        "rhoa": jnp.asarray(sweep["rhoa"]),
        "pmid": jnp.asarray(sweep["pmid"]),
        "t": jnp.asarray(sweep["t"]),
    }


def _run(inputs, sweep, i):
    return cc_mod.cond_coff(
        inputs["mask"],
        inputs["rp"],
        inputs["tsqrt"],
        inputs["airdm3"],
        inputs["rhoa"],
        inputs["pmid"],
        inputs["t"],
        mmcg=float(sweep["call_mmcg"][i]),
        se=float(sweep["call_se"][i]),
        dmol=float(sweep["call_dmol"][i]),
        difvol=float(sweep["call_difvol"][i]),
        ifuchs=int(sweep["call_ifuchs"][i]),
        idcmfp=int(sweep["call_idcmfp"][i]),
    )


def _ids(sweep):
    return [
        f"{sweep['call_name'][i]}-f{sweep['call_ifuchs'][i]}-d{sweep['call_idcmfp'][i]}"
        for i in range(sweep["call_name"].size)
    ]


with np.load(GOLDEN, allow_pickle=False) as _g:
    CALL_INDICES = list(range(_g["call_name"].size))
    CALL_IDS = _ids(_g)


@pytest.mark.parametrize("i", CALL_INDICES, ids=CALL_IDS)
def test_the_port_is_byte_equal_to_the_compiled_routine(sweep, inputs, i):
    cc, sinkarr = _run(inputs, sweep, i)
    assert_matches_reference(np.asarray(cc), sweep["cc"][0, i], f"cc[{CALL_IDS[i]}]")
    assert_matches_reference(np.asarray(sinkarr), sweep["sinkarr"][0, i], f"sinkarr[{CALL_IDS[i]}]")


def test_every_call_actually_produced_something(sweep, inputs):
    """The parametrised test above passes trivially against an all-zero golden
    and an all-zero port. This is the guard that says both sides did work."""
    live = np.asarray(sweep["mask"]) == 1
    for i in CALL_INDICES:
        cc, sinkarr = _run(inputs, sweep, i)
        assert np.all(np.asarray(cc)[live] > 0.0), CALL_IDS[i]
        assert np.all(np.asarray(sinkarr)[live] > 0.0), CALL_IDS[i]


def test_masked_rows_are_written_zero_not_left_alone(sweep, inputs):
    """`:167-168` zeroes both outputs before any branch, so an inactive mode
    carries 0.0 and not a stale value. The masked rows carry inf and NaN, so a
    port that multiplies by the mask gets NaN instead."""
    off = np.asarray(sweep["block"]) == list(np.load(GOLDEN, allow_pickle=False)["blocks"]).index(
        "masked_off"
    )
    for i in CALL_INDICES:
        cc, sinkarr = _run(inputs, sweep, i)
        assert np.all(np.asarray(cc)[off] == 0.0), CALL_IDS[i]
        assert np.all(np.asarray(sinkarr)[off] == 0.0), CALL_IDS[i]


def test_the_pressure_divide_must_not_become_a_reciprocal_multiply(sweep, inputs, monkeypatch):
    """Name the mutation, then apply it.

    Replacing `numerics.true_divide` with a plain `/` is exactly the port that
    phase D shipped and that lost byte equality across jax versions. It must
    fail here, on the `idcmfp = 2` calls, and it must not disturb `idcmfp = 1`
    -- which reads no pressure at all.
    """
    # The division has to happen in jnp, not numpy: numpy divides, and it is
    # XLA that rewrites. Written `np.asarray(pmid) / c` this comparison finds
    # one difference -- the NaN row comparing unequal to itself -- and would
    # have passed a mutation test that could not fail.
    plain = np.asarray(inputs["pmid"] / cc_mod.P_STANDARD)
    guarded = np.asarray(numerics.true_divide(inputs["pmid"], cc_mod.P_STANDARD))
    finite = np.isfinite(plain) & np.isfinite(guarded)
    assert (plain[finite] != guarded[finite]).sum() == 7, (
        "the pmid axis no longer distinguishes the two forms, so this mutation "
        "test has stopped being able to fail"
    )

    monkeypatch.setattr(cc_mod.numerics, "true_divide", lambda x, c: jnp.asarray(x) / c)

    broke, untouched = 0, 0
    for i in CALL_INDICES:
        cc, _ = _run(inputs, sweep, i)
        same = np.array_equal(np.asarray(cc), sweep["cc"][0, i])
        if int(sweep["call_idcmfp"][i]) == 2:
            broke += not same
        else:
            untouched += same
    assert broke == 12, f"only {broke} of the 12 idcmfp=2 calls noticed the rewrite"
    assert untouched == 12, "the rewrite disturbed a call that reads no pressure"


@pytest.mark.parametrize(("ifuchs", "idcmfp"), [(0, 1), (3, 1), (1, 0), (1, 3), (1, -1)])
def test_a_switch_outside_one_or_two_is_refused(inputs, ifuchs, idcmfp):
    """Stricter than the Fortran, deliberately. `glomap_box_config_mod.F90:155`
    reads both from the namelist and `validate_config` constrains neither; out
    of range, `idcmfp` leaves `dcoff_cp` never assigned and both Fuchs branches
    read it, while `ifuchs` silently returns cc = 0 -- no condensation at all,
    with no ereport."""
    with pytest.raises(ValueError):
        cc_mod.cond_coff(
            inputs["mask"],
            inputs["rp"],
            inputs["tsqrt"],
            inputs["airdm3"],
            inputs["rhoa"],
            inputs["pmid"],
            inputs["t"],
            mmcg=0.098,
            se=1.0,
            dmol=4.5e-10,
            difvol=51.96,
            ifuchs=ifuchs,
            idcmfp=idcmfp,
        )


@pytest.mark.parametrize("ifuchs", (1, 2))
def test_t_is_not_read_at_idcmfp_one(inputs, ifuchs):
    """The port must share the Fortran's argument map: at `idcmfp = 1`,
    replacing the whole temperature array with nonsense changes nothing,
    because only `tsqrt` is read. At `idcmfp = 2` it changes everything."""
    kw = dict(mmcg=0.098, se=1.0, dmol=4.5e-10, difvol=51.96, ifuchs=ifuchs)
    nonsense = jnp.full_like(inputs["t"], 999.0)
    args = (
        inputs["mask"],
        inputs["rp"],
        inputs["tsqrt"],
        inputs["airdm3"],
        inputs["rhoa"],
        inputs["pmid"],
    )

    a, _ = cc_mod.cond_coff(*args, inputs["t"], idcmfp=1, **kw)
    b, _ = cc_mod.cond_coff(*args, nonsense, idcmfp=1, **kw)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    c, _ = cc_mod.cond_coff(*args, inputs["t"], idcmfp=2, **kw)
    d, _ = cc_mod.cond_coff(*args, nonsense, idcmfp=2, **kw)
    assert not np.array_equal(np.asarray(c), np.asarray(d))


def test_the_interfacial_correction_is_exactly_one_at_the_live_sticking_efficiency(inputs):
    """`akn = 1.0/(1.0 + 1.33*kn*fkn*(1.0/se - 1.0))` at `:210`, and
    `ukca_conden.F90:235-237` runs `se = 1.0`, so `1.0/se - 1.0` is exactly
    `0.0` and `akn` is exactly `1.0`. So `cc` at `ifuchs = 2` reduces to
    `term6*dcoff_cp*rp*fkn` -- recomputed here from the port's own scalar terms,
    which is what makes this a check on the branch rather than on itself.

    At `se = 0.5` the same recomputation must NOT match, or the test is only
    asserting that two zero arrays agree.
    """
    mask, rp, tsqrt, rhoa = inputs["mask"], inputs["rp"], inputs["tsqrt"], inputs["rhoa"]
    k = cc_mod.scalar_terms(0.098, 4.5e-10, 51.96)
    dcoff = numerics.safe_divide(k.term5 * tsqrt, rhoa, mask)
    mfp = numerics.safe_divide(jnp.full_like(rhoa, k.term2), inputs["airdm3"], mask)
    kn = numerics.safe_divide(mfp, rp, mask)
    fkn = numerics.safe_divide(1.0 + kn, 1.0 + 1.71 * kn + 1.33 * kn * kn, mask)
    without_akn = jnp.where(mask, k.term6 * dcoff * rp * fkn, 0.0)

    common = dict(mmcg=0.098, dmol=4.5e-10, difvol=51.96, ifuchs=2, idcmfp=1)
    args = (mask, rp, tsqrt, inputs["airdm3"], rhoa, inputs["pmid"], inputs["t"])

    at_one, _ = cc_mod.cond_coff(*args, se=1.0, **common)
    np.testing.assert_array_equal(np.asarray(at_one), np.asarray(without_akn))

    at_half, _ = cc_mod.cond_coff(*args, se=0.5, **common)
    assert not np.array_equal(np.asarray(at_half), np.asarray(without_akn))


def test_mm_da_is_derived_where_it_is_used_and_not_cached(inputs):
    """CLAUDE.md forbids a derived quantity in the constants table, and the
    constants docstring names this expression as the example. The value must
    still be right: `avogadro*boltzmann/rgas`, in that order."""
    assert not hasattr(constants, "MM_DA"), "mm_da has been cached; it is derived, not a constant"
    k = cc_mod.scalar_terms(0.098, 4.5e-10, 51.96)
    assert k.mm_da == constants.AVOGADRO * constants.BOLTZMANN / constants.RGAS


def test_the_inline_constants_still_appear_in_the_routine():
    """`dair` and the reference pressure are locals in the Fortran, not module
    constants, so they cannot be extracted by name -- the same treatment
    `test_constants.py` gives `eps_ab` and `sqd_clamp`."""
    text = SOURCE.read_text(encoding="utf-8")
    assert cc_mod.DAIR == 19.7
    assert "dair=19.7" in text.replace(" ", "")
    assert cc_mod.P_STANDARD == 101325.0
    assert "101325.0" in text
    # The integer literal exponent term8 relies on.
    assert re.search(
        r"term8=\(dair\*\*\(1\.0/3\.0\)\+difvol\*\*\(1\.0/3\.0\)\)\*\*2", text.replace(" ", "")
    )


def test_term8_squares_by_multiplication_which_currently_costs_nothing(inputs):
    """Measured, so the precaution is honest about its own value: for both live
    `difvol` values CPython's `base**2` and `base*base` give the same double, so
    writing the multiplication changes no answer today. It is kept because that
    equality is a property of this platform's `pow` rather than of the algebra,
    and because the same shape in `jnp` -- where `x**2` lowers to a power -- is
    what broke the ZSR polynomial in phase D."""
    for difvol in (51.96, 204.14):
        base = cc_mod.DAIR ** (1.0 / 3.0) + difvol ** (1.0 / 3.0)
        assert base * base == base**2
        assert cc_mod.scalar_terms(0.098, 4.5e-10, difvol).term8 == base * base


@pytest.mark.parametrize("ifuchs", (1, 2))
@pytest.mark.parametrize("idcmfp", (1, 2))
def test_the_gradient_is_finite_on_every_live_row(sweep, inputs, ifuchs, idcmfp):
    """What order 2 needs, and what the double-`where` idiom buys.

    A single `where` around each division would return the same values and a
    `NaN` cotangent, because reverse mode differentiates the branch not taken.
    """
    live = np.asarray(sweep["mask"]) == 1

    def total(rp):
        cc, _ = cc_mod.cond_coff(
            inputs["mask"],
            rp,
            inputs["tsqrt"],
            inputs["airdm3"],
            inputs["rhoa"],
            inputs["pmid"],
            inputs["t"],
            mmcg=0.098,
            se=1.0,
            dmol=4.5e-10,
            difvol=51.96,
            ifuchs=ifuchs,
            idcmfp=idcmfp,
        )
        return cc.sum()

    g = np.asarray(jax.grad(total)(inputs["rp"]))
    assert np.isfinite(g[live]).all()
    assert np.abs(g[live]).max() > 0.0


def test_a_poisoned_masked_row_can_still_poison_the_cotangent(sweep, inputs):
    """The limitation, measured rather than left implicit.

    `safe_divide` substitutes a safe *denominator*. It does nothing about a
    numerator that is already `inf` or `NaN`, and at `ifuchs = 1` the numerator
    of `cc` is `term6*dcoff_cp*rp` -- which is `0.0 * inf` on a masked row whose
    `rp` is infinite. Four of the seven masked rows come back with a non-finite
    cotangent there; at `ifuchs = 2` none do, because the product is formed
    without a division around it.

    No model state can produce those inputs -- `rp` is a wet radius -- so this
    is a property of the fixture's deliberate poison and not of the physics. It
    is pinned so that a later phase which *does* need clean cotangents through
    an arbitrary masked box knows the cost is masking the inputs, not the
    divisions.
    """
    off = np.asarray(sweep["mask"]) == 0

    def total(rp, ifuchs):
        cc, _ = cc_mod.cond_coff(
            inputs["mask"],
            rp,
            inputs["tsqrt"],
            inputs["airdm3"],
            inputs["rhoa"],
            inputs["pmid"],
            inputs["t"],
            mmcg=0.098,
            se=1.0,
            dmol=4.5e-10,
            difvol=51.96,
            ifuchs=ifuchs,
            idcmfp=1,
        )
        return cc.sum()

    g1 = np.asarray(jax.grad(lambda rp: total(rp, 1))(inputs["rp"]))
    g2 = np.asarray(jax.grad(lambda rp: total(rp, 2))(inputs["rp"]))
    assert (~np.isfinite(g1[off])).sum() == 4
    assert np.isfinite(g2[off]).all()
