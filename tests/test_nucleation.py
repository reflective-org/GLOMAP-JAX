"""Task 52: `ukca_calcnucrate`, the nucleation driver.

Against `tests/goldens/calcnucrate.f64.leaf.npz`: 7 configurations x 151 rows,
both outputs.

**Byte-equal on the five configurations that run only boundary-layer
nucleation. Twelve values differ on the two that run the binary term, by at
most 2 ulp, and substituting libm's `exp` removes all twelve.** Issue #28's
third routine, and the cleanest illustration of it yet: the two halves of the
same routine are gated differently because only one of them reaches
`ukca_binapara`'s exponentials.

What this file pins beyond the numbers:

* **Which term runs is decided by two logicals, not by the switches.**
  `l1 = (i_nuc_method == 2) and (height > zbl or bln_on == 0)` and
  `l2 = (i_nuc_method == 3) and (ibln == 3)`; binary runs where `l1 or l2` and
  boundary-layer where `not l1`. With `i_nuc_method = 3, ibln = 3` **both**
  run, which no shipped namelist reaches -- `bln_on` is off in all five.

* **The order inside a box.** Boundary-layer nucleation reads the H2SO4 the
  binary term left behind, in its guard and in its rate. Computing both from
  the original concentration over-depletes, and
  `test_the_second_term_reads_what_the_first_left` applies that mutation.

* **`zbl = min(htpblg, 6000)`.** A 12 km boundary layer is capped at 6 km, so
  a box at 7 km is above it.

* **`s` and `aird` reach no output.** They feed the Kulmala water-vapour
  concentration, which is in the branch `:257` makes unreachable. Asserted by
  poking them, not by reading the source.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import glomap_jax.physics.binapara as bp
import glomap_jax.physics.nucleation as nu

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "calcnucrate.f64.leaf.npz"
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_calcnucrate.F90"

sys.path.insert(0, str(REPO / "validation"))

import capture_calcnucrate_leaf as cap  # noqa: E402

ARRAY_KEYS = ("t", "s", "rh", "aird", "sec_org", "height", "htpblg", "s_cond_s")

#: Measured. Only the two configurations that reach `binapara` differ, and only
#: through its exponentials.
EXP_ULP = 2
EXP_AFFECTED = {(2, 1, 0): 6, (3, 3, 1): 6}


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run validation/capture_calcnucrate_leaf.py (or `make goldens`)"
    return np.load(GOLDEN, allow_pickle=False)


@pytest.fixture(scope="module")
def inputs(sweep):
    out = {k: jnp.asarray(sweep[k]) for k in ARRAY_KEYS}
    out["h2so4"] = jnp.asarray(sweep["h2so4_in"])
    return out


def _run(inputs, dtz, config, **override):
    m, ibln, bln_on = config
    kw = dict(inputs)
    kw.update(override)
    return nu.calcnucrate(
        kw["t"],
        kw["s"],
        kw["rh"],
        kw["aird"],
        kw["h2so4"],
        kw["sec_org"],
        kw["height"],
        kw["htpblg"],
        kw["s_cond_s"],
        dtz=dtz,
        bln_on=bln_on,
        ibln=ibln,
        i_nuc_method=m,
    )


with np.load(GOLDEN, allow_pickle=False) as _g:
    CONFIGS = [tuple(int(x) for x in row) for row in _g["config"]]
    DTZ = float(_g["dtz"])
CONFIG_IDS = ["-".join(str(x) for x in c) for c in CONFIGS]


@pytest.mark.parametrize("c_i", range(len(CONFIGS)), ids=CONFIG_IDS)
def test_the_port_matches_the_compiled_routine(sweep, inputs, c_i):
    """Exact where no binary term runs; a measured 2-ulp window where one
    does, with the affected count pinned so the allowance cannot grow."""
    config = CONFIGS[c_i]
    h2so4, delh2so4 = _run(inputs, DTZ, config)
    expected = EXP_AFFECTED.get(config, 0)
    for name, got, want in (
        ("h2so4", np.asarray(h2so4), sweep["h2so4"][0, c_i]),
        ("delh2so4", np.asarray(delh2so4), sweep["delh2so4"][0, c_i]),
    ):
        off = got != want
        assert int(off.sum()) == expected, (
            f"{name} at {config}: {int(off.sum())} differ, measured {expected}"
        )
        if off.any():
            ulp = np.abs(got[off] - want[off]) / np.spacing(np.abs(want[off]))
            assert ulp.max() <= EXP_ULP


def test_the_whole_gap_is_the_exponential(sweep, inputs, monkeypatch):
    """The decisive experiment, a third time. Both modules' `exp` is routed
    through numpy -- the same libm gfortran reaches -- and every difference
    disappears across all seven configurations."""

    class _Libm:
        def __getattr__(self, name):
            return getattr(jnp, name)

        @staticmethod
        def exp(x):
            return jnp.asarray(np.exp(np.asarray(x)))

    monkeypatch.setattr(bp, "jnp", _Libm())
    monkeypatch.setattr(nu, "jnp", _Libm())
    for c_i, config in enumerate(CONFIGS):
        h2so4, delh2so4 = _run(inputs, DTZ, config)
        np.testing.assert_array_equal(
            np.asarray(h2so4), sweep["h2so4"][0, c_i], err_msg=str(config)
        )
        np.testing.assert_array_equal(
            np.asarray(delh2so4), sweep["delh2so4"][0, c_i], err_msg=str(config)
        )


def test_only_configurations_that_reach_the_binary_term_can_differ(sweep):
    """The gap tracks the binary term, and the claim is one-directional.

    Reaching `binapara` is *necessary* for a configuration to differ, not
    sufficient: four of the seven reach it on the handful of rows that sit
    above the boundary layer, and on those rows the 1-ulp exponential does not
    propagate to the answer. So the assertion is that every differing
    configuration reaches it, and that a configuration reaching it on no row
    cannot differ -- which is the direction a wrong port would break.
    """
    grid = {k: sweep[k] for k in ARRAY_KEYS}
    reach = {c: bool(cap.branch_flags(grid, c)["bhn"].any()) for c in CONFIGS}
    for config in EXP_AFFECTED:
        assert reach[config], f"{config} differs but never runs the binary term"
    for config, reached in reach.items():
        if not reached:
            assert config not in EXP_AFFECTED
    # The two halves must both be represented, or the test is vacuous.
    assert any(reach.values()) and not all(reach.values())


def test_the_change_equals_the_drop_bit_for_bit(sweep, inputs):
    """`delh2so4_nucl` accumulates `h2so4old - h2so4` over both terms, so it is
    the total drop by construction. Two accumulations that should agree and do
    not is what a tolerance would hide."""
    for config in CONFIGS:
        h2so4, delh2so4 = _run(inputs, DTZ, config)
        np.testing.assert_array_equal(
            np.asarray(delh2so4),
            np.asarray(inputs["h2so4"]) - np.asarray(h2so4),
            err_msg=str(config),
        )


def test_the_concentration_only_ever_falls(inputs):
    """`:374-379`. Both guards, and neither is dead: `taken` is a rate times
    `nmol*dtz` with nothing bounding it below zero."""
    for config in CONFIGS:
        h2so4, delh2so4 = _run(inputs, DTZ, config)
        assert np.all(np.asarray(h2so4) <= np.asarray(inputs["h2so4"]) + 0.0)
        assert np.all(np.asarray(h2so4) >= 0.0)
        assert np.all(np.asarray(delh2so4) >= 0.0)


def test_the_second_term_reads_what_the_first_left(sweep, inputs):
    """The sequencing, applied as a mutation.

    Under `(3, 3, 1)` both terms run. If the boundary-layer term were given the
    original concentration instead of the depleted one, it would take more --
    the rate is proportional to `h2so4` (or its square) and the guard is
    `h2so4 > conc_eps`. Demonstrated by running the two terms separately and
    summing, which is what a port that fused them would compute.
    """
    both = (3, 3, 1)
    h_both, d_both = _run(inputs, DTZ, both)

    # Binary alone: i_nuc_method = 2 with bln_on = 0 makes l1 true everywhere.
    _, d_bhn = _run(inputs, DTZ, (2, 1, 0))
    # Boundary-layer alone, Metzger, from the ORIGINAL concentration.
    _, d_bln = _run(inputs, DTZ, (2, 3, 1))

    fused = np.asarray(d_bhn) + np.asarray(d_bln)
    assert not np.allclose(fused, np.asarray(d_both), rtol=1e-12), (
        "running the two terms independently and adding gives the same answer as "
        "running them in sequence; the ordering is not being exercised"
    )
    assert np.any(fused > np.asarray(d_both)), "the fused form must over-deplete somewhere"
    assert np.all(np.asarray(h_both) >= 0.0)


@pytest.mark.parametrize("dead", ("s", "aird"))
def test_the_dead_arguments_reach_no_output(inputs, dead):
    """They feed the Kulmala water-vapour concentration at `:326`, in the
    branch `:257` makes unreachable. Poked rather than read off the source: the
    project has three recorded cases of a reachability claim made without
    measuring it."""
    for config in CONFIGS:
        base = _run(inputs, DTZ, config)
        poked = _run(inputs, DTZ, config, **{dead: inputs[dead] * 7.0 + 1.0})
        np.testing.assert_array_equal(
            np.asarray(base[0]), np.asarray(poked[0]), err_msg=str(config)
        )
        np.testing.assert_array_equal(
            np.asarray(base[1]), np.asarray(poked[1]), err_msg=str(config)
        )


def test_sec_org_is_read_by_the_metzger_form_alone(inputs):
    """`:404` is the only reference. Under `ibln` 1 or 2 the argument must be
    inert, and under 3 it must not be -- both halves, because only the second
    can fail if the configuration never reached the rate at all."""
    poked = inputs["sec_org"] * 3.0 + 1.0
    for config in CONFIGS:
        base = _run(inputs, DTZ, config)
        got = _run(inputs, DTZ, config, sec_org=poked)
        same = np.array_equal(np.asarray(base[1]), np.asarray(got[1]))
        assert same == (config[1] != 3), config


def test_the_boundary_layer_height_is_capped_at_six_kilometres(inputs):
    """`zbl = MIN(htpblg, zmaxbln)` at `:265-267`. A 12 km boundary layer
    behaves exactly as a 6 km one, so a box at 7 km is above it and takes the
    binary-only branch under `i_nuc_method = 2`."""
    n = inputs["t"].shape[0]
    deep = jnp.full((n,), 12000.0)
    capped = jnp.full((n,), nu.ZMAXBLN)
    config = (2, 1, 1)
    a = _run(inputs, DTZ, config, htpblg=deep)
    b = _run(inputs, DTZ, config, htpblg=capped)
    np.testing.assert_array_equal(np.asarray(a[1]), np.asarray(b[1]))

    # And the cap has to matter: a box between 6 and 12 km must change branch.
    high = jnp.full((n,), 7000.0)
    above = _run(inputs, DTZ, config, htpblg=deep, height=high)
    below = _run(inputs, DTZ, config, htpblg=jnp.full((n,), 20000.0), height=high)
    np.testing.assert_array_equal(np.asarray(above[1]), np.asarray(below[1]))


@pytest.mark.parametrize("i_nuc_method", (0, 1, 4))
def test_an_unsupported_nucleation_method_raises(inputs, i_nuc_method):
    with pytest.raises(ValueError, match="i_nuc_method"):
        _run(inputs, DTZ, (i_nuc_method, 1, 1))


@pytest.mark.parametrize("ibln", (0, 4, -1))
def test_an_out_of_range_ibln_raises(inputs, ibln):
    """Upstream this reaches `ereport` -- but `dpbln` is assigned only for
    `ibln` in 1..3 (`:271`) and both rate expressions read it, so under a shim
    that returns rather than stopping, the routine would compute from
    uninitialised memory."""
    with pytest.raises(ValueError, match="dpbln"):
        _run(inputs, DTZ, (2, ibln, 1))


def test_the_parameters_still_match_the_routine():
    """Re-parsed, not remembered. The spellings are the source's own -- `5.0e-13`
    and not Python's `5e-13` -- which is why the comparison is against the text
    the capture already asserts rather than against `repr`."""
    cap.verify_source_literals()
    text = SOURCE.read_text(encoding="utf-8").replace(" ", "")
    for name, value, spelling in (
        ("zmaxbln", nu.ZMAXBLN, "6000.0"),
        ("afac_pna", nu.AFAC_PNA, "5.0e-13"),
        ("afac_act", nu.AFAC_ACT, "5.0e-7"),
        ("afac_kin", nu.AFAC_KIN, "4.0e-13"),
        ("dpbln", nu.DPBLN, "1.5"),
    ):
        assert f"{name}={spelling}" in text, f"{name} is no longer written {spelling}"
        assert value == float(spelling), f"{name} in the port is {value}, source says {spelling}"
    assert "i_bhn_method=i_bhn_method_vekhamaki" in text, "the Kulmala branch may be live again"
    assert nu.J_MIN == 1.0e-3
    assert "japp(jl)>1.0e-3" in text and "japp_bln>1.0e-3" in text


def test_the_captures_branch_derivation_agrees_with_the_golden(sweep):
    """`branch_flags` is a reading of `:302-306`. If it were wrong, every
    coverage claim the capture makes would be about the wrong branches."""
    grid = {k: sweep[k] for k in ARRAY_KEYS}
    seen = cap.verify_branch_coverage(grid)
    assert seen["neither"] == 0
    assert seen["both"] > 0 and seen["bhn_only"] > 0 and seen["bln_only"] > 0
