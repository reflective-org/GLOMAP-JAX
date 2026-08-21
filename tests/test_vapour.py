"""Task 38 — `ukca_vapour`, byte-equal against the compiled routine.

Gate A: the port and the Fortran are driven with the same inputs in one
process and compared with `assert_array_equal`. Not `RTOL_TRANSCENDENTAL`,
which the plan proposed and which is four orders too loose for what the live
path actually contains -- `LOG` and `SQRT`, both bit-identical to gfortran on
the capture platform. The `EXP` and the fractional power are in the dead chain.

The grid is built here rather than read from a golden so this runs the moment
the extension is built. When task 35b's fixture lands, the same comparison runs
without a toolchain.
"""

from __future__ import annotations

import functools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "validation"))

import extract_vapour_literals as ex  # noqa: E402

from glomap_jax.core import numerics  # noqa: E402
from glomap_jax.physics import vapour as vp  # noqa: E402

SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_vapour.F90"
F2PY = REPO / "validation" / "f2py"
NAMELIST = REPO / "fortran" / "namelists" / "boundary_layer.nml"

needs_binding = pytest.mark.skipif(
    not sorted(F2PY.glob("glomap_f2py*.so")),
    reason="binding not built; run validation/build_f2py.sh",
)


@functools.lru_cache(maxsize=1)
def temperature_grid() -> np.ndarray:
    """Coarse coverage, plus the points where a branch changes.

    The interesting temperatures are roots of expressions in the routine, so
    they are solved for here rather than transcribed -- a hardcoded root that
    drifts by a ulp stops landing on the edge it was chosen for, which is the
    only reason it is in the grid.
    """
    coarse = np.arange(150.0, 340.0, 0.5)

    # b = ks3 + ks4/T is exactly zero here: the cancellation pole where the
    # whole solution collapses to 0/0 and the MAX clamp is what rescues it.
    scalars = ex.extract()["scalars"]
    edges = [-scalars["ks4"] / scalars["ks3"]]

    # The temperatures whose `wts` lands on a NINT tie -- an odd multiple of
    # 2.5, where `wts/5` is k + 0.5. Fortran rounds those away from zero and
    # `jnp.round` rounds to even, and the result *indexes the density table*,
    # so a tie going the wrong way selects a different row.
    #
    # Solved for, not transcribed. Without them the grid cannot tell
    # `vapour_round` from `jnp.round` at all: swapping the two left every test
    # here green until these points were added.
    edges += [_solve_for_wts(tie) for tie in np.arange(42.5, 100.0, 5.0)]

    neighbours = [np.nextafter(v, d) for v in edges for d in (-np.inf, np.inf)]
    return np.unique(np.concatenate([coarse, np.array(edges + neighbours)]))


def _solve_for_wts(target: float) -> float:
    """The temperature at which `wts` equals `target`, by bisection.

    `wts` is not monotone in T over the whole range -- it peaks near the hot
    discriminant edge and falls back -- so the bracket is taken from the first
    sign change on a fine scan rather than assumed.
    """
    pmid, s = np.full(1, 1.0e5), np.full(1, 6.31470508962842913e-03)

    def wts_at(temp):
        return float(vp.weight_percent(np.array([temp]), pmid, s, fix_neg_pvol_wat=False)[0])

    scan = np.arange(150.0, 340.0, 0.05)
    values = np.array([wts_at(x) for x in scan]) - target
    crossings = np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))
    assert crossings.size, f"no temperature in 150-340 K gives wts = {target}"
    lo, hi = scan[crossings[0]], scan[crossings[0] + 1]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        if (wts_at(lo) - target) * (wts_at(mid) - target) <= 0.0:
            hi = mid
        else:
            lo = mid
    return lo


def _reference(t, pmid, s, rp, flag):
    """Run the compiled routine in a subprocess, one per flag setting."""
    script = f"""
import json, sys
import numpy as np
sys.path.insert(0, {str(F2PY)!r})
import glomap_f2py as g
assert int(g.wrap_init({str(NAMELIST)!r})) == 0
assert int(g.wrap_set_fix_neg_pvol_wat({flag})) == 0
t = np.array({t.tolist()!r}); pmid = np.array({pmid.tolist()!r})
s = np.array({s.tolist()!r}); rp = np.array({rp.tolist()!r})
before = [int(v) for v in g.wrap_ereport_count()]
wts, rho, ierr = g.leaf_vapour(t, pmid, s, rp)
after = [int(v) for v in g.wrap_ereport_count()]
print("@@R@@" + json.dumps({{"ierr": int(ierr), "wts": np.asarray(wts).tolist(),
                             "rho": np.asarray(rho).tolist(),
                             "shim": [before, after]}}))
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    import json

    out = json.loads(proc.stdout[proc.stdout.rindex("@@R@@") + 5 :])
    assert out["ierr"] == 0
    assert out["shim"][0] == out["shim"][1], "the routine reached ereport; the sweep is void"
    return np.array(out["wts"]), np.array(out["rho"])


@needs_binding
@pytest.mark.fortran
@pytest.mark.parametrize("flag", [False, True], ids=["default", "fix_neg_pvol_wat"])
def test_the_port_is_byte_equal_to_the_compiled_routine(flag):
    t = temperature_grid()
    pmid = np.full(t.size, 1.0e5)
    s = np.full(t.size, 6.31470508962842913e-03)
    rp = np.full(t.size, 1.0e-7)

    want_wts, want_rho = _reference(t, pmid, s, rp, int(flag))
    got_wts, got_rho = vp.vapour(t, pmid, s, fix_neg_pvol_wat=flag)

    np.testing.assert_array_equal(np.asarray(got_wts), want_wts, err_msg="wts")
    np.testing.assert_array_equal(np.asarray(got_rho), want_rho, err_msg="rhosol_strat")


@needs_binding
@pytest.mark.fortran
def test_the_dead_chain_really_is_dead():
    """`rp` is the only argument feeding `kelvin`/`kelvin_out`, and the port
    does not take it at all. Sweeping it over eight orders and requiring both
    outputs to be byte-identical is what licenses the omission."""
    t = np.linspace(200.0, 320.0, 64)
    pmid = np.full(t.size, 1.0e5)
    s = np.full(t.size, 6.31470508962842913e-03)

    baseline = None
    for rp_value in (1.0e-10, 1.0e-8, 1.0e-7, 1.0e-4, 1.0e-2):
        wts, rho = _reference(t, pmid, s, np.full(t.size, rp_value), 1)
        if baseline is None:
            baseline = (wts, rho)
            continue
        np.testing.assert_array_equal(wts, baseline[0], err_msg=f"wts moved at rp={rp_value}")
        np.testing.assert_array_equal(rho, baseline[1], err_msg=f"rho moved at rp={rp_value}")


@needs_binding
@pytest.mark.fortran
def test_the_cancellation_pole_is_reached_and_survives_it():
    """The point the whole grid exists for.

    At the root of `b = ks3 + ks4/T` the denominator is *exactly* zero, so
    `d = a*a` exactly, `SQRT(d) = -a` exactly, the numerator is exactly 0.0 and
    `xsb` is 0/0. Everything downstream is NaN until the clamp, and gfortran's
    `MAX` returns 41.0 there where `jnp.maximum` would propagate the NaN.

    This is a live grid point, not a hypothetical: it is why
    `numerics.fortran_max` exists.
    """
    scalars = ex.extract()["scalars"]
    t = np.array([-scalars["ks4"] / scalars["ks3"]])
    assert scalars["ks3"] + scalars["ks4"] / t[0] == 0.0, "the pole moved; re-derive it"

    pmid, s, rp = np.full(1, 1.0e5), np.full(1, 6.3147050896e-03), np.full(1, 1.0e-7)
    want_wts, want_rho = _reference(t, pmid, s, rp, 0)
    assert want_wts[0] == 41.0, f"expected the clamp to rescue the pole, got {want_wts[0]!r}"

    got_wts, got_rho = vp.vapour(t, pmid, s, fix_neg_pvol_wat=False)
    np.testing.assert_array_equal(np.asarray(got_wts), want_wts)
    np.testing.assert_array_equal(np.asarray(got_rho), want_rho)


def test_fortran_max_keeps_the_first_argument_when_the_second_is_nan():
    """`jnp.maximum` propagates; gfortran's `MAX` does not. Measured against
    the compiled routine in the test above; asserted directly here so the
    primitive has its own guard."""
    assert numerics.fortran_max(41.0, np.nan) == 41.0
    assert np.isnan(numerics.fortran_max(np.nan, 41.0))
    assert numerics.fortran_max(41.0, 50.0) == 50.0
    assert numerics.fortran_max(41.0, 3.0) == 41.0
    assert numerics.fortran_min(99.0, np.nan) == 99.0
    assert numerics.fortran_min(99.0, 3.0) == 3.0


def test_jnp_maximum_would_have_been_wrong_here():
    """Named so the reason `fortran_max` exists cannot be optimised away by
    someone who reads `jnp.maximum` as the obvious spelling."""
    assert np.isnan(np.maximum(41.0, np.nan))
    assert numerics.fortran_max(41.0, np.nan) == 41.0


def test_the_committed_literals_are_not_stale():
    assert ex.main(["--check"]) == 0


def test_the_check_mode_rejects_a_doctored_file(tmp_path, monkeypatch, capsys):
    doctored = tmp_path / "_vapour_literals.py"
    doctored.write_text("KS1 = 0.0\n", encoding="utf-8")
    monkeypatch.setattr(ex, "TARGET", doctored)
    assert ex.main(["--check"]) == 1
    assert "regenerate" in capsys.readouterr().out


def test_only_wts_and_rhosol_strat_are_intent_out():
    """The dead-chain analysis, checked against the source rather than
    remembered. If a future version routes `kelvin` to an output, the omission
    stops being safe and this fails before the port goes quietly wrong."""
    text = SOURCE.read_text(encoding="utf-8")
    outs = set()
    for line in text.splitlines():
        if "INTENT(OUT)" not in line or "::" not in line:
            continue
        # Strip the trailing comment first: `wts(nbox) ! MAX(41.0, WS*100)`
        # contains a comma, and splitting on commas without doing this yields
        # ` WS*100)` as a declared name.
        declaration = line.split("!", 1)[0].split("::")[1]
        outs.update(name.split("(")[0].strip() for name in declaration.split(","))
    outs.discard("")
    assert outs == {"wts", "rhosol_strat"}, outs


def test_the_percent_table_stops_below_the_rounded_clamp():
    """Why `l_fix_neg_pvol_wat` cannot reach `rhosol_strat`: where the arms
    differ, `wts` is 99 or more, `(NINT(wts/5))*5` is 100 or more, and the
    table stops at 95 -- so both arms fall through to 1300.0."""
    assert max(vp.PERCENT) == 95
    for wts in (99.0, 100.0, 103.8):
        assert float(numerics.vapour_round(np.array(wts))) >= 100.0
