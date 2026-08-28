"""Gate C: the ported model against the committed trajectory goldens.

`docs/harness.md` maps four gates. Gate 0 asks whether the same predicate went
the same way, gate A whether one routine agrees at machine precision, gate B
which call diverged. **Gate C asks whether the run agrees** -- and it is the
only one that can say the *model* is right rather than its parts.

This is the first test in the project to reach it. `drivers/box.py` builds the
environment and initial state from a namelist, steps `aero_step` 48 times, and
writes the same 26 diagnostics `glomap_box_output_mod` does. The comparison is
against the goldens captured from the compiled box model on the pinned
toolchain.

Result, across `boundary_layer`, `free_troposphere` and `marine_bcoc`:

* every **aerosol** column -- number, dry and wet diameter, density, and the
  per-component masses -- agrees to **5.4e-13 relative** over 48 steps, and is
  **bit-identical** once libm's `exp` replaces XLA's in the eight modules that
  call it;
* the two **gas** columns agree to **2.7e-7**, which is the goldens' own
  quantisation and not a divergence -- see below.

For scale, phase B measured the f32-vs-f64 precision floor of the reference at
**3.7e-4** over the same 48 steps, and **0.80** for `marine_bcoc`, where ageing
depletes the Aitken insoluble mode over seven orders of magnitude and the f32
reference loses the residual (issue #14). This port tracks that same column to
**4.1e-13**.

The gas columns are gated separately, and why
---------------------------------------------

`H2SO4_cm3` and `SEC_ORG_cm3` are stored in the goldens with seven significant
digits while every other column carries seventeen -- `23739040.0` against
`3.1631379098463355`. Seven digits is exactly what `ES14.6` gives, and
`validation/patches/0001-high-precision-output.patch` is supposed to have
bumped those two lines to `ES24.16` along with the rest. Issue #32.

So the gas bound here is the goldens' resolution, not the port's. It is a
separate constant with its own name so that re-capturing the goldens makes the
test fail rather than silently pass at a loose value -- and
`test_the_gas_columns_are_still_quantised` asserts the quantisation directly,
so the day it goes away, this file has to be updated deliberately.
"""

import re
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import glomap_jax.physics.binapara as _bp
import glomap_jax.physics.coag_coff as _cg
import glomap_jax.physics.coagwithnucl as _cw
import glomap_jax.physics.cond_coff as _cc
import glomap_jax.physics.conden as _cn
import glomap_jax.physics.nucleation as _nu
import glomap_jax.physics.remode as _rm
import glomap_jax.physics.solvecoagnucl as _sv
from glomap_jax.drivers.box import box_env, init_state, run
from glomap_jax.physics import budget_indices as bi
from glomap_jax.physics import coag_mode as cmm
from glomap_jax.physics import gas_indices as gi
from glomap_jax.physics import modes as mm

REPO = Path(__file__).resolve().parents[1]
NAMELISTS = REPO / "fortran" / "namelists"
GOLDENS = REPO / "tests" / "goldens"

sys.path.insert(0, str(REPO / "validation"))

#: Measured over 48 steps, per case:
#:
#:   boundary_layer     4.4e-15   worst in N_aitsol_cm3
#:   free_troposphere   5.4e-13   worst in N_accsol_cm3
#:   marine_bcoc        4.1e-13   worst in N_aitins_cm3
#:
#: The bound is the next round number above the worst of those. It was set at
#: 1e-13 first, from the boundary_layer figure alone, and two cases failed --
#: which is the right way round for a guess to be caught.
#:
#: `N_aitins_cm3` being the worst in marine_bcoc is not a coincidence: phase B
#: recorded that ageing depletes the Aitken insoluble mode over seven orders of
#: magnitude there, and that the f32 reference loses the residual entirely
#: (0.80 relative, issue #14). This port tracks it to 4e-13.
#:
#: The residual is `exp` (issue #28); with libm's the aerosol columns are
#: bit-identical, which `test_libm_exp_makes_the_aerosol_trajectory_exact`
#: asserts.
AEROSOL_RELATIVE = 1.0e-12

#: NOT the port's error. The goldens store these two columns at seven
#: significant digits (issue #32), so this is their resolution. Named apart
#: from AEROSOL_RELATIVE so a re-capture cannot quietly reuse it.
GAS_COLUMN_QUANTISATION = 5.0e-7

CASES = ("boundary_layer", "free_troposphere", "marine_bcoc")

EXP_MODULES = (_bp, _nu, _cc, _cn, _cg, _cw, _sv, _rm)


def _namelist(path: Path) -> dict:
    """Enough of a Fortran namelist reader for the three shipped cases.

    Deliberately small: it reads scalars and comma-separated reals and nothing
    else, and every key it fails to parse is simply absent, so a caller that
    needs one gets a `KeyError` rather than a default it did not choose.
    """
    text = re.sub(r"!.*", "", path.read_text(encoding="utf-8"))
    out: dict[str, list[float] | str] = {}
    for match in re.finditer(r"(\w+)\s*=\s*([^\n/]+)", text):
        key, raw = match.group(1), match.group(2).strip().rstrip(",")
        if raw.startswith("'"):
            out[key] = raw.strip("'").split("'")[0]
            continue
        values, ok = [], True
        for piece in raw.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                values.append(float(piece))
            except ValueError:
                ok = False
                break
        if ok and values:
            out[key] = values
    return out


def _trajectory(case: str):
    """Run the ported model for one shipped namelist."""
    n = _namelist(NAMELISTS / f"{case}.nml")
    setup = int(n["i_mode_setup"][0])
    tables, gas, budget = mm.build(setup), gi.build(setup), bi.build(setup)
    env = box_env(
        1,
        t=n["temperature"][0],
        pmid=n["pressure"][0],
        rh=n["rel_humid"][0],
        spec_humid=n.get("spec_humid", [-1.0])[0],
        height=n["height"][0],
        pbl_height=n["pbl_height"][0],
        box_volume=n.get("box_volume", [1.0])[0],
    )
    mfrac = None
    if "mfrac_init" in n:
        ncp = int(tables.ncp)
        flat = n["mfrac_init"]
        if len(flat) >= 8 * ncp:
            mfrac = [flat[i * ncp : (i + 1) * ncp] for i in range(8)]
    state = init_state(
        tables,
        gas,
        env,
        nbox=1,
        nadvg=int(gas.nadvg),
        nchemg=int(gas.nchemg),
        nd_init=n["nd_init"],
        dp_init=n["dp_init"],
        mfrac_init=mfrac,
        h2so4_init=n.get("h2so4_init", [0.0])[0],
        sec_org_init=n.get("sec_org_init", [0.0])[0],
        h2so4_prod=n.get("h2so4_prod", [0.0])[0],
        sec_org_prod=n.get("sec_org_prod", [0.0])[0],
    )
    switches = {
        name: int(n.get(name, [default])[0])
        for name, default in (
            ("cond_on", 1),
            ("nucl_on", 1),
            ("coag_on", 1),
            ("bln_on", 0),
            ("icoag", 1),
            ("imerge", 1),
            ("ifuchs", 1),
            ("idcmfp", 1),
            ("icondiam", 1),
            ("ibln", 1),
            ("i_nuc_method", 2),
        )
    }
    _final, rows = run(
        tables,
        gas,
        budget,
        cmm.COAG_MODE,
        state,
        env,
        nsteps=int(n["nsteps"][0]),
        dt_chem=n["dt_chem"][0],
        nmts=int(n["nmts"][0]),
        nzts=int(n["nzts"][0]),
        output_every=int(n.get("output_every", [1])[0]),
        nbudaer=int(budget.nbudaer),
        **switches,
    )
    golden = np.load(GOLDENS / f"{case}.f64.trajectory.npz", allow_pickle=False)
    return rows, golden["values"], [str(c) for c in golden["columns"]]


_CACHE: dict[str, tuple] = {}


def _cached(case: str):
    """One 48-step run per case, shared across tests -- each takes ~50 s."""
    if case not in _CACHE:
        _CACHE[case] = _trajectory(case)
    return _CACHE[case]


def _split(cols):
    gas = [j for j, c in enumerate(cols) if c in ("H2SO4_cm3", "SEC_ORG_cm3")]
    aerosol = [j for j, c in enumerate(cols) if j not in gas and not c.startswith("time")]
    return aerosol, gas


@pytest.mark.parametrize("case", CASES)
def test_the_aerosol_trajectory_matches_the_golden(case):
    """48 steps, every aerosol diagnostic, against the compiled box model."""
    rows, want, cols = _cached(case)
    assert rows.shape == want.shape, f"{rows.shape} vs {want.shape}"
    aerosol, _gas = _split(cols)
    worst, where = 0.0, None
    for j in aerosol:
        nz = want[:, j] != 0.0
        if not nz.any():
            np.testing.assert_array_equal(rows[:, j], want[:, j], err_msg=cols[j])
            continue
        rel = np.abs(rows[nz, j] - want[nz, j]) / np.abs(want[nz, j])
        if rel.max() > worst:
            worst, where = float(rel.max()), cols[j]
    assert worst <= AEROSOL_RELATIVE, f"{case}: worst {worst:.3e} in {where}"


@pytest.mark.parametrize("case", CASES)
def test_the_gas_trajectory_matches_to_the_goldens_own_resolution(case):
    """Bounded by issue #32, not by the port. See the module docstring."""
    rows, want, cols = _cached(case)
    _aerosol, gas = _split(cols)
    for j in gas:
        nz = want[:, j] != 0.0
        if not nz.any():
            continue
        rel = np.abs(rows[nz, j] - want[nz, j]) / np.abs(want[nz, j])
        assert rel.max() <= GAS_COLUMN_QUANTISATION, f"{case} {cols[j]}: {rel.max():.3e}"


@pytest.mark.parametrize("case", CASES)
def test_the_gas_columns_are_still_quantised(case):
    """Issue #32, asserted so the loose bound above cannot outlive it.

    Every aerosol column round-trips through 17 significant digits; the two gas
    columns do not. When the goldens are re-captured with the overlay applied
    to all 26 columns this fails, and `GAS_COLUMN_QUANTISATION` has to come
    down deliberately rather than by nobody noticing.
    """
    _rows, want, cols = _cached(case)
    aerosol, gas = _split(cols)

    def significant(x: float) -> int:
        return len(f"{x:.17e}".split("e")[0].replace(".", "").rstrip("0"))

    gas_digits = max(significant(want[1, j]) for j in gas if want[1, j] != 0.0)
    aerosol_digits = max(significant(want[1, j]) for j in aerosol if want[1, j] != 0.0)
    assert gas_digits <= 9, (
        f"{case}: the gas columns now carry {gas_digits} significant digits -- issue "
        "#32 may be fixed, in which case tighten GAS_COLUMN_QUANTISATION"
    )
    assert aerosol_digits >= 15, f"{case}: the aerosol columns lost precision too"


def test_libm_exp_makes_the_aerosol_trajectory_exact(monkeypatch):
    """The attribution, at the level of a whole run rather than one routine.

    48 steps x 15 competition substeps x eight routines, and every aerosol
    column comes back bit-identical once `exp` reaches the same libm gfortran
    does. One case, because each run costs ~50 s and the claim is about the
    mechanism rather than the case.
    """

    class _Libm:
        def __getattr__(self, name):
            return getattr(jnp, name)

        @staticmethod
        def exp(x):
            return jnp.asarray(np.exp(np.asarray(x)))

    for module in EXP_MODULES:
        monkeypatch.setattr(module, "jnp", _Libm())
    rows, want, cols = _trajectory("boundary_layer")
    aerosol, _gas = _split(cols)
    for j in aerosol:
        np.testing.assert_array_equal(rows[:, j], want[:, j], err_msg=cols[j])


@pytest.mark.parametrize("case", CASES)
def test_the_run_has_the_shape_the_namelist_asks_for(case):
    """49 rows for 48 steps: the initial state plus one per step. A driver that
    recorded before rather than after `aero_step`, or that dropped the initial
    row, would still pass a tolerance comparison on 48 of them."""
    rows, want, _cols = _cached(case)
    n = _namelist(NAMELISTS / f"{case}.nml")
    assert rows.shape[0] == int(n["nsteps"][0]) + 1 == want.shape[0]
    np.testing.assert_array_equal(rows[:, 0], want[:, 0])  # time_s
    assert rows[0, 0] == 0.0 and rows[-1, 0] == n["nsteps"][0] * n["dt_chem"][0]
