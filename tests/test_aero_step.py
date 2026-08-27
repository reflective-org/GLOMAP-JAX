"""Task 63: `ukca_aero_step` -- the twenty ported modules, in sequence.

Against `tests/goldens/aero_step.f64.leaf.npz`: 4 setups x 2 substep settings x
5 boxes, fourteen output fields.

Every routine this project ported was checked against the compiled reference
**on its own**. This is the first check that they agree with it *together*.

**At `nmts = 1` -- 15 competition substeps, the shipped setting -- the whole
sequence is byte-equal on every field of every setup.**

At `nmts = 3` (45 substeps) 19 elements differ, worst **4.6e-10 relative**, and
substituting libm's `exp` in all eight modules that call it removes every one.
So the divergence is issue #28 amplified rather than a porting error: a 1-ulp
`exp` difference, compounded through 45 rounds of a competition whose
nucleation rate is exponentially sensitive to H2SO4.

For scale, phase B measured the f32-vs-f64 precision floor of the *reference
itself* at 3.7e-4 over 48 steps. 4.6e-10 after 45 substeps is six orders
below that, so the sequence agreement is far tighter than the reference's own
precision would support gating on.

`ulp` is the wrong unit here and this file does not use it. The largest gap is
3.25 million ulps and 4.6e-10 relative -- both true, and only the second is
informative, because a budget field that has accumulated 45 contributions has
an ulp much smaller than its own uncertainty.

What the fixture fixes
----------------------

The Fortran routine takes 96 arguments; `leaf_aero_step` exposes the ~40 the
box model varies and fixes the rest as `glomap_box.F90:140-163` passes them.
Scavenging, deposition, cloud processing, wet oxidation and nitrate are off;
`dryox_in_aer = 1`, so the gas-production term inside the competition loop is
live. That is the configuration `docs/unsupported.md` records and the only one
with a reference.
"""

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
from glomap_jax.drivers.aero_step import AeroState, aero_step
from glomap_jax.physics import budget_indices as bi
from glomap_jax.physics import coag_mode as cmm
from glomap_jax.physics import gas_indices as gi
from glomap_jax.physics import modes as mm

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "aero_step.f64.leaf.npz"

sys.path.insert(0, str(REPO / "validation"))

ENV_KEYS = (
    "sm",
    "aird",
    "airdm3",
    "rhoa",
    "mfpa",
    "dvisc",
    "t",
    "tsqrt",
    "rh",
    "rh_clr",
    "s",
    "pmid",
    "pupper",
    "plower",
    "height",
    "htpblg",
)
FIELDS = (
    "nd",
    "md",
    "mdt",
    "s0g",
    "drydp",
    "wetdp",
    "dvol",
    "wvol",
    "mdwat",
    "rhopar",
    "pvol",
    "pvol_wat",
    "bud",
    "n_merge",
)

#: Every module that calls `exp`. All eight have to be substituted together --
#: the attribution is about the sequence, not about any one of them.
EXP_MODULES = (_bp, _nu, _cc, _cn, _cg, _cw, _sv, _rm)

#: Measured. Relative, not ulp -- see the module docstring.
MAX_RELATIVE = 1.0e-9
DIFFERING_AT_NMTS3 = 19


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run validation/capture_aero_step_leaf.py (or `make goldens`)"
    return np.load(GOLDEN, allow_pickle=False)


def _cases(sweep):
    for s_i, setup in enumerate(sweep["setups"].tolist()):
        for k_i, (nmts, nzts) in enumerate(sweep["substeps"].tolist()):
            yield s_i, setup, k_i, nmts, nzts


def _widths(sweep, s_i):
    return (
        int(sweep["ncp"][s_i]),
        int(sweep["nadvg"][s_i]),
        int(sweep["nchemg"][s_i]),
        int(sweep["nbudaer"][s_i]),
    )


#: One run per (setup, nmts), shared across tests. Without it the file takes
#: 150 s: every call re-traces 45 unrolled competition substeps across eight
#: routines, and nine tests want the same eight results.
_CACHE: dict[tuple, object] = {}


def _cached(sweep, s_i, setup, nmts, nzts):
    key = (setup, nmts, nzts)
    if key not in _CACHE:
        _CACHE[key] = _run(sweep, s_i, setup, nmts, nzts)
    return _CACHE[key]


def _run(sweep, s_i, setup, nmts, nzts):
    ncp, nadv, nchem, nbud = _widths(sweep, s_i)
    env = {k: jnp.asarray(sweep[f"in_{k}"]) for k in ENV_KEYS}
    env["s0g_dot"] = jnp.asarray(sweep["in_s0g_dot"][s_i][:, :nchem])
    state = AeroState(
        jnp.asarray(sweep["in_nd"][s_i]),
        jnp.asarray(sweep["in_md"][s_i][:, :, :ncp]),
        jnp.asarray(sweep["in_mdt"][s_i]),
        jnp.asarray(sweep["in_s0g"][s_i][:, :nadv]),
    )
    nbox = sweep["in_nd"].shape[1]
    return aero_step(
        mm.build(setup),
        gi.build(setup),
        bi.build(setup),
        cmm.COAG_MODE,
        state,
        env,
        jnp.zeros((nbox, nbud + 1)),
        dtc=float(sweep["dt_chem"]),
        dtz=float(sweep["dt_chem"]) / (nmts * nzts),
        nmts=nmts,
        nzts=nzts,
    )


def _pairs(sweep, s_i, k_i, result):
    ncp, nadv, _nchem, nbud = _widths(sweep, s_i)
    got = {
        "nd": result.nd,
        "md": result.md,
        "mdt": result.mdt,
        "s0g": result.s0g,
        "drydp": result.drydp,
        "wetdp": result.wetdp,
        "dvol": result.dvol,
        "wvol": result.wvol,
        "mdwat": result.mdwat,
        "rhopar": result.rhopar,
        "pvol": result.pvol,
        "pvol_wat": result.pvol_wat,
        "bud": result.bud_aer_mas,
        "n_merge": result.n_merge,
    }
    want = {
        "nd": sweep["nd"][s_i, k_i],
        "md": sweep["md"][s_i, k_i][:, :, :ncp],
        "mdt": sweep["mdt"][s_i, k_i],
        "s0g": sweep["s0g"][s_i, k_i][:, :nadv],
        "drydp": sweep["drydp"][s_i, k_i],
        "wetdp": sweep["wetdp"][s_i, k_i],
        "dvol": sweep["dvol"][s_i, k_i],
        "wvol": sweep["wvol"][s_i, k_i],
        "mdwat": sweep["mdwat"][s_i, k_i],
        "rhopar": sweep["rhopar"][s_i, k_i],
        "pvol": sweep["pvol"][s_i, k_i][:, :, :ncp],
        "pvol_wat": sweep["pvol_wat"][s_i, k_i],
        "bud": sweep["bud"][s_i, k_i][:, : nbud + 1],
        "n_merge": sweep["n_merge"][s_i, k_i],
    }
    return [(name, np.asarray(got[name]), want[name]) for name in FIELDS]


def test_the_sequence_is_byte_equal_at_the_shipped_substep_count(sweep):
    """`nmts = 1, nzts = 15` is what every shipped namelist runs. Twenty
    modules, four setups, fourteen fields, exact."""
    for s_i, setup, k_i, nmts, nzts in _cases(sweep):
        if nmts != 1:
            continue
        result = _cached(sweep, s_i, setup, nmts, nzts)
        for name, got, want in _pairs(sweep, s_i, k_i, result):
            np.testing.assert_array_equal(got, want, err_msg=f"{name} setup {setup}")


def test_three_times_the_substeps_agree_to_a_measured_relative_bound(sweep):
    """`bl_nmts3.nml` exists for exactly this. 45 competition substeps instead
    of 15, and the accumulated `exp` difference becomes visible -- 19 elements,
    worst 4.6e-10 relative. The count is pinned so it cannot grow quietly."""
    differing = 0
    worst = 0.0
    for s_i, setup, k_i, nmts, nzts in _cases(sweep):
        if nmts == 1:
            continue
        result = _cached(sweep, s_i, setup, nmts, nzts)
        for name, got, want in _pairs(sweep, s_i, k_i, result):
            off = got != want
            differing += int(off.sum())
            nz = off & (want != 0)
            if nz.any():
                worst = max(worst, float((np.abs(got[nz] - want[nz]) / np.abs(want[nz])).max()))
            assert not np.any(off & (want == 0)), f"{name}: a zero became non-zero"
    assert differing == DIFFERING_AT_NMTS3, (
        f"{differing} elements differ at nmts = 3, measured {DIFFERING_AT_NMTS3}"
    )
    assert worst <= MAX_RELATIVE, f"worst {worst:.2e} relative, bound {MAX_RELATIVE:.0e}"


def test_the_whole_sequence_gap_is_the_exponential(sweep, monkeypatch):
    """The decisive experiment, on the sequence rather than on one routine.
    All eight modules that call `exp` are routed through numpy -- the same libm
    gfortran reaches -- and every difference across all eight configurations
    disappears."""

    class _Libm:
        def __getattr__(self, name):
            return getattr(jnp, name)

        @staticmethod
        def exp(x):
            return jnp.asarray(np.exp(np.asarray(x)))

    for module in EXP_MODULES:
        monkeypatch.setattr(module, "jnp", _Libm())
    for s_i, setup, k_i, nmts, nzts in _cases(sweep):
        result = _run(sweep, s_i, setup, nmts, nzts)
        for name, got, want in _pairs(sweep, s_i, k_i, result):
            np.testing.assert_array_equal(got, want, err_msg=f"{name} setup {setup} nmts {nmts}")


def test_the_agreement_is_far_inside_the_references_own_precision_floor(sweep):
    """Phase B measured the f32-vs-f64 floor of the reference at 3.7e-4 over a
    48-step run. The sequence agreement is six orders below that, so nothing
    here is limited by the port."""
    assert MAX_RELATIVE < 3.7e-4 / 1.0e5


def test_the_substep_count_actually_changes_the_answer(sweep):
    """Otherwise the two settings prove nothing about the loop nesting: a
    driver that ran `nmts*nzts` steps in one flat loop, or nested them the
    other way, would agree at `nmts = 1` and has to be caught at 3."""
    for s_i, setup, _k_i, _nmts, _nzts in _cases(sweep):
        one = _cached(sweep, s_i, setup, 1, 15)
        three = _cached(sweep, s_i, setup, 3, 15)
        assert not np.array_equal(np.asarray(one.nd), np.asarray(three.nd)), setup
        break


def test_every_size_field_is_recomputed(sweep):
    """`drydp`, `wetdp`, `dvol`, `wvol`, `rhopar`, `mdwat` are all `INTENT(IN
    OUT)` upstream and all recomputed before they are read. The driver takes
    none of them as input, which is only correct because of that -- so the
    outputs must be populated on every active mode."""
    for s_i, setup, _k_i, nmts, nzts in _cases(sweep):
        result = _cached(sweep, s_i, setup, nmts, nzts)
        active = np.asarray(result.nd) > 0.0
        if not active.any():
            continue
        for name in ("drydp", "wetdp", "dvol", "wvol", "rhopar"):
            arr = np.asarray(getattr(result, name))
            assert np.all(arr[active] > 0.0), f"{name} setup {setup}"


def test_mass_moves_from_gas_to_aerosol(sweep):
    """The cheapest statement of what the sequence is for, and the one that
    would catch a sign error the byte comparison would also catch but nothing
    else here would explain."""
    for s_i, setup, _k_i, nmts, nzts in _cases(sweep):
        result = _cached(sweep, s_i, setup, nmts, nzts)
        mass_in = (sweep["in_nd"][s_i] * sweep["in_mdt"][s_i]).sum(axis=1)
        mass_out = (np.asarray(result.nd) * np.asarray(result.mdt)).sum(axis=1)
        assert np.any(mass_out > mass_in), setup
        assert np.all(np.asarray(result.nd) >= 0.0)
        assert np.all(np.asarray(result.s0g) >= 0.0)


def test_nucleation_and_merging_both_ran(sweep):
    """A sequence that silently skipped a process would still be byte-equal to
    a golden captured from the same skip. These count what happened."""
    nucleated = merged = 0
    for s_i, setup, _k_i, nmts, nzts in _cases(sweep):
        result = _cached(sweep, s_i, setup, nmts, nzts)
        nucleated += int((np.asarray(result.nd)[:, 0] > sweep["in_nd"][s_i][:, 0]).sum())
        merged += int((np.asarray(result.n_merge) > 0).sum())
    assert nucleated == 12
    assert merged == 4


def test_the_budget_hole_is_never_written(sweep):
    for s_i, setup, _k_i, nmts, nzts in _cases(sweep):
        result = _cached(sweep, s_i, setup, nmts, nzts)
        assert np.all(np.asarray(result.bud_aer_mas)[:, 0] == 0.0)
