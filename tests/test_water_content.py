"""Task 40 — `ukca_water_content_v`, byte-equal against the compiled routine.

Byte equality, not the plan's `RTOL_TRANSCENDENTAL`: this routine calls no
transcendental at all. Its arithmetic is integer-literal powers, adds, one
`MIN` and one divide.

**The reference is driven without `wrap_init`, and that is not a shortcut.**
`glomap_box_config_mod.F90:322` hardcodes `l_fix_ukca_water_content = .TRUE.`,
and `init_ukca_for_box` then runs `init_state` -> `ukca_volume_mode` ->
`ukca_water_content_v`, which patches its own SAVEd coefficient table in place
at `:235` and never restores it. So after init the unpatched table is gone for
the life of the process, and a both-settings comparison written the obvious way
compares the patched table against itself and passes. Issue #22.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
F2PY = REPO / "validation" / "f2py"
NAMELIST = REPO / "fortran" / "namelists" / "boundary_layer.nml"

from glomap_jax.physics import water_content as wc  # noqa: E402
from glomap_jax.physics import water_tables as wt  # noqa: E402

needs_binding = pytest.mark.skipif(
    not sorted(F2PY.glob("glomap_f2py*.so")),
    reason="binding not built; run validation/build_f2py.sh",
)

# H+, NH4+, Cl-, SO4^2- -- the four the box caller can actually populate.
BOX_SPECIES = (1, 3, -4, -2)


def _reference(cl, ions, rh, mask, flag: int, *, init: bool = False) -> np.ndarray:
    script = f"""
import json, sys
import numpy as np
sys.path.insert(0, {str(F2PY)!r})
import glomap_f2py as g
g.wrap_ereport_reset()
assert int(g.wrap_set_fix_water_content({flag})) == 0
if {int(init)}:
    assert int(g.wrap_init({str(NAMELIST)!r})) == 0
    assert int(g.wrap_set_fix_water_content({flag})) == 0
before = [int(v) for v in g.wrap_ereport_count()]
out, ierr = g.leaf_water_content(
    np.array(json.loads({json.dumps(mask.tolist())!r}), dtype=np.int32),
    np.array(json.loads({json.dumps(ions.tolist())!r}), dtype=np.int32),
    np.array(json.loads({json.dumps(cl.tolist())!r})),
    np.array(json.loads({json.dumps(rh.tolist())!r})))
after = [int(v) for v in g.wrap_ereport_count()]
print("@@R@@" + json.dumps({{"ierr": int(ierr), "shim": [before, after],
                             "wc": np.asarray(out).tolist()}}))
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    out = json.loads(proc.stdout[proc.stdout.rindex("@@R@@") + 5 :])
    assert out["ierr"] == 0, out["ierr"]
    assert out["shim"][0] == out["shim"][1], "the routine reached ereport; the comparison is void"
    return np.array(out["wc"])


def _grid():
    """All presence combinations the caller can reach, plus the ones it cannot.

    The humidity axis crosses several `rh_min` floors, which is what makes the
    unfixed arm's ratchet observable: there the floor raises `aw` permanently
    for every later pair, while the fixed arm re-reads the original humidity.
    """
    rows = []
    for combo in range(1 << len(BOX_SPECIES)):
        for magnitude in (1e-20, 1e-16, 1e-13, 1e-11):
            for rh in (0.02, 0.19, 0.35, 0.5, 0.62, 0.85):
                cl = np.zeros(8)
                ions = np.zeros(8, dtype=np.int32)
                for bit, species in enumerate(BOX_SPECIES):
                    if combo >> bit & 1:
                        cl[wt.ion_slot(species)] = magnitude * (1 + bit)
                        ions[wt.ion_slot(species)] = 1
                rows.append((cl, ions, rh))

    # Nitrate: pair (1,-3) is the only one the patched coefficient touches, and
    # it is dead through the box caller because ncp = 6 while cp_no3 = 7.
    for rh in (0.05, 0.3, 0.7):
        cl = np.zeros(8)
        ions = np.zeros(8, dtype=np.int32)
        for species in (1, -3):
            cl[wt.ion_slot(species)] = 1e-13
            ions[wt.ion_slot(species)] = 1
        rows.append((cl, ions, rh))

    cl = np.array([r[0] for r in rows])
    ions = np.array([r[1] for r in rows], dtype=np.int32)
    rh = np.array([r[2] for r in rows])
    mask = np.ones(len(rows), dtype=np.int32)
    mask[::7] = 0  # interleaved, so a port that ignores the mask is visible
    return cl, ions, rh, mask


@needs_binding
@pytest.mark.fortran
@pytest.mark.parametrize("flag", [0, 1], ids=["unfixed", "fixed"])
def test_the_port_is_byte_equal_to_the_compiled_routine(flag):
    cl, ions, rh, mask = _grid()
    want = _reference(cl, ions, rh, mask, flag)
    got = wc.water_content(
        cl, rh, ions.astype(bool), mask.astype(bool), fix_water_content=bool(flag)
    )
    np.testing.assert_array_equal(np.asarray(got), want)


@needs_binding
@pytest.mark.fortran
def test_the_two_flag_settings_genuinely_differ():
    """Otherwise the parametrised test above is one test wearing two hats.

    Asserted for the flag's *two independent effects* separately, because a
    capture that reached only one of them would look like full coverage: the
    patched coefficient shows up on nitrate rows, and the ratcheting humidity
    floor shows up on low-`rh` rows that carry no nitrate at all.
    """
    cl, ions, rh, mask = _grid()
    unfixed = _reference(cl, ions, rh, mask, 0)
    fixed = _reference(cl, ions, rh, mask, 1)
    differ = unfixed != fixed
    assert differ.any(), "the two arms agree everywhere; the fixture reaches neither effect"

    nitrate = ions[:, wt.ion_slot(-3)] == 1
    assert (differ & nitrate).any(), "no nitrate row differs; the patched coefficient is untested"
    assert (differ & ~nitrate).any(), (
        "only nitrate rows differ; the ratcheting humidity floor is untested, and it is "
        "the half of this flag that reaches configurations the box model can actually run"
    )


@needs_binding
@pytest.mark.fortran
def test_after_wrap_init_the_unfixed_arm_is_unreachable():
    """The finding that made this file drive the reference without init.

    `init_ukca_for_box` hardcodes the flag on and then runs the routine through
    `init_state`, so the in-place patch has already fired by the time any
    caller can express a preference. This pins that: with init, both flag
    values return the *same* numbers, and they are the fixed ones.

    If this ever starts failing, the latch has been fixed upstream and the
    `init=False` route in `leaf_common.run_child` can go.
    """
    cl = np.zeros((1, 8))
    ions = np.zeros((1, 8), dtype=np.int32)
    for species in (1, -3):
        cl[0, wt.ion_slot(species)] = 1e-13
        ions[0, wt.ion_slot(species)] = 1
    rh, mask = np.array([0.7]), np.ones(1, dtype=np.int32)

    with_init_off = _reference(cl, ions, rh, mask, 0, init=True)
    with_init_on = _reference(cl, ions, rh, mask, 1, init=True)
    cold_off = _reference(cl, ions, rh, mask, 0)

    np.testing.assert_array_equal(with_init_off, with_init_on)
    assert with_init_off[0] != cold_off[0], (
        "the post-init result matches the cold unfixed one, so the latch is gone"
    )


@needs_binding
@pytest.mark.fortran
def test_the_pair_scan_is_loop_carried():
    """ "Compute all twelve pairs, then apply" is a different model.

    An early pair drains the shared ion pools, so a later pair competing for
    the same ion sees less of it. Built to bite: H+ is scarce relative to the
    two anions that want it.
    """
    cl = np.zeros((1, 8))
    ions = np.zeros((1, 8), dtype=np.int32)
    cl[0, wt.ion_slot(1)] = 1e-14  # H+, scarce
    cl[0, wt.ion_slot(-4)] = 1e-12  # Cl-, plentiful
    cl[0, wt.ion_slot(-2)] = 1e-12  # SO4--, plentiful
    for species in (1, -4, -2):
        ions[0, wt.ion_slot(species)] = 1
    rh, mask = np.array([0.5]), np.ones(1, dtype=np.int32)

    want = _reference(cl, ions, rh, mask, 1)
    got = wc.water_content(cl, rh, ions.astype(bool), mask.astype(bool), fix_water_content=True)
    np.testing.assert_array_equal(np.asarray(got), want)

    # The unordered model: every pair sees the full pools.
    naive = 0.0
    for cation, anion in wc.PAIRS:
        ic, ia = wt.ion_slot(cation), wt.ion_slot(anion)
        if not (ions[0, ic] and ions[0, ia]):
            continue
        n_c, n_a = wc.stoichiometry(cation, anion)
        naive += min(cl[0, ic] / n_c, cl[0, ia] / n_a)
    carried = sum(
        float(v[0])
        for v in wc._pair_concentrations(cl, ions.astype(bool), mask.astype(bool))[0].values()
    )
    assert naive != carried, "the pools are not being drawn down; the scan is not carried"


def test_the_stoichiometry_is_crossed():
    """`n(ic) = z(ia)` and `n(ia) = z(ic)`. For 2H+ SO4--, the divisor on the
    *cation* is the anion's charge of 2. Reading it uncrossed swaps them."""
    assert wc.stoichiometry(1, -2) == (2.0, 1.0)
    assert wc.stoichiometry(1, -4) == (1.0, 1.0)


def test_the_charge_normalising_branch_is_dead_for_the_shipped_charges():
    """`:255-259` divides `n` through when the two charges are equal and not 1.
    No cation has charge 2, so it never fires. Pinned rather than deleted: a
    future divalent cation makes it live, and the port should already be right.
    """
    cations = [wc.ION_CHARGE[wt.ion_slot(c)] for c in range(1, wt.NCATION + 1)]
    assert set(cations) == {1.0}, f"a cation charge changed: {cations}"
    for cation, anion in wc.PAIRS:
        z_c = wc.ION_CHARGE[wt.ion_slot(cation)]
        z_a = wc.ION_CHARGE[wt.ion_slot(anion)]
        assert not (abs(z_c - z_a) < 1e-15 and abs(z_a - 1.0) > 1e-15)


def test_the_pair_order_is_cation_outer_anion_inner():
    """The carry makes the order load-bearing, so it is asserted rather than
    left to the comprehension that produces it."""
    assert wc.PAIRS[:5] == ((1, -4), (1, -3), (1, -2), (1, -1), (2, -4))
    assert len(wc.PAIRS) == 12
