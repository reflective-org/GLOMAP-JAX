"""Task 54: `ukca_conden`, the widest routine in the port.

Against `tests/goldens/conden.f64.leaf.npz`: 7 setups x 2 switch combos x 4
`(ifuchs, idcmfp, icondiam)` configurations, all seven outputs.

**Byte-equal on 54 of the 56 configurations. 348 elements differ across the
other two, and substituting libm's `exp` removes every one.** Issue #28's
fourth routine, and its most sharply localised appearance yet: the two that
differ are setup 8 with `l_dust_mp_ageing` on and `icondiam = 2`, and nothing
else. `icondiam = 1` sets every `aa_modes` entry to `0.0`, so
`y2 = EXP(0.5*aa*aa*LOG(sigma)**2)` is `EXP(0)` -- exactly `1.0` on both sides.
The gap needs a non-zero exponent *and* a mode where XLA and libm disagree, and
modes 6 and 7 only enter the loop when `topmode` moves off `mode_ait_insol`.

The thirty write sites are generated, not transcribed
------------------------------------------------------

`_conden_literals.py` comes from `validation/extract_conden_literals.py`, which
parses the second mode loop and asserts what it found. Four hundred lines of
near-identical `WHERE` blocks is exactly where a hand port drops one or attaches
it to the wrong mode, and no golden comparison would localise that: a missing
site shows up as a slightly smaller aerosol.

What is pinned here beyond the numbers:

* **A budget index gates the mass.** `deltams` is zeroed and assigned only
  inside `IF (nmascond... > 0)`, and `md`/`mdt` are updated from it. The
  54-pair alignment that makes this inert is re-derived on every run from
  `budget_indices`, `gas_indices` and `modes` -- issue #30.

* **`ageterm1`'s second index is a soluble mode name used as an ordinal.** The
  generated table stores the ordinal and the extractor asserts
  `ageterm == nc_mode - 4`, so the misleading name cannot be copied across.

* **Two fidelity flags, both defaulting to the Fortran.** UP-10's
  `conden_insol_num_eps_by_sol_mode` and issue #29's
  `conden_ocaccins_double_count`; the second is exercised at both settings
  against real data, and the test asserts the two arms genuinely differ.

* **UP-4 is unreachable and reproduced anyway.** `delgc_cond > gc` cannot fire,
  and the port contains the branch regardless -- "cannot fire" is an argument,
  and the invariant test is what makes it a measurement.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import glomap_jax.physics.cond_coff as cc_mod
import glomap_jax.physics.conden as cn
from glomap_jax.config.fidelity import FidelityConfig
from glomap_jax.physics import budget_indices as bi
from glomap_jax.physics import gas_indices as gi
from glomap_jax.physics import modes as mm
from glomap_jax.physics._conden_literals import DUPLICATED, WRITE_SITES

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "conden.f64.leaf.npz"

sys.path.insert(0, str(REPO / "validation"))

import extract_conden_literals as extractor  # noqa: E402

#: Measured. Only setup 8 / dust_ageing / icondiam = 2 differs, and only
#: through `exp`.
EXP_ULP = 5
EXP_AFFECTED = {"md": 71, "mdt": 23, "bud": 159, "ageterm1": 85, "s_cond_s": 10}
EXACT_FIELDS = ("gc", "delgc")


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run validation/capture_conden_leaf.py (or `make goldens`)"
    return np.load(GOLDEN, allow_pickle=False)


def _cases(sweep):
    for s_i, setup in enumerate(sweep["setups"].tolist()):
        for c_i, combo in enumerate(str(c) for c in sweep["combos"]):
            for k_i, config in enumerate(sweep["config"].tolist()):
                yield s_i, setup, c_i, combo, k_i, tuple(config)


def _run(sweep, s_i, setup, c_i, combo, config, *, fidelity=None):
    g, b = gi.build(setup), bi.build(setup)
    ncp = int(sweep["ncp"][s_i, c_i])
    nchemg = int(sweep["nchemg"][s_i, c_i])
    nbud = int(sweep["nbudaer"][s_i, c_i])
    tables = mm.build(setup, l_dust_mp_ageing=(combo == "dust_ageing"))
    nbox = sweep["in_nd"].shape[0]
    ifuchs, idcmfp, icondiam = config
    return (
        cn.conden(
            tables,
            g,
            b,
            jnp.asarray(sweep["in_nd"]),
            jnp.asarray(sweep["in_tsqrt"]),
            jnp.asarray(sweep["in_rhoa"]),
            jnp.asarray(sweep["in_airdm3"]),
            jnp.asarray(sweep["in_wetdp"]),
            jnp.asarray(sweep["in_pmid"]),
            jnp.asarray(sweep["in_t"]),
            jnp.asarray(sweep["in_md"][s_i, c_i][:, :, :ncp]),
            jnp.asarray(sweep["in_mdt"][s_i, c_i]),
            jnp.asarray(sweep["in_gc"][s_i, c_i][:, :nchemg]),
            jnp.zeros((nbox, nbud + 1)),
            dtz=float(sweep["dtz"]),
            ifuchs=ifuchs,
            idcmfp=idcmfp,
            icondiam=icondiam,
            **({"fidelity": fidelity} if fidelity is not None else {}),
        ),
        (ncp, nchemg, nbud),
    )


def _expected(sweep, s_i, c_i, k_i, widths):
    ncp, nchemg, nbud = widths
    return {
        "md": sweep["md"][s_i, c_i, k_i][:, :, :ncp],
        "mdt": sweep["mdt"][s_i, c_i, k_i],
        "gc": sweep["gc"][s_i, c_i, k_i][:, :nchemg],
        "bud": sweep["bud"][s_i, c_i, k_i][:, : nbud + 1],
        "delgc": sweep["delgc"][s_i, c_i, k_i][:, :nchemg],
        "ageterm1": sweep["ageterm1"][s_i, c_i, k_i][:, :, :nchemg],
        "s_cond_s": sweep["s_cond_s"][s_i, c_i, k_i],
    }


FIELDS = ("md", "mdt", "gc", "bud", "delgc", "ageterm1", "s_cond_s")


def test_the_port_matches_the_compiled_routine(sweep):
    """All 56 configurations, all seven outputs. Exact where no `exp` reaches
    the answer, within a measured 5-ulp window where one does, and the affected
    counts pinned per field so the allowance cannot grow."""
    counts = dict.fromkeys(FIELDS, 0)
    worst = 0.0
    for s_i, setup, c_i, combo, k_i, config in _cases(sweep):
        out, widths = _run(sweep, s_i, setup, c_i, combo, config)
        want = _expected(sweep, s_i, c_i, k_i, widths)
        for name, got in zip(FIELDS, out):
            got = np.asarray(got)
            off = got != want[name]
            counts[name] += int(off.sum())
            if off.any():
                nz = off & (want[name] != 0)
                if nz.any():
                    ulp = np.abs(got[nz] - want[name][nz]) / np.spacing(np.abs(want[name][nz]))
                    worst = max(worst, float(ulp.max()))
    for name in EXACT_FIELDS:
        assert counts[name] == 0, f"{name} is not exact: {counts[name]} elements differ"
    assert {k: v for k, v in counts.items() if v} == EXP_AFFECTED
    assert worst <= EXP_ULP, f"worst {worst:.1f} ulp, measured 5"


def test_only_one_configuration_pair_differs_and_it_is_the_one_with_a_live_exponent(sweep):
    """`icondiam = 1` sets every `aa_modes` entry to zero, so
    `y2 = EXP(0.5*aa*aa*LOG(sigma)**2)` is `EXP(0)` -- exactly 1.0 on both
    sides. The gap needs a non-zero exponent AND a mode where XLA and libm
    disagree, and modes 6 and 7 only enter the loop once `topmode` moves."""
    differing = set()
    for s_i, setup, c_i, combo, k_i, config in _cases(sweep):
        out, widths = _run(sweep, s_i, setup, c_i, combo, config)
        want = _expected(sweep, s_i, c_i, k_i, widths)
        if any(np.any(np.asarray(g) != want[n]) for n, g in zip(FIELDS, out)):
            differing.add((setup, combo, config))
    assert differing == {(8, "dust_ageing", (1, 1, 2)), (8, "dust_ageing", (2, 2, 2))}
    assert cn.AA_MODES[1] == (0.0,) * 8


def test_the_whole_gap_is_the_exponential(sweep, monkeypatch):
    """The decisive experiment, a fourth time."""

    class _Libm:
        def __getattr__(self, name):
            return getattr(jnp, name)

        @staticmethod
        def exp(x):
            return jnp.asarray(np.exp(np.asarray(x)))

    monkeypatch.setattr(cn, "jnp", _Libm())
    monkeypatch.setattr(cc_mod, "jnp", _Libm())
    for s_i, setup, c_i, combo, k_i, config in _cases(sweep):
        if setup != 8 or combo != "dust_ageing":
            continue
        out, widths = _run(sweep, s_i, setup, c_i, combo, config)
        want = _expected(sweep, s_i, c_i, k_i, widths)
        for name, got in zip(FIELDS, out):
            np.testing.assert_array_equal(np.asarray(got), want[name], err_msg=f"{name} {config}")


# --- the generated write-site table ---------------------------------------


def test_the_generated_write_sites_are_not_stale(capsys):
    assert extractor.main(["--check"]) == 0
    assert "up to date" in capsys.readouterr().out


def test_the_check_mode_rejects_a_doctored_file(tmp_path, monkeypatch):
    doctored = tmp_path / "_conden_literals.py"
    doctored.write_text("WRITE_SITES = ()\n", encoding="utf-8")
    monkeypatch.setattr(extractor, "TARGET", doctored)
    assert extractor.main(["--check"]) == 1


def test_the_table_has_thirty_sites_and_one_duplicate():
    assert len(WRITE_SITES) == 30
    seen = [(r[5], r[3]) for r in WRITE_SITES]
    assert len({*seen}) == 29
    assert DUPLICATED == ("nmascondocaccins",)


def test_the_ageterm_ordinal_is_the_insoluble_mode_not_the_name_the_source_writes():
    """The source writes `ageterm1(:,mode_ait_sol,jv)` for an
    accumulation-INSOLUBLE transfer. The stored ordinal must be `nc_mode - 4`,
    and every soluble site must carry no ordinal at all."""
    for imode, _icp, _sec, kind, nc_mode, name, ageterm in WRITE_SITES:
        del imode, name
        if kind == "soluble":
            assert ageterm == 0
        else:
            assert ageterm == nc_mode - 4, (kind, nc_mode, ageterm)
            assert 1 <= ageterm <= 4


def test_every_insoluble_site_takes_its_delta_from_the_insoluble_mode():
    """`deltami = delgc_cond*nc(:,mode_XXX_insol)/sumnc`. A site that took
    `nc(:,imode)` would grow the insoluble mode at the soluble mode's rate."""
    for imode, _icp, _sec, kind, nc_mode, _name, _ag in WRITE_SITES:
        if kind == "soluble":
            assert nc_mode == imode
        else:
            assert nc_mode >= 5, (imode, kind, nc_mode)


def test_every_condensable_pair_has_a_budget_slot():
    """Issue #30, re-derived from the ported tables. Without a slot the port
    skips the site -- as the Fortran does -- and the mass is lost with the
    diagnostic."""
    short = ("nucsol", "aitsol", "accsol", "corsol", "aitins", "accins", "corins", "supins")
    cps = {0: "su", 1: "bc", 2: "oc", 3: "cl", 4: "du", 5: "so"}
    checked = 0
    for setup in bi.supported_setups():
        m, g, t = bi.build(setup), gi.build(setup), mm.build(setup)
        active = [i for i in range(8) if t.mode[i]]
        condensable = sorted(
            {
                cps[int(g.condensable_choice[i])]
                for i in range(len(g.condensable))
                if g.condensable[i]
            }
        )
        for i in active:
            for cp in condensable:
                name = f"nmascond{cp}{short[i]}"
                assert name in bi.BUDGET_NAMES
                assert m.is_carried(name), f"setup {setup}: {name} uncarried"
                checked += 1
    assert checked == 54


# --- the two fidelity flags -----------------------------------------------


def test_the_double_count_flag_changes_the_answer_at_both_settings(sweep):
    """Issue #29. Both arms against real data, and the difference confined to
    the one budget field -- `ageterm1` is an assignment and `md`/`mdt` take
    `deltams`, so neither may move."""
    target = None
    for s_i, setup, c_i, combo, k_i, config in _cases(sweep):
        if (setup, combo, config) == (8, "dust_ageing", (1, 1, 1)):
            target = (s_i, setup, c_i, combo, k_i, config)
    assert target is not None
    s_i, setup, c_i, combo, k_i, config = target

    on, widths = _run(
        sweep,
        s_i,
        setup,
        c_i,
        combo,
        config,
        fidelity=FidelityConfig(conden_ocaccins_double_count=True),
    )
    off, _ = _run(
        sweep,
        s_i,
        setup,
        c_i,
        combo,
        config,
        fidelity=FidelityConfig(conden_ocaccins_double_count=False),
    )

    slot = bi.build(setup).slot("nmascondocaccins")
    assert slot != bi.NOT_CARRIED, "setup 8 does not carry the field; the test proves nothing"
    bud_on, bud_off = np.asarray(on[3]), np.asarray(off[3])
    assert not np.array_equal(bud_on, bud_off), "the flag changed nothing"
    assert np.allclose(bud_on[:, slot], 2.0 * bud_off[:, slot], rtol=1e-12)
    # And nothing else may move.
    for i, name in enumerate(FIELDS):
        if name == "bud":
            continue
        np.testing.assert_array_equal(np.asarray(on[i]), np.asarray(off[i]), err_msg=name)
    # The default arm is the one the reference agrees with.
    np.testing.assert_array_equal(bud_on, _expected(sweep, s_i, c_i, k_i, widths)["bud"])


def test_up10_is_results_changing_after_all_on_constructed_inputs(sweep):
    """UP-10, and the first both-settings evidence anyone has had for it.

    `docs/fidelity.md` recorded that no both-settings test was possible: the
    line the defect reaches is gated by `topmode > mode_ait_insol`, and forcing
    `l_dust_mp_ageing` on setup 8 still left the mask false, because
    `init_state` puts `nd(mode_acc_insol)` at exactly `1e-14` and the test is
    strictly greater. That is a statement about the **trajectory's initial
    state**, and it is why this needed a constructed fixture rather than a
    longer run.

    The leaf grid straddles the thresholds deliberately, and with it the two
    settings differ on 4 of the 56 configurations -- all four setup 8 /
    `dust_ageing` ones -- moving 4 elements of `bud_aer_mas` and 4 of
    `ageterm1`.

    **`ageterm1` is not a diagnostic.** It is the mass `ukca_ageing` transfers
    from insoluble to soluble, so UP-10 is results-changing rather than
    reporting-only once ageing is ported. `md`/`mdt` do not move here because
    the insoluble gain is deliberately left to that routine -- the commented-out
    block at `:769-778` says so.
    """
    differing = {}
    for s_i, setup, c_i, combo, _k_i, config in _cases(sweep):
        on, _ = _run(
            sweep,
            s_i,
            setup,
            c_i,
            combo,
            config,
            fidelity=FidelityConfig(conden_insol_num_eps_by_sol_mode=True),
        )
        off, _ = _run(
            sweep,
            s_i,
            setup,
            c_i,
            combo,
            config,
            fidelity=FidelityConfig(conden_insol_num_eps_by_sol_mode=False),
        )
        moved = {
            name: int((np.asarray(a) != np.asarray(b)).sum())
            for name, a, b in zip(FIELDS, on, off)
            if np.any(np.asarray(a) != np.asarray(b))
        }
        if moved:
            differing[(setup, combo, config)] = moved

    assert set(differing) == {
        (8, "dust_ageing", (1, 1, 1)),
        (8, "dust_ageing", (1, 1, 2)),
        (8, "dust_ageing", (2, 2, 1)),
        (8, "dust_ageing", (2, 2, 2)),
    }
    assert all(m == {"bud": 4, "ageterm1": 4} for m in differing.values()), differing
    # md and mdt must NOT move: the insoluble gain is left to ukca_ageing.
    assert all("md" not in m and "mdt" not in m for m in differing.values())
    # And the default arm is the one the reference agrees with.
    for s_i, setup, c_i, combo, k_i, config in _cases(sweep):
        if (setup, combo, config) != (8, "dust_ageing", (1, 1, 1)):
            continue
        out, widths = _run(
            sweep,
            s_i,
            setup,
            c_i,
            combo,
            config,
            fidelity=FidelityConfig(conden_insol_num_eps_by_sol_mode=True),
        )
        want = _expected(sweep, s_i, c_i, k_i, widths)
        np.testing.assert_array_equal(np.asarray(out[5]), want["ageterm1"])


def test_the_num_eps_flag_is_actually_consumed():
    """Belt and braces on the registry's "no unread flag" rule: the flag must
    appear as code, not only in the prose that explains it."""
    source = (REPO / "src" / "glomap_jax" / "physics" / "conden.py").read_text(encoding="utf-8")
    assert "fidelity.conden_insol_num_eps_by_sol_mode" in source


# --- invariants -----------------------------------------------------------


def test_up4_cannot_fire(sweep):
    """`delgc_cond = gc*(1 - EXP(-sumnc*dtz))` and the exponential is
    non-negative, so `delgc_cond > gc` is unreachable. Asserted over the whole
    archive, which is UP-4's recorded disposition."""
    for s_i, _setup, c_i, _combo, k_i, _config in _cases(sweep):
        nchemg = int(sweep["nchemg"][s_i, c_i])
        assert np.all(
            sweep["delgc"][s_i, c_i, k_i][:, :nchemg] <= sweep["in_gc"][s_i, c_i][:, :nchemg]
        )


def test_an_unsupported_icondiam_raises(sweep):
    s_i, setup, c_i, combo, _k_i, _config = next(iter(_cases(sweep)))
    with pytest.raises(ValueError, match="icondiam"):
        _run(sweep, s_i, setup, c_i, combo, (1, 1, 3))


def test_the_hole_at_budget_slot_zero_is_never_written(sweep):
    for s_i, setup, c_i, combo, _k_i, config in _cases(sweep):
        out, _ = _run(sweep, s_i, setup, c_i, combo, config)
        assert np.all(np.asarray(out[3])[:, 0] == 0.0), (setup, combo, config)
