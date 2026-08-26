#!/usr/bin/env python3
"""Sweep `ukca_conden` over all seven supported setups (task 53).

    python validation/capture_conden_leaf.py --dry-run
    python validation/capture_conden_leaf.py     # writes tests/goldens/

`ukca_conden` condenses each condensable vapour onto the pre-existing aerosol.
It is the widest routine ported so far: four `INTENT(IN OUT)` arrays, three
`INTENT(OUT)`, a loop over gases containing two loops over modes, and 30
budget write sites.

Everything the sweep is built to settle
---------------------------------------

**`bud_aer_mas` is `(nbox, 0:nbudaer)`.** A zero lower bound, with slot 0 the
hole every uncarried index points at. The driver remaps it to a 1-based column
and asserts the extent, because `+1` stops being the right offset the moment
`nbudaer` changes shape and every budget field would shift by one silently.

**A budget index gates the physics.** `deltams` and `deltami` are zeroed at
`:363-364` and assigned *only inside* `IF (nmascond... > 0)`, and `md`/`mdt`
are then updated from `deltams` at `:762-767`. So an uncarried slot does not
merely lose a diagnostic, it loses the mass. Measured across all seven setups,
every (active mode, condensable component) pair has its slot carried -- 54 of
54 -- so the gating is inert today and the port reproduces it anyway. Issue #30.

**`:576-602` is the same block twice**, the only one of the 30 write sites that
is. `bud_aer_mas(:,nmascondocaccins)` is an accumulation, so it is doubled;
`ageterm1` is an assignment, so it is not; `md`/`mdt` use `deltams`, so they
are not either. Diagnostic-only, and reachable only with `l_dust_mp_ageing` on
setup 8, which is what `topmode > mode_ait_insol` turns on. Issue #29, and the
capture sweeps that switch so the golden records both.

**`topmode` bounds the first mode loop.** `:299` runs `DO imode=1,topmode`, and
`topmode` is `mode_ait_insol` (5) unless `l_dust_mp_ageing` is on
(`ukca_mode_setup.F90:418-422`). So by default `nc` is never assigned for modes
6 and 7 -- and the second loop reads `nc(:,mode_acc_insol)` only inside blocks
that carry the same `topmode > mode_ait_insol` guard. The two agree, which is
worth recording because if they ever stop agreeing the routine reads
uninitialised memory.

**UP-4 is asserted, not reached.** `:353-355` computes
`delgc_cond = delgc_cond/gc` where `= gc` was intended, guarded by
`delgc_cond > gc`. But `delgc_cond = gc*(1 - EXP(-sumnc*dtz))` and the
exponential is non-negative, so `delgc_cond <= gc` always. The capture asserts
that inequality on every row of every configuration rather than trying to
construct a counter-example -- which is what `docs/UPSTREAM_DEFECTS.md` says
UP-4's disposition is.

**`mask4i` is unreachable.** It needs `mode_sup_insol` active, and slot 8 needs
setup 12 or 13, which the box model does not implement. Asserted zero rather
than assumed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_modes import render_namelist
from leaf_common import CHILD_PREAMBLE, F2PY_DIR, NAMELISTS, REPO, check_varied, run_child

DEFAULT_OUT = REPO / "tests" / "goldens"
ARCHIVE = "conden.f64.leaf.npz"
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_conden.F90"
NAMELIST_TEXT = (NAMELISTS / "boundary_layer.nml").read_text(encoding="utf-8")

SETUPS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8)
NMODES = 8
NMODES_INS = 4
DTZ = 60.0

#: (ifuchs, idcmfp, icondiam). icondiam selects the condensation diameter:
#: 1 takes the geometric-mean number radius, 2 the Lehtinen et al (2003)
#: per-mode moment. Every namelist runs (1, 1, 1).
CONFIGS: tuple[tuple[int, int, int], ...] = ((1, 1, 1), (1, 1, 2), (2, 2, 1), (2, 2, 2))

#: `l_dust_mp_ageing` moves `topmode` from 5 to 8, which is the only way to
#: reach the duplicated block of issue #29 and the insoluble blocks of UP-10.
DUST_AGEING: tuple[bool, ...] = (False, True)

BLOCKS = ("dense", "sparse", "gas_guard", "empty_modes")


def _block_code(name: str) -> int:
    return BLOCKS.index(name)


def build_grid(ncp: int, nchemg: int) -> dict[str, np.ndarray]:
    """Box rows wide enough to move every mask both ways.

    `nd` spans `num_eps` (1e-8 for the soluble modes, 1e-14 for two of the
    insoluble ones) from far below to far above, so `mask3` and `mask3i` are
    each false somewhere and true somewhere. `gc` spans `conc_eps = 1e-8` for
    the same reason on `mask1`.

    The composition is deliberately crude -- equal mass in every component --
    because `ukca_conden` never reads `md` except to add to it, and a realistic
    composition would only make the golden harder to read.
    """
    rows: list[dict] = []

    def add(*, nd_scale, gc_scale, t=273.15, pmid=1.0e5, blk):
        # Distinct per mode so a slot mix-up cannot hide, spanning num_eps.
        nd = np.array([1.0e3, 5.0e2, 1.0e2, 1.0e0, 3.0e2, 5.0e1, 1.0e1, 1.0], dtype=np.float64)
        rows.append(
            {
                "nd": nd * nd_scale,
                "wetdp": np.array(
                    [8.0e-9, 6.0e-8, 4.0e-7, 4.0e-6, 1.0e-7, 6.0e-7, 6.0e-6, 2.0e-5],
                    dtype=np.float64,
                ),
                "gc_scale": gc_scale,
                "t": t,
                "pmid": pmid,
                "blk": blk,
            }
        )

    # Dense: everything above threshold, several temperatures and pressures.
    for t in (240.0, 273.15, 292.0):
        for pmid in (2.0e4, 1.0e5):
            add(nd_scale=1.0e4, gc_scale=1.0e8, t=t, pmid=pmid, blk="dense")
    # Sparse: number concentrations straddling num_eps.
    for nd_scale in (1.0e-12, 1.0e-10, 1.0e-6, 1.0e-2, 1.0e2):
        add(nd_scale=nd_scale, gc_scale=1.0e8, blk="sparse")
    # The gas guard, conc_eps = 1e-8, from both sides.
    for gc_scale in (0.0, 1.0e-10, 1.0e-9, 1.0e-8, 1.0e-7, 1.0):
        add(nd_scale=1.0e4, gc_scale=gc_scale, blk="gas_guard")
    # Every mode empty: sumnc is zero and mask2 must reject the row before
    # anything divides by it.
    add(nd_scale=0.0, gc_scale=1.0e8, blk="empty_modes")

    n = len(rows)
    nd = np.stack([r["nd"] for r in rows])
    wetdp = np.stack([r["wetdp"] for r in rows])
    md = np.full((n, NMODES, ncp), 1.0e-20, dtype=np.float64)
    mdt = md.sum(axis=2)
    gc = np.zeros((n, nchemg), dtype=np.float64)
    # Every gas slot gets the row's scale; `condensable` decides which matter,
    # and sweeping them all is how the capture shows the others are inert.
    for i, r in enumerate(rows):
        gc[i, :] = r["gc_scale"]

    t = np.array([r["t"] for r in rows], dtype=np.float64)
    pmid = np.array([r["pmid"] for r in rows], dtype=np.float64)
    return {
        "nd": nd,
        "wetdp": wetdp,
        "md": md,
        "mdt": mdt,
        "gc": gc,
        "t": t,
        "pmid": pmid,
        "tsqrt": np.sqrt(t),
        # Physically consistent with (t, pmid) but built by plain arithmetic.
        "rhoa": pmid / (287.05 * t),
        "airdm3": pmid / (1.3804e-23 * t),
        "block": np.array([_block_code(r["blk"]) for r in rows], dtype=np.int32),
    }


CHILD_BODY = (
    "\nimport tempfile\n"
    f"sys.path.insert(0, {str(REPO / 'validation')!r})\n"
    "from capture_conden_leaf import build_grid, CONFIGS, DTZ, NMODES, NMODES_INS\n"
    "from leaf_common import bind_call\n"
    "\n"
    "call = bind_call(g)\n"
    "_sz = call('sizes', g.wrap_sizes)\n"
    "_nbox, _nm, _ncp, _nchemg, _nadvg, _nbudaer = [int(v) for v in _sz[:6]]\n"
    "grid = build_grid(_ncp, _nchemg)\n"
    "nb = grid['nd'].shape[0]\n"
    "out = {k: [] for k in ('md','mdt','gc','bud','delgc','ageterm1','s_cond_s')}\n"
    "for _f, _d, _c in CONFIGS:\n"
    "    _md, _mdt, _gc, _bud, _dg, _ag, _sc, _e = call(\n"
    "        'conden %d/%d/%d' % (_f, _d, _c), g.leaf_conden,\n"
    "        _nbudaer + 1, NMODES_INS, _f, _d, _c, DTZ,\n"
    "        grid['nd'], grid['tsqrt'], grid['rhoa'], grid['airdm3'],\n"
    "        grid['wetdp'], grid['pmid'], grid['t'],\n"
    "        grid['md'], grid['mdt'], grid['gc'])\n"
    "    out['md'].append(np.asarray(_md)); out['mdt'].append(np.asarray(_mdt))\n"
    "    out['gc'].append(np.asarray(_gc)); out['bud'].append(np.asarray(_bud))\n"
    "    out['delgc'].append(np.asarray(_dg)); out['ageterm1'].append(np.asarray(_ag))\n"
    "    out['s_cond_s'].append(np.asarray(_sc))\n"
    "\n"
    "_mode, _e1 = call('mode', g.wrap_mode_int, 'mode', NMODES)\n"
    "_topmode, _e2 = call('topmode', g.wrap_topmode)\n"
    "_tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)\n"
    "_tmp.close()\n"
    "np.savez(_tmp.name, mode=np.asarray(_mode), topmode=np.int64(_topmode),\n"
    "         ncp=np.int64(_ncp), nchemg=np.int64(_nchemg), nbudaer=np.int64(_nbudaer),\n"
    "         **{k: np.stack(v) for k, v in out.items()})\n"
    "print('@@RESULT@@' + json.dumps({'npz_path': _tmp.name, 'setup': _setup}))\n"
)

_CHILD = CHILD_PREAMBLE.format(f2py=str(F2PY_DIR)) + CHILD_BODY


# --- verification -----------------------------------------------------------

#: `render_namelist` combinations. `dust_ageing` is the only way to move
#: `topmode` off `mode_ait_insol`, and therefore the only way to reach the
#: insoluble condensation blocks at all.
COMBOS: tuple[str, ...] = ("default", "dust_ageing")

#: Collisions that are findings about the Fortran rather than capture bugs.
#: Each is declared with a reason, and an UNdeclared one fails the capture --
#: as does a declared one that stops happening.
#:
#: 1. `l_dust_mp_ageing` reaches `ukca_conden` only through `topmode`, which it
#:    moves from `mode_ait_insol` (5) to `nmodes` (8). That extends the first
#:    mode loop at `:299` and opens the `topmode > mode_ait_insol` guards in the
#:    second. In setups 1-5 no mode above 5 is active and in setup 6 no gas is
#:    condensable, so the switch is inert. **Setup 8 is the only supported setup
#:    where it does anything**, and its two combos are required to differ.
#:
#: 2. Setups 2 and 8 are the same program when `topmode = 5`. Measured: their
#:    `condensable`, `condensable_choice`, `mm_gas`, `dimen`, `sigmag` and
#:    `num_eps` tables are identical, all ten of setup 2's condensation budget
#:    names occupy the same slot index in setup 8, and their `mode` vectors
#:    differ only in modes 6 and 7 -- which `topmode = 5` excludes from the
#:    first loop and whose second-loop blocks carry the same guard.
EXPECTED_COLLISIONS: list[tuple[str, str]] = [
    *((f"setup_{s}_default", f"setup_{s}_dust_ageing") for s in (1, 2, 3, 4, 5, 6)),
    ("setup_2_default", "setup_8_default"),
    ("setup_2_dust_ageing", "setup_8_default"),
]


def verify_source_structure() -> dict[str, int]:
    """Re-read the structure this capture's claims rest on."""
    import re

    text = SOURCE.read_text(encoding="utf-8")
    body = re.sub(r"&\s*\n\s*", "", text)
    squeezed = text.replace(" ", "")
    # Continuations are joined in `body`; statements that span lines have to
    # be matched against that, not against the raw text where `&` survives.
    squeezed_body = body.replace(" ", "")

    sites = re.findall(r"bud_aer_mas\(:,(\w+)\)\s*=\s*bud_aer_mas\(:,\1\)\s*\+\s*(delta\w+)", body)
    if len(sites) != 30:
        raise SystemExit(f"{len(sites)} bud_aer_mas write sites, expected 30")
    if len(set(sites)) != 29:
        raise SystemExit(
            f"{len(set(sites))} distinct (name, delta) pairs, expected 29 -- issue #29's "
            "duplicated block may have been fixed or a second one introduced"
        )
    duplicated = [k for k in set(sites) if sites.count(k) > 1]
    if duplicated != [("nmascondocaccins", "deltami")]:
        raise SystemExit(f"the duplicated write site is now {duplicated}, not nmascondocaccins")
    # Every `*ins` name takes deltami and every soluble name deltams.
    for name, delta in sites:
        if (delta == "deltami") != name.endswith("ins"):
            raise SystemExit(f"{name} accumulates {delta}; the naming rule no longer holds")

    required = {
        "deltams zeroed": "deltams(:)=0.0" in squeezed,
        "deltami zeroed": "deltami(:)=0.0" in squeezed,
        "first loop bound": "DOimode=1,topmode" in squeezed,
        "second loop bound": "DOimode=mode_nuc_sol,mode_cor_sol" in squeezed,
        "up4 site": "delgc_cond(:,jv)=delgc_cond(:,jv)/gc(:,jv)" in squeezed,
        "md update": "md(:,imode,icp)=(md(:,imode,icp)*nd(:,imode)+deltams(:))/nd(:,imode)"
        in squeezed_body,
        "bud lower bound": "bud_aer_mas(nbox,0:nbudaer)" in squeezed_body,
    }
    missing = sorted(k for k, ok in required.items() if not ok)
    if missing:
        raise SystemExit("the vendored routine no longer contains: " + ", ".join(missing))
    return {"write_sites": len(sites), "distinct": len(set(sites))}


def verify_budget_gating() -> dict[str, int]:
    """Every (active mode, condensable component) pair must have a slot.

    The gating at `:363-364` means an uncarried slot loses the mass, not just
    the diagnostic. This re-derives the alignment from the three ported tables
    rather than trusting it -- issue #30.
    """
    sys.path.insert(0, str(REPO / "src"))
    from glomap_jax.physics import budget_indices as bi
    from glomap_jax.physics import gas_indices as gi
    from glomap_jax.physics import modes as md

    short = ("nucsol", "aitsol", "accsol", "corsol", "aitins", "accins", "corins", "supins")
    cps = {0: "su", 1: "bc", 2: "oc", 3: "cl", 4: "du", 5: "so"}
    checked = 0
    for setup in SETUPS:
        m, g, t = bi.build(setup), gi.build(setup), md.build(setup)
        active = [i for i in range(NMODES) if t.mode[i]]
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
                if name not in bi.BUDGET_NAMES:
                    raise SystemExit(f"setup {setup}: no budget name {name}")
                if not m.is_carried(name):
                    raise SystemExit(
                        f"setup {setup}: {name} is not carried, so condensation of {cp} onto "
                        f"mode {i + 1} is silently dropped -- issue #30's alignment has broken"
                    )
                checked += 1
    return {"pairs": checked}


#: Padding widths. `ncp`, `nchemg` and `nbudaer` all differ between setups --
#: `nbudaer` takes seven distinct values across the seven (phase C) -- so the
#: per-setup arrays cannot be stacked as they come. Everything is padded to a
#: fixed width with zeros and the true widths are stored beside it, which is the
#: convention `budget_indices.PADDED_WIDTH` already uses.
NCP_MAX = 8
NCHEMG_MAX = 16
NBUD_MAX = 139  # nbudaer + 1, the largest across the supported setups


def _pad(array: np.ndarray, widths: dict[int, int]) -> np.ndarray:
    """Zero-pad the named trailing axes to their fixed width."""
    pad = [(0, 0)] * array.ndim
    for axis, width in widths.items():
        if array.shape[axis] > width:
            raise SystemExit(f"axis {axis} of shape {array.shape} exceeds the padded width {width}")
        pad[axis] = (0, width - array.shape[axis])
    return np.pad(array, pad, mode="constant")


#: Which trailing axes each captured array has to be padded on.
PAD_AXES: dict[str, dict[int, int]] = {
    "md": {2: NCP_MAX},
    "gc": {2: NCHEMG_MAX},
    "delgc": {2: NCHEMG_MAX},
    "ageterm1": {3: NCHEMG_MAX},
    "bud": {2: NBUD_MAX},
}


def verify_outputs(data: dict, grid: dict) -> dict:
    report: dict[str, int] = {"mask4i_hits": 0, "insoluble_ageterm": 0, "budget_written": 0}

    for key, arrays in data.items():
        if key in ("mode", "topmode", "ncp", "nchemg", "nbudaer"):
            continue
        if not np.all(np.isfinite(arrays)):
            raise SystemExit(f"{key} is not finite")

    for s_i, setup in enumerate(SETUPS):
        for c_i, combo in enumerate(COMBOS):
            topmode = int(data["topmode"][s_i, c_i])
            expected_topmode = 8 if combo == "dust_ageing" else 5
            if topmode != expected_topmode:
                raise SystemExit(
                    f"setup {setup}/{combo}: topmode is {topmode}, expected {expected_topmode}"
                )
            for k_i, config in enumerate(CONFIGS):
                label = f"setup {setup}/{combo}/{config}"
                gc_out = data["gc"][s_i, c_i, k_i]
                delgc = data["delgc"][s_i, c_i, k_i]
                gc_in = data["in_gc"][s_i, c_i]

                # UP-4's invariant. delgc_cond = gc*(1-EXP(-sumnc*dtz)) and the
                # exponential is non-negative, so the `delgc_cond > gc` guard
                # at :353 can never be true.
                if np.any(delgc > gc_in + 0.0):
                    raise SystemExit(
                        f"{label}: delgc_cond exceeds gc on "
                        f"{int((delgc > gc_in).sum())} entries -- UP-4's guard is reachable "
                        "after all and its invariant-test disposition is wrong"
                    )
                if np.any(gc_out > gc_in):
                    raise SystemExit(f"{label}: condensation increased a gas concentration")
                if np.any(gc_out < 0.0):
                    raise SystemExit(f"{label}: a gas concentration went negative")
                # mdt may only grow, and md with it.
                if np.any(data["mdt"][s_i, c_i, k_i] < data["in_mdt"][s_i, c_i]):
                    raise SystemExit(f"{label}: mdt decreased")
                if np.any(data["s_cond_s"][s_i, c_i, k_i] < 0.0):
                    raise SystemExit(f"{label}: s_cond_s went negative")

                report["budget_written"] += int(np.any(data["bud"][s_i, c_i, k_i] != 0.0))
                report["insoluble_ageterm"] += int(np.any(data["ageterm1"][s_i, c_i, k_i] != 0.0))

    # Slot 0 of bud_aer_mas is the hole and must stay empty.
    if np.any(data["bud"][..., 0] != 0.0):
        raise SystemExit("budget slot 0 -- the NOT_CARRIED hole -- was written")

    if report["budget_written"] == 0:
        raise SystemExit("no configuration wrote any budget field; the grid misses condensation")
    if report["insoluble_ageterm"] == 0:
        raise SystemExit(
            "ageterm1 is zero everywhere, so no insoluble condensation ran and the "
            "whole mask3i half of the routine is unvalidated"
        )
    return report


def capture(out_dir: Path, quiet: bool = False) -> Path:
    verify_source_structure()
    verify_budget_gating()

    per_setup: dict[str, list] = {}
    grid_ref = True
    for setup in SETUPS:
        per_combo: dict[str, list] = {}
        for combo in COMBOS:
            record = run_child(
                CHILD_BODY,
                namelist_text=render_namelist(NAMELIST_TEXT, setup, combo),
                setup=setup,
                label=f"conden setup {setup} / {combo}",
            )
            try:
                with np.load(record["npz_path"]) as d:
                    for k in d.files:
                        value = d[k]
                        if k in PAD_AXES:
                            value = _pad(value, PAD_AXES[k])
                        per_combo.setdefault(k, []).append(value)
            finally:
                Path(record["npz_path"]).unlink(missing_ok=True)
            # The input grid's width follows this setup's `ncp` and `nchemg`,
            # so the reference copy has to be built per configuration and
            # padded the same way the outputs are. Comparing against a grid
            # built at the widest would make `mdt` -- a sum over `ncp` columns
            # -- larger than anything the routine was given.
            local = build_grid(int(per_combo["ncp"][-1]), int(per_combo["nchemg"][-1]))
            per_combo.setdefault("in_md", []).append(_pad(local["md"], {2: NCP_MAX}))
            per_combo.setdefault("in_mdt", []).append(local["mdt"])
            per_combo.setdefault("in_gc", []).append(_pad(local["gc"], {1: NCHEMG_MAX}))
        for k, v in per_combo.items():
            per_setup.setdefault(k, []).append(np.stack(v))
        if not quiet:
            print(f"  i_mode_setup = {setup}: {len(COMBOS)} combos x {len(CONFIGS)} configs")

    data = {k: np.stack(v) for k, v in per_setup.items()}
    # Setup-independent inputs, stored once. The width-dependent ones are in
    # `data` already, one per configuration.
    grid = build_grid(NCP_MAX, NCHEMG_MAX)
    del grid_ref

    # Fingerprint on four outputs, not on `gc` alone. `gc` is a weak
    # discriminator here: it depends on the condensable gases' molar masses and
    # the mode geometry, both of which several setups share, so setups 2 and 4
    # produce identical gas concentrations from this grid while writing
    # completely different budget fields. Phase C recorded the same failure --
    # an anti-collapse guard keyed on the only two scalars that happened to
    # differ -- and this is the same mistake with the fields swapped.
    check_varied(
        {
            f"setup_{s}_{c}": {
                "gc": data["gc"][i, j, 0].tolist(),
                "bud": data["bud"][i, j, 0].tolist(),
                "ageterm1": data["ageterm1"][i, j, 0].tolist(),
                "mdt": data["mdt"][i, j, 0].tolist(),
            }
            for i, s in enumerate(SETUPS)
            for j, c in enumerate(COMBOS)
        },
        expected_identical=EXPECTED_COLLISIONS,
        what="conden setups",
    )

    report = verify_outputs(data, grid)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ARCHIVE
    np.savez_compressed(
        path,
        setups=np.array(SETUPS, dtype=np.int64),
        combos=np.array(COMBOS, dtype="U16"),
        config=np.array(CONFIGS, dtype=np.int64),
        blocks=np.array(BLOCKS, dtype="U16"),
        dtz=np.float64(DTZ),
        **{f"in_{k}": v for k, v in grid.items() if k not in ("md", "mdt", "gc")},
        **data,
    )
    if not quiet:
        print(f"  wrote {path.name}: gc {data['gc'].shape}")
        print("  " + ", ".join(f"{k}={v}" for k, v in report.items()))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        sites = verify_source_structure()
        pairs = verify_budget_gating()
        grid = build_grid(6, 12)
        print(f"leaf conden sweep -> {args.out / ARCHIVE}")
        print(f"  rows        {grid['nd'].shape[0]:>6,}")
        for b in BLOCKS:
            print(f"    {b:<14}{int((grid['block'] == _block_code(b)).sum()):>6,}")
        print(f"  setups      {len(SETUPS):>6,}  x {len(COMBOS)} combos x {len(CONFIGS)} configs")
        print(f"  write sites {sites['write_sites']:>6,} ({sites['distinct']} distinct)")
        print(f"  budget pairs{pairs['pairs']:>6,} checked, all carried")
        return 0

    print(f"sweeping leaf_conden -> {args.out}")
    capture(args.out)
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
