#!/usr/bin/env python3
"""Sweep `ukca_coagwithnucl` over all seven supported setups (task 56).

    python validation/capture_coagwithnucl_leaf.py --dry-run
    python validation/capture_coagwithnucl_leaf.py    # writes tests/goldens/

`ukca_coagwithnucl` turns the pre-computed coagulation kernels into number and
mass transfers: it builds `A`, `B` and `C` per mode, calls
`ukca_solvecoagnucl_v` for the number, accumulates the per-component mass
transfers in `mtran`, redistributes them through `coag_mode`, and writes 178
budget diagnostics.

The `icp` loop is loop-carried, and this is the fixture that proves it
--------------------------------------------------------------------

`:544-575` is the loop CLAUDE.md names as one of the five that need a
sequential `lax.scan`:

```fortran
DO icp=1,ncp
  ...
  WHERE (mask1(:) .AND. (mdcpnew(:) < 0.0))
    nd(:,imode)=0.0
    mask1(:)=.FALSE. ! set false so not used for other icp values
  END WHERE
  mask3(:)=mask1(:) .AND. (mdcpnew(:) >= 0.0)
  WHERE (mask3(:))
    md(:,imode,icp)=mdcpnew(:)/nd(:,imode)
    mdt(:,imode)=mdt(:,imode)+md(:,imode,icp)
  END WHERE
END DO
```

`mask1` is **mutated inside the loop**, and the source comment says why: once
any component's new mass goes negative, the mode's number is zeroed and every
*later* component is skipped. A `vmap` over `icp` evaluates all components
against the original mask and gets a different answer -- which is what
CLAUDE.md means by "compute all deltas, then apply changes the answer", and
what the port's test has to demonstrate rather than assert.

**Reaching `mdcpnew < 0` is the whole difficulty.** Issue #13 records it as one
of the branches no trajectory fixture reaches. It needs the net transfer out of
a component to exceed the mass that was there, which the grid arranges directly
by giving a mode a large kernel, a small `md` for one component and a full
timestep.

`coag_mode` decides where the mass lands
----------------------------------------

`:534-536` scatters `mtran(:,icp,imode,jmode)` into
`mtrantoi(:,coag_mode(imode,jmode),icp)`, so several `(imode, jmode)` pairs
accumulate into the same destination. Phase C measured `coag_mode` symmetric on
all 64 entries and warned that a transposed transcription would be undetectable
in the table; this is the routine where the subscript order finally has
consequences, so the capture records `coag_mode` beside the outputs.

`intraoff` and `interoff` are swept
-----------------------------------

They gate the `A` and `B` terms at `:299` and `:313`. Both are namelist
variables, both default off in every shipped namelist, and switching either on
removes a whole term from the solver -- so all four combinations are captured.

`iextra_checks > 1` is refused, not captured: `:583` calls
`ukca_mode_check_mdt`, which `docs/unsupported.md` records as not ported
because it zeroes number concentration and changes mass budgets.
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
ARCHIVE = "coagwithnucl.f64.leaf.npz"
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_coagwithnucl.F90"
NAMELIST_TEXT = (NAMELISTS / "boundary_layer.nml").read_text(encoding="utf-8")

SETUPS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8)
COMBOS: tuple[str, ...] = ("default", "dust_ageing")
NMODES = 8
NMODES_SOL = 4
NMODES_INS = 4
DTZ = 300.0

#: `(intraoff, interoff)`. Both default 0 in every shipped namelist.
SWITCHES: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (0, 1), (1, 1))

NCP_MAX = 8
NCHEMG_MAX = 16
NBUD_MAX = 139

BLOCKS = ("ordinary", "negative_mass", "sparse", "nucleating")


def _block_code(name: str) -> int:
    return BLOCKS.index(name)


def _pad(array: np.ndarray, widths: dict[int, int]) -> np.ndarray:
    pad = [(0, 0)] * array.ndim
    for axis, width in widths.items():
        if array.shape[axis] > width:
            raise SystemExit(f"axis {axis} of {array.shape} exceeds {width}")
        pad[axis] = (0, width - array.shape[axis])
    return np.pad(array, pad, mode="constant")


#: The child stacks over the four switch pairs before saving, so every
#: captured array carries a leading switch axis and the width-bearing axis is
#: one further right than the routine's own signature suggests. Getting this
#: wrong is a `np.stack` failure rather than a silent shift, which is the only
#: reason it is safe to write down.
PAD_AXES: dict[str, dict[int, int]] = {
    "md": {3: NCP_MAX},  # (switch, box, mode, cp)
    "bud": {2: NBUD_MAX},  # (switch, box, slot)
    "ageterm2": {4: NCP_MAX},  # (switch, box, sol, ins, cp)
    "in_md": {2: NCP_MAX},  # (box, mode, cp) -- built here, no switch axis
    "in_delgc": {1: NCHEMG_MAX},  # (box, gas)
}


def build_grid(ncp: int, nchemg: int) -> dict[str, np.ndarray]:
    """Box rows, including ones built to drive a component's mass negative.

    The kernels are handed in directly rather than computed: `kii_arr` and
    `kij_arr` are `ukca_calc_coag_kernel`'s outputs and task 50 already pinned
    them, so constructing them here is what lets the `mdcpnew < 0` reset be
    reached at all.
    """
    rows: list[dict] = []

    def add(*, nd_scale, kii, kij, md_scale, delgc, blk):
        rows.append(
            {
                "nd": np.array([1.0e4, 5.0e3, 1.0e3, 1.0e2, 2.0e3, 5.0e2, 1.0e2, 1.0e1]) * nd_scale,
                "kii": kii,
                "kij": kij,
                "md_scale": md_scale,
                "delgc": delgc,
                "blk": blk,
            }
        )

    # Ordinary: kernels of the size ukca_calc_coag_kernel actually returns.
    for kii in (1.0e-9, 1.0e-8, 1.0e-7):
        for nd_scale in (1.0, 1.0e2):
            add(nd_scale=nd_scale, kii=kii, kij=kii * 0.5, md_scale=1.0, delgc=0.0, blk="ordinary")
    # Nucleating: delgc_nucl above conc_eps drives the C term of the nuc mode.
    for delgc in (1.0e-7, 1.0e-2, 1.0e3, 1.0e6):
        add(nd_scale=1.0, kii=1.0e-8, kij=5.0e-9, md_scale=1.0, delgc=delgc, blk="nucleating")
    # Sparse: number concentrations straddling num_eps, so mask1 is false
    # somewhere and the mmid/mfrac_0 reset at :377-383 runs.
    for nd_scale in (0.0, 1.0e-13, 1.0e-9, 1.0e-5):
        add(nd_scale=nd_scale, kii=1.0e-8, kij=5.0e-9, md_scale=1.0, delgc=0.0, blk="sparse")
    # Negative mass: a large inter-modal kernel with a tiny per-component mass,
    # so the transfer out exceeds what is there and mdcpnew goes negative --
    # the branch issue #13 says no trajectory reaches.
    for kij in (1.0e-4, 1.0e-3, 1.0e-2):
        for md_scale in (1.0e-8, 1.0e-4):
            add(
                nd_scale=1.0, kii=1.0e-9, kij=kij, md_scale=md_scale, delgc=0.0, blk="negative_mass"
            )

    n = len(rows)
    nd = np.stack([r["nd"] for r in rows])
    # Distinct per component so a slot mix-up cannot hide, scaled per row.
    md = np.zeros((n, NMODES, ncp), dtype=np.float64)
    for i, r in enumerate(rows):
        for c in range(ncp):
            md[i, :, c] = 1.0e-19 * (c + 1) * r["md_scale"]
    mdt = md.sum(axis=2)
    kii = np.stack([np.full(NMODES, r["kii"]) for r in rows])
    kij = np.zeros((n, NMODES, NMODES), dtype=np.float64)
    for i, r in enumerate(rows):
        for a in range(NMODES):
            for b in range(NMODES):
                if a != b:
                    kij[i, a, b] = r["kij"]
    delgc = np.zeros((n, nchemg), dtype=np.float64)
    for i, r in enumerate(rows):
        delgc[i, :] = r["delgc"]
    return {
        "nd": nd,
        "md": md,
        "mdt": mdt,
        "kii": kii,
        "kij": kij,
        "delgc": delgc,
        "block": np.array([_block_code(r["blk"]) for r in rows], dtype=np.int32),
    }


CHILD_BODY = (
    "\nimport tempfile\n"
    f"sys.path.insert(0, {str(REPO / 'validation')!r})\n"
    "from capture_coagwithnucl_leaf import (build_grid, SWITCHES, DTZ, NMODES,\n"
    "                                        NMODES_SOL, NMODES_INS)\n"
    "from leaf_common import bind_call\n"
    "\n"
    "call = bind_call(g)\n"
    "_sz = call('sizes', g.wrap_sizes)\n"
    "_nbox, _nm, _ncp, _nchemg, _nadvg, _nbudaer = [int(v) for v in _sz[:6]]\n"
    "grid = build_grid(_ncp, _nchemg)\n"
    "out = {k: [] for k in ('nd','md','mdt','bud','ageterm2')}\n"
    "for _intra, _inter in SWITCHES:\n"
    "    _nd, _md, _mdt, _bud, _ag, _e = call(\n"
    "        'coag %d/%d' % (_intra, _inter), g.leaf_coagwithnucl,\n"
    "        _nbudaer + 1, NMODES_SOL, NMODES_INS, _intra, _inter, 0, DTZ,\n"
    "        grid['nd'], grid['md'], grid['mdt'], grid['delgc'],\n"
    "        grid['kii'], grid['kij'])\n"
    "    out['nd'].append(np.asarray(_nd)); out['md'].append(np.asarray(_md))\n"
    "    out['mdt'].append(np.asarray(_mdt)); out['bud'].append(np.asarray(_bud))\n"
    "    out['ageterm2'].append(np.asarray(_ag))\n"
    "\n"
    "_mode, _e1 = call('mode', g.wrap_mode_int, 'mode', NMODES)\n"
    "_topmode, _e2 = call('topmode', g.wrap_topmode)\n"
    "_cm, _e3 = call('coag_mode', g.wrap_coag_mode, NMODES)\n"
    "_tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)\n"
    "_tmp.close()\n"
    "np.savez(_tmp.name, mode=np.asarray(_mode), topmode=np.int64(_topmode),\n"
    "         coag_mode=np.asarray(_cm), ncp=np.int64(_ncp), nchemg=np.int64(_nchemg),\n"
    "         nbudaer=np.int64(_nbudaer),\n"
    "         **{k: np.stack(v) for k, v in out.items()})\n"
    "print('@@RESULT@@' + json.dumps({'npz_path': _tmp.name, 'setup': _setup}))\n"
)

_CHILD = CHILD_PREAMBLE.format(f2py=str(F2PY_DIR)) + CHILD_BODY


# --- verification -----------------------------------------------------------


def verify_source_structure() -> dict[str, int]:
    """Re-read the loop this capture and the port are built around."""
    import re

    text = SOURCE.read_text(encoding="utf-8")
    body = re.sub(r"&\s*\n\s*", "", text)
    sq = re.sub(r"[ \t]+", "", body)
    required = {
        "icp loop mutates mask1": "mask1(:)=.FALSE." in sq,
        "the comment says why": "set false so not used for other icp values" in text,
        "mdcpnew guard": "WHERE(mask1(:).AND.(mdcpnew(:)<0.0))" in sq,
        "nd zeroed": "nd(:,imode)=0.0" in sq,
        "mask3 after": "mask3(:)=mask1(:).AND.(mdcpnew(:)>=0.0)" in sq,
        "coag_mode scatter": "mtrantoi(:,coag_mode(imode,jmode),icp)=" in sq,
        "intraoff gate": "IF(intraoff/=1)THEN" in sq,
        "interoff gate": "IF(interoff/=1)THEN" in sq,
        "xxx_eps": "mask4(:)=ABS(xxx(:))>xxx_eps" in sq,
        "check_mdt": "CALLukca_mode_check_mdt(" in sq,
    }
    missing = sorted(k for k, ok in required.items() if not ok)
    if missing:
        raise SystemExit("the vendored routine no longer contains: " + ", ".join(missing))
    sites = re.findall(r"bud_aer_mas\(:,(nmascoag\w+)\)", sq)
    return {"checked": len(required), "budget_sites": len(set(sites))}


def verify_outputs(data: dict, grid_of) -> dict:
    report = {"negative_mass_rows": 0, "budget_written": 0, "ageterm2_written": 0}
    n_setup, n_combo, n_switch = data["nd"].shape[:3]

    for s_i in range(n_setup):
        for c_i in range(n_combo):
            grid = grid_of(s_i, c_i)
            for k_i in range(n_switch):
                label = f"setup {s_i}/{c_i}/{k_i}"
                nd = data["nd"][s_i, c_i, k_i]
                md = data["md"][s_i, c_i, k_i]
                mdt = data["mdt"][s_i, c_i, k_i]
                for name, arr in (
                    ("nd", nd),
                    ("md", md),
                    ("mdt", mdt),
                    ("bud", data["bud"][s_i, c_i, k_i]),
                    ("ageterm2", data["ageterm2"][s_i, c_i, k_i]),
                ):
                    if not np.all(np.isfinite(arr)):
                        raise SystemExit(f"{label}: {name} is not finite")
                if np.any(nd < 0.0):
                    raise SystemExit(f"{label}: nd went negative")
                # `:566-569`: a component whose new mass is negative zeroes the
                # mode's number. Count the rows that took it.
                zeroed = (nd == 0.0) & (grid["nd"] > 0.0)
                report["negative_mass_rows"] += int(zeroed.sum())
                report["budget_written"] += int(np.any(data["bud"][s_i, c_i, k_i] != 0.0))
                report["ageterm2_written"] += int(np.any(data["ageterm2"][s_i, c_i, k_i] != 0.0))

    if np.any(data["bud"][..., 0] != 0.0):
        raise SystemExit("budget slot 0 -- the NOT_CARRIED hole -- was written")
    if report["negative_mass_rows"] == 0:
        raise SystemExit(
            "no row reaches the mdcpnew < 0 reset, which is the branch issue #13 "
            "names and the reason the icp loop must be sequential -- a port that "
            "vmapped it would pass every comparison in this golden"
        )
    if report["budget_written"] == 0:
        raise SystemExit("no configuration wrote a budget field")
    if report["ageterm2_written"] == 0:
        raise SystemExit("ageterm2 is zero everywhere; the insoluble half never ran")
    return report


def capture(out_dir: Path, quiet: bool = False) -> Path:
    verify_source_structure()

    per_setup: dict[str, list] = {}
    widths: dict[tuple[int, int], tuple[int, int]] = {}
    for s_i, setup in enumerate(SETUPS):
        per_combo: dict[str, list] = {}
        for c_i, combo in enumerate(COMBOS):
            record = run_child(
                CHILD_BODY,
                namelist_text=render_namelist(NAMELIST_TEXT, setup, combo),
                setup=setup,
                label=f"coagwithnucl setup {setup} / {combo}",
            )
            try:
                with np.load(record["npz_path"]) as d:
                    for k in d.files:
                        value = d[k]
                        if k in PAD_AXES:
                            value = _pad(value, PAD_AXES[k])
                        per_combo.setdefault(k, []).append(value)
                    ncp, nchemg = int(d["ncp"]), int(d["nchemg"])
            finally:
                Path(record["npz_path"]).unlink(missing_ok=True)
            widths[(s_i, c_i)] = (ncp, nchemg)
            local = build_grid(ncp, nchemg)
            per_combo.setdefault("in_nd", []).append(local["nd"])
            per_combo.setdefault("in_md", []).append(_pad(local["md"], {2: NCP_MAX}))
            per_combo.setdefault("in_mdt", []).append(local["mdt"])
            per_combo.setdefault("in_delgc", []).append(_pad(local["delgc"], {1: NCHEMG_MAX}))
        for k, v in per_combo.items():
            per_setup.setdefault(k, []).append(np.stack(v))
        if not quiet:
            print(f"  i_mode_setup = {setup}: {len(COMBOS)} combos x {len(SWITCHES)} switch pairs")

    data = {k: np.stack(v) for k, v in per_setup.items()}
    grid = build_grid(NCP_MAX, NCHEMG_MAX)

    check_varied(
        {
            f"setup_{s}_{c}": {
                "nd": data["nd"][i, j, 0].tolist(),
                "bud": data["bud"][i, j, 0].tolist(),
                "ageterm2": data["ageterm2"][i, j, 0].tolist(),
            }
            for i, s in enumerate(SETUPS)
            for j, c in enumerate(COMBOS)
        },
        expected_identical=EXPECTED_COLLISIONS,
        what="coagwithnucl setups",
    )

    report = verify_outputs(data, lambda s_i, c_i: {"nd": data["in_nd"][s_i, c_i]})

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ARCHIVE
    np.savez_compressed(
        path,
        setups=np.array(SETUPS, dtype=np.int64),
        combos=np.array(COMBOS, dtype="U16"),
        switches=np.array(SWITCHES, dtype=np.int64),
        blocks=np.array(BLOCKS, dtype="U16"),
        dtz=np.float64(DTZ),
        block=grid["block"],
        in_kii=grid["kii"],
        in_kij=grid["kij"],
        **data,
    )
    if not quiet:
        print(f"  wrote {path.name}: nd {data['nd'].shape}")
        print("  " + ", ".join(f"{k}={v}" for k, v in report.items()))
    return path


#: Collisions that are findings about the Fortran rather than capture bugs.
#:
#: `l_dust_mp_ageing` reaches this routine only through `topmode`, which bounds
#: the soluble loop's insoluble inner loop at `:342` and the insoluble outer
#: loop at `:427`. Setups 1, 3 and 5 have four active modes and setups 2 and 4
#: have five, the fifth being `mode_ait_insol` -- which `topmode = 5` already
#: covers. So the switch is inert in those five.
#:
#: Setups 6 and 8 are required to DIFFER, and do: setup 6 has `mode_acc_insol`
#: and `mode_cor_insol` active and setup 8 has all three insoluble modes, so
#: raising `topmode` brings modes 6 and 7 into both loops. Note the contrast
#: with `ukca_conden`, where setup 6 collided because it has no condensable
#: gas at all -- coagulation needs none.
EXPECTED_COLLISIONS: list[tuple[str, str]] = [
    (f"setup_{s}_default", f"setup_{s}_dust_ageing") for s in (1, 2, 3, 4, 5)
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        info = verify_source_structure()
        grid = build_grid(6, 12)
        print(f"leaf coagwithnucl sweep -> {args.out / ARCHIVE}")
        print(f"  rows        {grid['nd'].shape[0]:>6,}")
        for b in BLOCKS:
            print(f"    {b:<16}{int((grid['block'] == _block_code(b)).sum()):>6,}")
        print(
            f"  setups      {len(SETUPS):>6,} x {len(COMBOS)} combos x {len(SWITCHES)} switch pairs"
        )
        print(f"  budget names{info['budget_sites']:>6,} distinct nmascoag* write sites")
        return 0

    print(f"sweeping leaf_coagwithnucl -> {args.out}")
    capture(args.out)
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
