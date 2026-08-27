#!/usr/bin/env python3
"""Sweep `ukca_aero_step` -- the whole microphysics sequence (task 62).

    python validation/capture_aero_step_leaf.py --dry-run
    python validation/capture_aero_step_leaf.py     # writes tests/goldens/

Every routine this project has ported was checked against the reference **on
its own**. None has been checked against the others. `ukca_aero_step` is the
sequence: an initial re-mode, then `nmts` outer substeps each containing the
coagulation kernel and `nzts` competition substeps of gas production,
condensation, nucleation, coagulation and ageing, followed by re-mode and two
size recalculations.

So the twenty modules that each agree with the compiled routine either agree in
sequence or do not, and this is the fixture that says which.

What the wrapper bakes in
-------------------------

`ukca_aero_step` takes 96 arguments. `leaf_aero_step` exposes the ~40 the box
model varies and fixes the rest exactly as `glomap_box.F90:140-163` passes them:
every scavenging, deposition, cloud and nitrate switch off, `dryox_in_aer = 1`
so the gas-production term inside the competition loop is live,
`wetox_in_aer = 0`. That is the configuration `docs/unsupported.md` records, and
it is the only one this project has a reference for.

`verbose = 0`, deliberately: `ukca_calcminmaxndmdt` and `ukca_calcminmaxgc`
write to `umPrint` on every process at `verbose >= 2`, which under this binding
is thousands of lines through the shim per call.

What is and is not an input
---------------------------

`nd`, `md`, `mdt` and `s0g` are the state. `drydp`, `wetdp`, `dvol`, `wvol`,
`rhopar`, `mdwat`, `pvol` and `pvol_wat` are `INTENT(IN OUT)` but every one is
recomputed before it is read -- `:541` re-derives `drydp` before the initial
re-mode and `:564` re-derives the rest -- so their inputs cannot matter. The
grid feeds them zeros on purpose, and `verify_outputs` checks they come back
populated, which is a cheap way of confirming that reading.

The environment is the `boundary_layer` namelist's, because that is the
configuration the committed trajectory goldens were captured from: 288 K,
1e5 Pa, 60% RH, 500 m in a 1 km boundary layer.

Substep structure is swept
--------------------------

`(nmts, nzts)` covers `(1, 15)` -- the shipped setting -- and `(3, 15)`, which
`inputs/namelists/bl_nmts3.nml` added for exactly this reason. `dtz` is
`dtc/(nmts*nzts)`, so the two differ in how finely the competition between
nucleation and condensation is resolved, and a driver that mis-nested the two
loops would agree at `nmts = 1` and diverge at 3.
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
ARCHIVE = "aero_step.f64.leaf.npz"
NAMELIST_TEXT = (NAMELISTS / "boundary_layer.nml").read_text(encoding="utf-8")

#: Only the setups the box model can initialise with the shipped `nd_init`.
SETUPS: tuple[int, ...] = (1, 2, 4, 8)
NMODES = 8
NCP_MAX, NCHEMG_MAX, NADVG_MAX, NBUD_MAX = 8, 16, 20, 139

DT_CHEM = 1800.0
#: `(nmts, nzts)`. The shipped setting and `bl_nmts3`'s.
SUBSTEPS: tuple[tuple[int, int], ...] = ((1, 15), (3, 15))

#: `boundary_layer.nml`'s environment, the one the trajectory goldens use.
T = 288.0
PMID = 1.0e5
RH = 0.60
HEIGHT = 500.0
PBL = 1000.0

BLOCKS = ("namelist", "clean", "polluted")

#: The child stacks over the two substep settings before saving, so every
#: captured array carries a leading axis and the width-bearing axis is one
#: further right than the routine's signature suggests. Third fixture in a row
#: to trip on this; it fails as a `np.stack` error rather than a silent shift.
PAD_AXES: dict[str, dict[int, int]] = {
    "md": {3: NCP_MAX},  # (substep, box, mode, cp)
    "pvol": {3: NCP_MAX},
    "s0g": {2: NADVG_MAX},  # (substep, box, gas)
    "bud": {2: NBUD_MAX},
    "in_md": {2: NCP_MAX},
    "in_s0g": {1: NADVG_MAX},
    "in_s0g_dot": {1: NCHEMG_MAX},
}


def _block_code(name: str) -> int:
    return BLOCKS.index(name)


def _pad(array: np.ndarray, widths: dict[int, int]) -> np.ndarray:
    pad = [(0, 0)] * array.ndim
    for axis, width in widths.items():
        if array.shape[axis] > width:
            raise SystemExit(f"axis {axis} of {array.shape} exceeds {width}")
        pad[axis] = (0, width - array.shape[axis])
    return np.pad(array, pad, mode="constant")


def build_grid(ncp: int, nchemg: int, nadvg: int) -> dict[str, np.ndarray]:
    """Three aerosol loadings at the namelist's environment.

    `nd_init` from `boundary_layer.nml` is the middle one. The clean and
    polluted rows scale it, because the balance between nucleation and
    condensation is set by the condensation sink, and the namelist's own
    comment says nucleation there is suppressed by it -- so a single loading
    would exercise one side of the competition only.
    """
    nd_rows, s0g_rows, dot_rows, blocks = [], [], [], []

    def add(nd_scale, h2so4, prod, blk):
        nd = np.array([0.0, 1000.0, 100.0, 1.0, 0.0, 0.0, 0.0, 0.0]) * nd_scale
        nd_rows.append(nd)
        s0g = np.zeros(nadvg)
        s0g[:] = 0.0
        s0g_rows.append((s0g, h2so4))
        dot_rows.append(prod)
        blocks.append(_block_code(blk))

    add(1.0, 1.0e7, 1.0e5, "namelist")
    add(1.0e-3, 1.0e7, 1.0e5, "clean")
    add(1.0e-2, 1.0e8, 1.0e6, "clean")
    add(1.0e2, 1.0e6, 1.0e4, "polluted")
    add(1.0e3, 1.0e8, 1.0e6, "polluted")

    n = len(nd_rows)
    nd = np.stack(nd_rows)
    md = np.zeros((n, NMODES, ncp), dtype=np.float64)
    for i in range(n):
        for c in range(ncp):
            md[i, :, c] = 1.0e-19 * (c + 1)
    mdt = md.sum(axis=2)
    # `s0g` is a volume mixing ratio; the concentration is recovered inside
    # aero_step as `s0g*aird/sm`. Slot 2 is `mh2so4` in every supported setup.
    aird = PMID / (1.3804e-23 * T) * 1.0e-6
    s0g = np.zeros((n, nadvg), dtype=np.float64)
    s0g_dot = np.zeros((n, nchemg), dtype=np.float64)
    for i, (_row, h2so4) in enumerate(s0g_rows):
        s0g[i, 2] = h2so4 / aird
        s0g_dot[i, 2] = dot_rows[i] / aird
    ones = np.ones(n, dtype=np.float64)
    return {
        "nd": nd,
        "md": md,
        "mdt": mdt,
        "s0g": s0g,
        "s0g_dot": s0g_dot,
        "t": ones * T,
        "tsqrt": ones * np.sqrt(T),
        "pmid": ones * PMID,
        "pupper": ones * (PMID * 0.9),
        "plower": ones * (PMID * 1.1),
        "rh": ones * RH,
        "rh_clr": ones * RH,
        "s": ones * 6.0e-3,
        "aird": ones * aird,
        "airdm3": ones * (aird * 1.0e6),
        "rhoa": ones * (PMID / (287.05 * T)),
        "sm": ones * 1.0e5,
        "mfpa": ones * 6.6e-8,
        "dvisc": ones * 1.79e-5,
        "height": ones * HEIGHT,
        "htpblg": ones * PBL,
        "block": np.array(blocks, dtype=np.int32),
    }


CHILD_BODY = (
    "\nimport tempfile\n"
    f"sys.path.insert(0, {str(REPO / 'validation')!r})\n"
    "from capture_aero_step_leaf import build_grid, SUBSTEPS, DT_CHEM, NMODES\n"
    "from leaf_common import bind_call\n"
    "\n"
    "call = bind_call(g)\n"
    "_sz = call('sizes', g.wrap_sizes)\n"
    "_nbox, _nm, _ncp, _nchemg, _nadvg, _nbudaer = [int(v) for v in _sz[:6]]\n"
    "grid = build_grid(_ncp, _nchemg, _nadvg)\n"
    "keys = ('nd','mdt','md','mdwat','s0g','drydp','wetdp','rhopar','dvol',\n"
    "        'wvol','pvol','pvol_wat','bud','n_merge')\n"
    "out = {k: [] for k in keys}\n"
    "for _nmts, _nzts in SUBSTEPS:\n"
    "    _dtz = DT_CHEM / (_nmts * _nzts)\n"
    "    _z2 = np.zeros((grid['nd'].shape[0], NMODES))\n"
    "    _r = call('aero_step %d/%d' % (_nmts, _nzts), g.leaf_aero_step,\n"
    "        _nbudaer + 1, DT_CHEM, _dtz, _nmts, _nzts,\n"
    "        1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 1, 0, 0,\n"
    "        grid['nd'], grid['mdt'], grid['md'], _z2, grid['s0g'],\n"
    "        _z2, _z2, _z2, _z2, _z2,\n"
    "        grid['sm'], grid['aird'], grid['airdm3'], grid['rhoa'],\n"
    "        grid['mfpa'], grid['dvisc'],\n"
    "        grid['t'], grid['tsqrt'], grid['rh'], grid['rh_clr'], grid['s'],\n"
    "        grid['pmid'], grid['pupper'], grid['plower'],\n"
    "        grid['s0g_dot'], grid['height'], grid['htpblg'])\n"
    "    for _k, _v in zip(keys, _r[:14]):\n"
    "        out[_k].append(np.asarray(_v))\n"
    "\n"
    "_mode, _e1 = call('mode', g.wrap_mode_int, 'mode', NMODES)\n"
    "_tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)\n"
    "_tmp.close()\n"
    "np.savez(_tmp.name, mode=np.asarray(_mode), ncp=np.int64(_ncp),\n"
    "         nchemg=np.int64(_nchemg), nadvg=np.int64(_nadvg),\n"
    "         nbudaer=np.int64(_nbudaer),\n"
    "         **{k: np.stack(v) for k, v in out.items()})\n"
    "print('@@RESULT@@' + json.dumps({'npz_path': _tmp.name, 'setup': _setup}))\n"
)

_CHILD = CHILD_PREAMBLE.format(f2py=str(F2PY_DIR)) + CHILD_BODY


# --- verification -----------------------------------------------------------


def verify_outputs(data: dict, grid_of) -> dict:
    report = {"nucleated": 0, "condensed": 0, "merged": 0, "budget_written": 0}
    n_setup, n_sub = data["nd"].shape[:2]

    for s_i in range(n_setup):
        grid = grid_of(s_i)
        for k_i in range(n_sub):
            tag = f"setup index {s_i}, substep {k_i}"
            for name in ("nd", "md", "mdt", "s0g", "drydp", "wetdp", "rhopar", "bud"):
                arr = data[name][s_i, k_i]
                if not np.all(np.isfinite(arr)):
                    raise SystemExit(f"{tag}: {name} is not finite")
            if np.any(data["nd"][s_i, k_i] < 0.0):
                raise SystemExit(f"{tag}: nd went negative")
            if np.any(data["s0g"][s_i, k_i] < 0.0):
                raise SystemExit(f"{tag}: s0g went negative")

            # Every INTENT(IN OUT) size field is recomputed before it is read,
            # so feeding zeros in must give something non-zero out.
            for name in ("drydp", "wetdp", "dvol", "wvol", "rhopar"):
                active = data["nd"][s_i, k_i] > 0.0
                if active.any() and not np.any(data[name][s_i, k_i][active] > 0.0):
                    raise SystemExit(
                        f"{tag}: {name} came back zero on every active mode, so it was "
                        "not recomputed -- the pre-loop drydiam/volume_mode did not run"
                    )

            report["nucleated"] += int((data["nd"][s_i, k_i][:, 0] > grid["nd"][:, 0]).sum())
            # NOT `s0g` falling: `dryox_in_aer = 1` means the gas gains from
            # `s0g_dot` inside the competition loop, so H2SO4 rises net even
            # while condensation removes some of it. The first draft asserted
            # the fall and the capture refused the golden. What condensation
            # does show in is the aerosol mass.
            mass_in = (grid["nd"] * grid["mdt"]).sum(axis=1)
            mass_out = (data["nd"][s_i, k_i] * data["mdt"][s_i, k_i]).sum(axis=1)
            report["condensed"] += int((mass_out > mass_in).sum())
            report["merged"] += int((data["n_merge"][s_i, k_i] > 0).sum())
            report["budget_written"] += int(np.any(data["bud"][s_i, k_i] != 0.0))

    if np.any(data["bud"][..., 0] != 0.0):
        raise SystemExit("budget slot 0 -- the NOT_CARRIED hole -- was written")
    if report["nucleated"] == 0:
        raise SystemExit(
            "the nucleation mode never gained number, so nucleation contributed "
            "nothing on any row and the sequence is only testing condensation"
        )
    if report["condensed"] == 0:
        raise SystemExit(
            "total aerosol mass never grew, so neither condensation nor nucleation "
            "put anything into the particles"
        )
    if report["budget_written"] == 0:
        raise SystemExit("no configuration wrote a budget field")
    return report


def capture(out_dir: Path, quiet: bool = False) -> Path:
    per_setup: dict[str, list] = {}
    for setup in SETUPS:
        record = run_child(
            CHILD_BODY,
            namelist_text=render_namelist(NAMELIST_TEXT, setup, "default"),
            setup=setup,
            label=f"aero_step setup {setup}",
        )
        try:
            with np.load(record["npz_path"]) as d:
                for k in d.files:
                    value = _pad(d[k], PAD_AXES[k]) if k in PAD_AXES else d[k]
                    per_setup.setdefault(k, []).append(value)
                ncp, nchemg, nadvg = int(d["ncp"]), int(d["nchemg"]), int(d["nadvg"])
        finally:
            Path(record["npz_path"]).unlink(missing_ok=True)
        local = build_grid(ncp, nchemg, nadvg)
        per_setup.setdefault("in_nd", []).append(local["nd"])
        per_setup.setdefault("in_md", []).append(_pad(local["md"], {2: NCP_MAX}))
        per_setup.setdefault("in_mdt", []).append(local["mdt"])
        per_setup.setdefault("in_s0g", []).append(_pad(local["s0g"], {1: NADVG_MAX}))
        per_setup.setdefault("in_s0g_dot", []).append(_pad(local["s0g_dot"], {1: NCHEMG_MAX}))
        if not quiet:
            print(f"  i_mode_setup = {setup}: {len(SUBSTEPS)} substep settings")

    data = {k: np.stack(v) for k, v in per_setup.items()}
    grid = build_grid(NCP_MAX, NCHEMG_MAX, NADVG_MAX)

    check_varied(
        {
            f"setup_{s}_nmts{m}": {
                "nd": data["nd"][i, k].tolist(),
                "s0g": data["s0g"][i, k].tolist(),
                "bud": data["bud"][i, k].tolist(),
            }
            for i, s in enumerate(SETUPS)
            for k, (m, _z) in enumerate(SUBSTEPS)
        },
        expected_identical=[],
        what="aero_step configurations",
    )
    report = verify_outputs(
        data,
        lambda s_i: {
            "nd": data["in_nd"][s_i],
            "s0g": data["in_s0g"][s_i],
            "mdt": data["in_mdt"][s_i],
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ARCHIVE
    np.savez_compressed(
        path,
        setups=np.array(SETUPS, dtype=np.int64),
        substeps=np.array(SUBSTEPS, dtype=np.int64),
        blocks=np.array(BLOCKS, dtype="U16"),
        dt_chem=np.float64(DT_CHEM),
        **{f"in_{k}": v for k, v in grid.items() if k not in ("nd", "md", "mdt", "s0g", "s0g_dot")},
        **data,
    )
    if not quiet:
        print(f"  wrote {path.name}: nd {data['nd'].shape}")
        print("  " + ", ".join(f"{k}={v}" for k, v in report.items()))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        grid = build_grid(6, 12, 14)
        print(f"leaf aero_step sweep -> {args.out / ARCHIVE}")
        print(f"  boxes       {grid['nd'].shape[0]:>6,}")
        for b in BLOCKS:
            print(f"    {b:<12}{int((grid['block'] == _block_code(b)).sum()):>6,}")
        print(f"  setups      {len(SETUPS):>6,} x {len(SUBSTEPS)} substep settings")
        print(f"  dtz         {[DT_CHEM / (m * z) for m, z in SUBSTEPS]}")
        return 0

    print(f"sweeping leaf_aero_step -> {args.out}")
    capture(args.out)
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
