#!/usr/bin/env python3
"""Sweep `ukca_remode` over all seven setups and all three `imerge` (task 60).

    python validation/capture_remode_leaf.py --dry-run
    python validation/capture_remode_leaf.py     # writes tests/goldens/

`ukca_remode` merges the top tail of a mode into the next one up when its dry
diameter has grown past a threshold. **Issue #12: no shipped namelist ever
merges a mode**, so this routine has zero trajectory coverage and every branch
in it has to be reached by construction.

`drydp` is the input, and it is `ukca_calc_drydiam`'s output -- already ported
and byte-equal -- which is what makes handing it in directly legitimate rather
than inventing physics.

The mode loop is loop-carried, reachably
----------------------------------------

`:206` runs `DO imode=1,nmodemax_merge` and each pass writes `nd(jl,imode+1)`
and `md(jl,imode+1,:)` (`:369-380`), which the *next* pass reads at `:213`
as `dp_ip1` and at `:373` as the receiving mode's state. Unlike `ukca_ageing`'s
two collisions -- both unreachable -- this one needs nothing exotic: a box whose
nucleation *and* Aitken modes have both grown past threshold merges twice, and
the second merge sees the first's result. The grid carries such boxes on
purpose, and the port's test requires the broadcast form to differ.

Branch coverage the grid is built for
-------------------------------------

* `imerge` 1, 2 and 3 -- the mid-point, the bin edge, and the geometric mean of
  the two diameters. `imerge = 3` merges **unconditionally** (`:234` is
  `(dp > dp_thresh1) .OR. (imerge == 3)`), so it is the only setting that runs
  on an unmerged distribution.
* `pmid` straddling `1.0e4`: `:203-204` drops `nmodemax_merge` from 3 to 2 in
  the stratosphere, so the accumulation mode stops merging.
* `dp > dp_thresh1` both ways -- the criterion no namelist reaches.
* `nd > num_eps` both ways: the false arm at `:389-396` resets `md`/`mdt` to
  `mmid*mfrac_0` instead of merging.
* `newn > num_eps` both ways: false means the merge is computed and then
  discarded.
* the `frac_n < 0.5` and `frac_m < 0.001` clamps, both of which are silent.

`umErf` is the one primitive this routine leans on, and the numerics sweep found
it bit-identical between gfortran and JAX on the capture platform -- which is
why the merge fraction can be gated exactly. `EXP` appears once, at `:252`
(`dp2`), so issue #28 has exactly one place to bite.
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
ARCHIVE = "remode.f64.leaf.npz"
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_remode.F90"
NAMELIST_TEXT = (NAMELISTS / "boundary_layer.nml").read_text(encoding="utf-8")

SETUPS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8)
IMERGES: tuple[int, ...] = (1, 2, 3)
NMODES = 8
NCP_MAX, NBUD_MAX = 8, 139

#: `:204`. Below this the accumulation mode stops merging.
P_STRAT = 1.0e4

BLOCKS = ("unmerged", "one_merge", "two_merges", "sparse", "strat")

#: The child stacks over the three `imerge` settings before saving, so every
#: captured array carries a leading switch axis and the width-bearing axis sits
#: one further right than the routine's signature suggests. Getting it wrong is
#: a `np.stack` failure, not a silent shift.
PAD_AXES: dict[str, dict[int, int]] = {
    "md": {3: NCP_MAX},  # (imerge, box, mode, cp)
    "bud": {2: NBUD_MAX},  # (imerge, box, slot)
    "in_md": {2: NCP_MAX},  # (box, mode, cp) -- built here, no imerge axis
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


#: A dry diameter per mode that no namelist produces. The merge criterion is
#: `drydp(imode) > ddpmid(imode+1)` at `imerge = 1`, and `ddpmid` for the
#: soluble modes is roughly 1e-8, 6e-8, 4e-7, 3e-6 -- so a mode has to have
#: grown into the *next* mode's mid-point, which is what "no namelist ever
#: merges" means.
DRYDP_BASE = (8.0e-9, 6.0e-8, 4.0e-7, 4.0e-6, 1.0e-7, 6.0e-7, 6.0e-6, 2.0e-5)


def build_grid(ncp: int) -> dict[str, np.ndarray]:
    rows: list[dict] = []

    def add(*, scale, nd_scale, pmid, blk):
        rows.append({"scale": scale, "nd_scale": nd_scale, "pmid": pmid, "blk": blk})

    # Unmerged: diameters as the model produces them, so no criterion fires
    # (except imerge = 3, which fires regardless).
    for nd_scale in (1.0, 1.0e2):
        add(scale=(1.0,) * NMODES, nd_scale=nd_scale, pmid=1.0e5, blk="unmerged")
    # One merge: only the nucleation mode has grown past threshold.
    for grow in (2.0, 6.0, 20.0):
        add(scale=(grow,) + (1.0,) * (NMODES - 1), nd_scale=1.0, pmid=1.0e5, blk="one_merge")
    # Two merges in the same box: nucleation AND Aitken past threshold, so the
    # second pass reads what the first wrote. This is the loop-carried case.
    for grow in (4.0, 10.0, 30.0):
        add(scale=(grow, grow) + (1.0,) * (NMODES - 2), nd_scale=1.0, pmid=1.0e5, blk="two_merges")
    # Three merges: all of nuc, Aitken and accumulation.
    add(
        scale=(10.0, 10.0, 10.0) + (1.0,) * (NMODES - 3), nd_scale=1.0, pmid=1.0e5, blk="two_merges"
    )
    # Sparse: number below num_eps, so the reset arm at :389-396 runs instead.
    for nd_scale in (0.0, 1.0e-13, 1.0e-9):
        add(scale=(10.0, 10.0) + (1.0,) * (NMODES - 2), nd_scale=nd_scale, pmid=1.0e5, blk="sparse")
    # Stratosphere: nmodemax_merge drops to 2, so the accumulation mode is
    # excluded. Straddles the threshold with both neighbouring doubles.
    for pmid in (
        float(np.nextafter(P_STRAT, -np.inf)),
        P_STRAT,
        float(np.nextafter(P_STRAT, np.inf)),
        5.0e3,
        2.0e4,
    ):
        add(scale=(10.0, 10.0, 10.0) + (1.0,) * (NMODES - 3), nd_scale=1.0, pmid=pmid, blk="strat")

    n = len(rows)
    drydp = np.stack([np.array(DRYDP_BASE) * np.array(r["scale"]) for r in rows])
    nd = np.stack(
        [
            np.array([1.0e4, 5.0e3, 1.0e3, 1.0e2, 2.0e3, 5.0e2, 1.0e2, 1.0e1]) * r["nd_scale"]
            for r in rows
        ]
    )
    md = np.zeros((n, NMODES, ncp), dtype=np.float64)
    for i in range(n):
        for c in range(ncp):
            md[i, :, c] = 1.0e-19 * (c + 1)
    return {
        "nd": nd,
        "md": md,
        "mdt": md.sum(axis=2),
        "drydp": drydp,
        "pmid": np.array([r["pmid"] for r in rows], dtype=np.float64),
        "block": np.array([_block_code(r["blk"]) for r in rows], dtype=np.int32),
    }


CHILD_BODY = (
    "\nimport tempfile\n"
    f"sys.path.insert(0, {str(REPO / 'validation')!r})\n"
    "from capture_remode_leaf import build_grid, IMERGES, NMODES\n"
    "from leaf_common import bind_call\n"
    "\n"
    "call = bind_call(g)\n"
    "_sz = call('sizes', g.wrap_sizes)\n"
    "_nbox, _nm, _ncp, _nchemg, _nadvg, _nbudaer = [int(v) for v in _sz[:6]]\n"
    "grid = build_grid(_ncp)\n"
    "out = {k: [] for k in ('nd','md','mdt','bud','n_merge')}\n"
    "for _im in IMERGES:\n"
    "    _nd, _md, _mdt, _bud, _nmg, _e = call('remode %d' % _im, g.leaf_remode,\n"
    "        _nbudaer + 1, _im, grid['nd'], grid['md'], grid['mdt'],\n"
    "        grid['drydp'], grid['pmid'])\n"
    "    out['nd'].append(np.asarray(_nd)); out['md'].append(np.asarray(_md))\n"
    "    out['mdt'].append(np.asarray(_mdt)); out['bud'].append(np.asarray(_bud))\n"
    "    out['n_merge'].append(np.asarray(_nmg))\n"
    "_mode, _e1 = call('mode', g.wrap_mode_int, 'mode', NMODES)\n"
    "_tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)\n"
    "_tmp.close()\n"
    "np.savez(_tmp.name, mode=np.asarray(_mode), ncp=np.int64(_ncp),\n"
    "         nbudaer=np.int64(_nbudaer),\n"
    "         **{k: np.stack(v) for k, v in out.items()})\n"
    "print('@@RESULT@@' + json.dumps({'npz_path': _tmp.name, 'setup': _setup}))\n"
)

_CHILD = CHILD_PREAMBLE.format(f2py=str(F2PY_DIR)) + CHILD_BODY


# --- verification -----------------------------------------------------------


def verify_source_structure() -> dict[str, int]:
    import re

    text = SOURCE.read_text(encoding="utf-8")
    sq = re.sub(r"\s+", "", re.sub(r"!.*", "", re.sub(r"&\s*\n\s*", "", text)))
    required = {
        "strat bound": "WHERE(pmid(:)<1.0e4)nmodemax_merge(:)=2" in sq,
        "merge criterion": "IF((dp>dp_thresh1).OR.(imerge==3))THEN" in sq,
        "frac_n clamp": "IF(frac_n<0.5)frac_n=0.5" in sq,
        "frac_m clamp": "IF(frac_m<0.001)frac_m=0.001" in sq,
        "dp2": "dp2=EXP(LOG(dp)+3.0*log2sg)" in sq,
        "erf number": "frac_n=0.5*(1.0+umErf(erfnum))" in sq,
        "erf mass": "frac_m=0.5*(1.0+umErf(erfmas))" in sq,
        "receiving mode written": "nd(jl,imode+1)=newnp1" in sq,
        "reset arm": "md(jl,imode,icp)=mmid(imode)*mfrac_0(imode,icp)" in sq,
    }
    missing = sorted(k for k, ok in required.items() if not ok)
    if missing:
        raise SystemExit("the vendored routine no longer contains: " + ", ".join(missing))
    sites = re.findall(r"bud_aer_mas\(jl,(nmasmerg\w+)\)", sq)
    return {"checked": len(required), "budget_names": len(set(sites))}


def verify_outputs(data: dict, grid: dict) -> dict:
    report = {
        "merged_slots": 0,
        "two_merge_boxes": 0,
        "mdt_moved_uncounted": 0,
        "budget_written": 0,
    }
    n_setup = data["nd"].shape[0]

    for s_i in range(n_setup):
        for k_i, imerge in enumerate(IMERGES):
            nd = data["nd"][s_i, k_i]
            n_merge = data["n_merge"][s_i, k_i]
            for name, arr in (
                ("nd", nd),
                ("md", data["md"][s_i, k_i]),
                ("mdt", data["mdt"][s_i, k_i]),
                ("bud", data["bud"][s_i, k_i]),
            ):
                if not np.all(np.isfinite(arr)):
                    raise SystemExit(f"setup index {s_i}, imerge {imerge}: {name} not finite")
            if np.any(nd < 0.0):
                raise SystemExit(f"setup index {s_i}, imerge {imerge}: nd went negative")

            # `:240`: n_merge counts a merge attempt per (box, mode).
            report["merged_slots"] += int((n_merge > 0).sum())
            report["two_merge_boxes"] += int(((n_merge > 0).sum(axis=1) >= 2).sum())
            # (box, mode) pairs whose mdt moved without n_merge being
            # incremented. That is NOT the reset arm alone: the receiving mode
            # `imode+1` also has its mdt rewritten at `:369-378` while only
            # `imode` gets counted at `:240`. Named for what it measures rather
            # than for what it was first assumed to measure.
            changed_mdt = data["mdt"][s_i, k_i] != grid["mdt"]
            report["mdt_moved_uncounted"] += int((changed_mdt & (n_merge == 0)).sum())
            report["budget_written"] += int(np.any(data["bud"][s_i, k_i] != 0.0))

            # `:203-204`: in the stratosphere the accumulation mode is excluded.
            strat = grid["pmid"] < P_STRAT
            if np.any(n_merge[strat, 2] > 0):
                raise SystemExit(
                    f"imerge {imerge}: the accumulation mode merged below "
                    f"{P_STRAT}, where nmodemax_merge is 2"
                )

    if np.any(data["bud"][..., 0] != 0.0):
        raise SystemExit("budget slot 0 -- the NOT_CARRIED hole -- was written")
    if report["merged_slots"] == 0:
        raise SystemExit(
            "nothing merged anywhere; issue #12 is about exactly this and the whole "
            "routine would be unvalidated"
        )
    if report["two_merge_boxes"] == 0:
        raise SystemExit(
            "no box merges twice, so the loop-carried dependency between mode "
            "imode and imode+1 is never exercised and a broadcast port would pass"
        )
    if report["budget_written"] == 0:
        raise SystemExit("no configuration wrote a merge budget field")
    return report


def capture(out_dir: Path, quiet: bool = False) -> Path:
    verify_source_structure()

    per_setup: dict[str, list] = {}
    for setup in SETUPS:
        record = run_child(
            CHILD_BODY,
            namelist_text=render_namelist(NAMELIST_TEXT, setup, "default"),
            setup=setup,
            label=f"remode setup {setup}",
        )
        try:
            with np.load(record["npz_path"]) as d:
                for k in d.files:
                    value = _pad(d[k], PAD_AXES[k]) if k in PAD_AXES else d[k]
                    per_setup.setdefault(k, []).append(value)
                ncp = int(d["ncp"])
        finally:
            Path(record["npz_path"]).unlink(missing_ok=True)
        local = build_grid(ncp)
        per_setup.setdefault("in_md", []).append(_pad(local["md"], {2: NCP_MAX}))
        # `in_mdt` MUST be per-setup: it is `md.sum(axis=2)` over that setup's
        # own `ncp`, and a copy summed at the padded width is a value no
        # configuration was ever given. Stored once at NCP_MAX in the first
        # draft, it made the port look wrong on 94 of 136 mdt entries while
        # every other field was byte-equal.
        per_setup.setdefault("in_mdt", []).append(local["mdt"])
        if not quiet:
            print(f"  i_mode_setup = {setup}: {len(IMERGES)} imerge settings")

    data = {k: np.stack(v) for k, v in per_setup.items()}
    grid = build_grid(NCP_MAX)

    check_varied(
        {
            # Four fields, not two. Keyed on `nd` and `n_merge` alone this
            # reported 48 collisions: the merge *number* depends only on
            # ddpmid/ddplim0, sigmag and num_eps for the three soluble modes,
            # which are identical across setups 1-5 and 8. What separates them
            # is which components the modes carry and which budget slots they
            # own -- so `mdt` and `bud` are what discriminate. Third time an
            # anti-collapse fingerprint has been too narrow in this project.
            f"setup_{s}_imerge{m}": {
                "nd": data["nd"][i, k].tolist(),
                "n_merge": data["n_merge"][i, k].tolist(),
                "mdt": data["mdt"][i, k].tolist(),
                "bud": data["bud"][i, k].tolist(),
            }
            for i, s in enumerate(SETUPS)
            for k, m in enumerate(IMERGES)
        },
        expected_identical=EXPECTED_COLLISIONS,
        what="remode configurations",
    )
    report = verify_outputs(data, grid)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ARCHIVE
    np.savez_compressed(
        path,
        setups=np.array(SETUPS, dtype=np.int64),
        imerges=np.array(IMERGES, dtype=np.int64),
        blocks=np.array(BLOCKS, dtype="U16"),
        # `md` and `mdt` are per-setup and already in `data`; the rest are
        # setup-independent and stored once.
        **{f"in_{k}": v for k, v in grid.items() if k not in ("md", "mdt")},
        **data,
    )
    if not quiet:
        print(f"  wrote {path.name}: nd {data['nd'].shape}")
        print("  " + ", ".join(f"{k}={v}" for k, v in report.items()))
    return path


#: Setup 6 activates only `mode_acc_insol` and `mode_cor_insol`. `ukca_remode`
#: merges soluble modes 1-3 alone (`:206`, bounded by `nmodemax_merge <= 3`),
#: so nothing in setup 6 can merge and all three `imerge` settings agree.
EXPECTED_COLLISIONS: list[tuple[str, str]] = [
    ("setup_6_imerge1", "setup_6_imerge2"),
    ("setup_6_imerge1", "setup_6_imerge3"),
    ("setup_6_imerge2", "setup_6_imerge3"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        info = verify_source_structure()
        grid = build_grid(6)
        print(f"leaf remode sweep -> {args.out / ARCHIVE}")
        print(f"  rows        {grid['nd'].shape[0]:>6,}")
        for b in BLOCKS:
            print(f"    {b:<12}{int((grid['block'] == _block_code(b)).sum()):>6,}")
        print(f"  setups      {len(SETUPS):>6,} x {len(IMERGES)} imerge settings")
        print(f"  merge budget names: {info['budget_names']}")
        return 0

    print(f"sweeping leaf_remode -> {args.out}")
    capture(args.out)
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
