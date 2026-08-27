#!/usr/bin/env python3
"""Sweep `ukca_ageing` over all seven supported setups (task 58).

    python validation/capture_ageing_leaf.py --dry-run
    python validation/capture_ageing_leaf.py     # writes tests/goldens/

`ukca_ageing` moves particles from an insoluble mode to its soluble partner
once enough soluble material has accumulated on them. `ageterm1` is what
condensation put there (`ukca_conden`), `ageterm2` what coagulation did
(`ukca_coagwithnucl`); both are inputs, both already ported and pinned, and
both are zero on every trajectory this project can run -- so constructing them
is the only way to reach any of this.

Two claims in CLAUDE.md that this capture checks
------------------------------------------------

CLAUDE.md lists `ukca_ageing` twice among the five loops needing sequential
treatment: "over modes (7->4 and 8->4 collide) and over `jv`".

**The mode collision is unreachable.** `tmode` is `imode - 3` below
`mode_sup_insol` and `imode - 4` at it (`:219-223`), so 7 and 8 both target
mode 4 -- structurally true. But `mode_sup_insol` needs setup 12 or 13, which
the box model does not implement, so no supported setup activates mode 8.
Measured across all seven setups at both `l_dust_mp_ageing` settings: the
insoluble modes that enter the loop are `{5, 6, 7}` and their targets `{2, 3,
4}`, always distinct. The mode loop is therefore parallelisable in every
configuration this project can validate, and no both-settings test for it can
exist. `verify_target_modes` asserts that rather than leaving it implied.

**The `jv` collision is real but static.** `cp_coag_added(icp)` at `:265` stops
the coagulation term being added twice when two condensable gases share a
component -- `msec_org` and `msec_orgi` both map to `cp_oc`. That is a genuine
loop-carried guard, and it depends only on `condensable_choice`, which is a
compile-time table: the gas that adds the term is the lowest `jv` with that
component. So the loop unrolls, and what the port's test must show is that
*ignoring* the guard double-counts.

UP-3 is here, and the rescale is a no-op
----------------------------------------

`:302-306`::

    IF (naged > nd(jl,imode)) THEN
      naged=nd(jl,imode)          ! limit so no -ves
      totage(:)=totage(:)*nd(jl,imode)/naged
    END IF

`naged` is overwritten *before* it is used as the divisor, so `nd/naged` is
exactly 1 and the rescale does nothing. The comment says it "reduces ageing if
limited by insoluble particles"; it does not. `ageing_totage_rescale_noop`
defaults to reproducing that, and CLAUDE.md records that the naive fix loses
mass -- so the grid is built to reach the branch, which needs `naged` to exceed
the insoluble mode's number.
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
ARCHIVE = "ageing.f64.leaf.npz"
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_ageing.F90"
NAMELIST_TEXT = (NAMELISTS / "boundary_layer.nml").read_text(encoding="utf-8")

SETUPS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8)
COMBOS: tuple[str, ...] = ("default", "dust_ageing")
NMODES, NMODES_SOL, NMODES_INS = 8, 4, 4
NCP_MAX, NCHEMG_MAX, NBUD_MAX = 8, 16, 139

BLOCKS = ("ordinary", "over_limit", "sparse", "no_material")

PAD_AXES: dict[str, dict[int, int]] = {
    "md": {2: NCP_MAX},
    "bud": {1: NBUD_MAX},
    "in_md": {1: NCP_MAX},
    "in_ageterm1": {2: NCHEMG_MAX},
    "in_ageterm2": {3: NCP_MAX},
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


def build_grid(ncp: int, nchemg: int) -> dict[str, np.ndarray]:
    """Rows spanning the ageing branches.

    `naged` is `totage_jv/age1ptcl/10` summed over gases, and `age1ptcl` is
    `(wetdp/dimen)**2` -- so the `naged > nd` branch is reached by making
    `ageterm1` large against a small insoluble `nd`, and the ordinary branch by
    the reverse. `wetdp` enters only through `age1ptcl`, quadratically.
    """
    rows: list[dict] = []

    def add(*, nd_scale, age1, age2, wetdp_scale, blk):
        rows.append(
            {
                "nd": np.array([1.0e4, 5.0e3, 1.0e3, 1.0e2, 2.0e3, 5.0e2, 1.0e2, 1.0e1]) * nd_scale,
                "age1": age1,
                "age2": age2,
                "wetdp_scale": wetdp_scale,
                "blk": blk,
            }
        )

    for age1 in (1.0e-14, 1.0e-12, 1.0e-10):
        for wetdp_scale in (0.5, 1.0, 2.0):
            add(nd_scale=1.0, age1=age1, age2=age1 * 0.5, wetdp_scale=wetdp_scale, blk="ordinary")
    # naged > nd: UP-3's branch, and it takes a lot of material to reach.
    # naged is `totage_jv/age1ptcl/10` summed over condensable gases, and
    # `age1ptcl = (wetdp/dimen)**2` is about 5e4 at wetdp = 1e-7 and
    # dimen = 4.5e-10 -- so naged is roughly ageterm1/5e5 per gas. Getting it
    # past an insoluble number of a few particles needs ageterm1 around 1e6,
    # which is large but is a count of molecules, not a concentration. The
    # first draft topped out at 1.0 and the capture refused the golden.
    for age1 in (1.0e6, 1.0e9, 1.0e12):
        for nd_scale in (1.0e-6, 1.0e-3, 1.0):
            add(nd_scale=nd_scale, age1=age1, age2=age1, wetdp_scale=1.0, blk="over_limit")
    # Number below num_eps, so the outer guard rejects the box entirely.
    for nd_scale in (0.0, 1.0e-13, 1.0e-9):
        add(nd_scale=nd_scale, age1=1.0e-10, age2=1.0e-10, wetdp_scale=1.0, blk="sparse")
    # Nothing accumulated: naged stays 0 and no transfer happens.
    add(nd_scale=1.0, age1=0.0, age2=0.0, wetdp_scale=1.0, blk="no_material")

    n = len(rows)
    nd = np.stack([r["nd"] for r in rows])
    wetdp = np.stack(
        [
            np.array([8.0e-9, 6.0e-8, 4.0e-7, 4.0e-6, 1.0e-7, 6.0e-7, 6.0e-6, 2.0e-5])
            * r["wetdp_scale"]
            for r in rows
        ]
    )
    md = np.zeros((n, NMODES, ncp), dtype=np.float64)
    for i in range(n):
        for c in range(ncp):
            md[i, :, c] = 1.0e-19 * (c + 1)
    mdt = md.sum(axis=2)
    ageterm1 = np.zeros((n, NMODES_INS, nchemg), dtype=np.float64)
    ageterm2 = np.zeros((n, NMODES_SOL, NMODES_INS, ncp), dtype=np.float64)
    for i, r in enumerate(rows):
        ageterm1[i, :, :] = r["age1"]
        ageterm2[i, :, :, :] = r["age2"]
    return {
        "nd": nd,
        "md": md,
        "mdt": mdt,
        "wetdp": wetdp,
        "ageterm1": ageterm1,
        "ageterm2": ageterm2,
        "block": np.array([_block_code(r["blk"]) for r in rows], dtype=np.int32),
    }


CHILD_BODY = (
    "\nimport tempfile\n"
    f"sys.path.insert(0, {str(REPO / 'validation')!r})\n"
    "from capture_ageing_leaf import build_grid, NMODES, NMODES_SOL, NMODES_INS\n"
    "from leaf_common import bind_call\n"
    "\n"
    "call = bind_call(g)\n"
    "_sz = call('sizes', g.wrap_sizes)\n"
    "_nbox, _nm, _ncp, _nchemg, _nadvg, _nbudaer = [int(v) for v in _sz[:6]]\n"
    "grid = build_grid(_ncp, _nchemg)\n"
    "# f2py infers n, nm, ncp_in, nchem, nmsol and nmins from the array\n"
    "# shapes and leaves only nbud1 explicit -- nbudaer is a module scalar with\n"
    "# no array to carry it. Passing the other sizes positionally shifts nbud1\n"
    "# onto n and fails loudly, which is how this was found.\n"
    "_nd, _md, _mdt, _bud, _e = call('ageing', g.leaf_ageing, _nbudaer + 1,\n"
    "    grid['nd'], grid['md'], grid['mdt'], grid['ageterm1'],\n"
    "    grid['ageterm2'], grid['wetdp'])\n"
    "_mode, _e1 = call('mode', g.wrap_mode_int, 'mode', NMODES)\n"
    "_topmode, _e2 = call('topmode', g.wrap_topmode)\n"
    "_tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)\n"
    "_tmp.close()\n"
    "np.savez(_tmp.name, nd=np.asarray(_nd), md=np.asarray(_md),\n"
    "         mdt=np.asarray(_mdt), bud=np.asarray(_bud),\n"
    "         mode=np.asarray(_mode), topmode=np.int64(_topmode),\n"
    "         ncp=np.int64(_ncp), nchemg=np.int64(_nchemg), nbudaer=np.int64(_nbudaer))\n"
    "print('@@RESULT@@' + json.dumps({'npz_path': _tmp.name, 'setup': _setup}))\n"
)

_CHILD = CHILD_PREAMBLE.format(f2py=str(F2PY_DIR)) + CHILD_BODY


# --- verification -----------------------------------------------------------


def verify_source_structure() -> dict[str, int]:
    import re

    text = SOURCE.read_text(encoding="utf-8")
    sq = re.sub(r"\s+", "", re.sub(r"!.*", "", re.sub(r"&\s*\n\s*", "", text)))
    required = {
        "tmode below sup": "tmode=imode-3" in sq,
        "tmode at sup": "tmode=imode-4" in sq,
        "up3 overwrite": "naged=nd(jl,imode)totage(:)=totage(:)*nd(jl,imode)/naged" in sq,
        "cp_coag_added guard": "IF(cp_coag_added(icp)==0)THEN" in sq,
        "naged divide by 10": "naged_jv(jv)=totage_jv/age1ptcl/10.0" in sq,
        "age1ptcl": "age1ptcl=wetdp(jl,imode)*wetdp(jl,imode)/dimen(jv)/dimen(jv)" in sq,
        "outer guard": "IF(naged>num_eps(imode))THEN" in sq,
    }
    missing = sorted(k for k, ok in required.items() if not ok)
    if missing:
        raise SystemExit("the vendored routine no longer contains: " + ", ".join(missing))
    return {"checked": len(required)}


def target_mode(imode: int) -> int:
    """`:219-223`, 1-based. 7 and 8 both target 4."""
    return imode - 3 if imode < 8 else imode - 4


def verify_target_modes() -> dict[str, int]:
    """CLAUDE.md's mode collision, checked rather than repeated.

    Raises if any supported setup ever activates `mode_sup_insol`, because then
    the collision becomes live and the port's mode loop must become sequential.
    """
    sys.path.insert(0, str(REPO / "src"))
    from glomap_jax.physics import modes as md

    pairs = 0
    for setup in SETUPS:
        for dust in (False, True):
            t = md.build(setup, l_dust_mp_ageing=dust)
            in_loop = [i for i in range(5, int(t.topmode) + 1) if t.mode[i - 1]]
            targets = [target_mode(i) for i in in_loop]
            if 8 in in_loop:
                raise SystemExit(
                    f"setup {setup} (dust={dust}) activates mode_sup_insol, so 7->4 and "
                    "8->4 now collide and the port's mode loop must be made sequential"
                )
            if len(targets) != len(set(targets)):
                raise SystemExit(f"setup {setup} (dust={dust}): targets {targets} collide")
            pairs += len(targets)
    return {"mode_pairs": pairs}


def verify_outputs(data: dict, grid_of) -> dict:
    report = {"aged_rows": 0, "over_limit_rows": 0, "budget_written": 0}
    for s_i in range(data["nd"].shape[0]):
        for c_i in range(data["nd"].shape[1]):
            grid = grid_of(s_i, c_i)
            nd, md = data["nd"][s_i, c_i], data["md"][s_i, c_i]
            for name, arr in (
                ("nd", nd),
                ("md", md),
                ("mdt", data["mdt"][s_i, c_i]),
                ("bud", data["bud"][s_i, c_i]),
            ):
                if not np.all(np.isfinite(arr)):
                    raise SystemExit(f"setup index {s_i}/{c_i}: {name} is not finite")
            if np.any(nd < 0.0):
                raise SystemExit(f"setup index {s_i}/{c_i}: nd went negative")
            # Ageing moves number from insoluble to soluble, so the insoluble
            # modes may only lose and their targets only gain.
            for imode in range(5, 9):
                t = target_mode(imode)
                moved = grid["nd"][:, imode - 1] - nd[:, imode - 1]
                if np.any(moved < -0.0):
                    raise SystemExit(f"insoluble mode {imode} gained number")
                gained = nd[:, t - 1] - grid["nd"][:, t - 1]
                if np.any(gained < -0.0):
                    raise SystemExit(f"soluble mode {t} lost number")
                report["aged_rows"] += int((moved > 0.0).sum())
                # The UP-3 branch: everything left, so the insoluble mode is
                # emptied exactly.
                report["over_limit_rows"] += int(
                    ((nd[:, imode - 1] == 0.0) & (grid["nd"][:, imode - 1] > 0.0)).sum()
                )
            report["budget_written"] += int(np.any(data["bud"][s_i, c_i] != 0.0))

    if np.any(data["bud"][..., 0] != 0.0):
        raise SystemExit("budget slot 0 -- the NOT_CARRIED hole -- was written")
    if report["aged_rows"] == 0:
        raise SystemExit("no particles aged anywhere; the grid misses the whole routine")
    if report["over_limit_rows"] == 0:
        raise SystemExit(
            "no row reaches naged > nd, which is UP-3's branch and the only place the "
            "totage rescale runs -- the flag would have nothing to be wrong about"
        )
    return report


def capture(out_dir: Path, quiet: bool = False) -> Path:
    verify_source_structure()
    verify_target_modes()

    per_setup: dict[str, list] = {}
    for setup in SETUPS:
        per_combo: dict[str, list] = {}
        for combo in COMBOS:
            record = run_child(
                CHILD_BODY,
                namelist_text=render_namelist(NAMELIST_TEXT, setup, combo),
                setup=setup,
                label=f"ageing setup {setup} / {combo}",
            )
            try:
                with np.load(record["npz_path"]) as d:
                    for k in d.files:
                        value = _pad(d[k], PAD_AXES[k]) if k in PAD_AXES else d[k]
                        per_combo.setdefault(k, []).append(value)
                    ncp, nchemg = int(d["ncp"]), int(d["nchemg"])
            finally:
                Path(record["npz_path"]).unlink(missing_ok=True)
            local = build_grid(ncp, nchemg)
            per_combo.setdefault("in_nd", []).append(local["nd"])
            per_combo.setdefault("in_md", []).append(_pad(local["md"], {2: NCP_MAX}))
            per_combo.setdefault("in_mdt", []).append(local["mdt"])
            per_combo.setdefault("in_ageterm1", []).append(_pad(local["ageterm1"], {2: NCHEMG_MAX}))
            per_combo.setdefault("in_ageterm2", []).append(_pad(local["ageterm2"], {3: NCP_MAX}))
        for k, v in per_combo.items():
            per_setup.setdefault(k, []).append(np.stack(v))
        if not quiet:
            print(f"  i_mode_setup = {setup}: {len(COMBOS)} combos")

    data = {k: np.stack(v) for k, v in per_setup.items()}
    grid = build_grid(NCP_MAX, NCHEMG_MAX)

    check_varied(
        {
            f"setup_{s}_{c}": {"nd": data["nd"][i, j].tolist(), "bud": data["bud"][i, j].tolist()}
            for i, s in enumerate(SETUPS)
            for j, c in enumerate(COMBOS)
        },
        expected_identical=EXPECTED_COLLISIONS,
        what="ageing setups",
    )
    report = verify_outputs(data, lambda s_i, c_i: {"nd": data["in_nd"][s_i, c_i]})

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ARCHIVE
    np.savez_compressed(
        path,
        setups=np.array(SETUPS, dtype=np.int64),
        combos=np.array(COMBOS, dtype="U16"),
        blocks=np.array(BLOCKS, dtype="U16"),
        block=grid["block"],
        in_wetdp=grid["wetdp"],
        **data,
    )
    if not quiet:
        print(f"  wrote {path.name}: nd {data['nd'].shape}")
        print("  " + ", ".join(f"{k}={v}" for k, v in report.items()))
    return path


#: Collisions that are findings about the Fortran, each with a reason.
#:
#: **Ageing is a complete no-op in four of the seven supported setups.** Setups
#: 1, 3 and 5 activate no insoluble mode at all, so the loop at `:218` has no
#: body to run. Setup 6 activates modes 6 and 7 but carries **no condensable
#: gas**, so `condensable(jv)` is false for every `jv`, `naged` stays zero and
#: the outer `naged > num_eps` guard rejects every box. All eight of those
#: (setup, combo) pairs therefore return their inputs unchanged and are
#: mutually identical.
#:
#: `l_dust_mp_ageing` reaches the routine only through `topmode`. Setups 2 and 4
#: have `mode_ait_insol` and nothing above it, which `topmode = 5` already
#: covers, so the switch is inert. **Setup 8 is the only configuration where it
#: does anything**, and its two combos are required to differ.
#:
#: Setups 2 and 8 agree at `topmode = 5`: both run mode 5 alone, and their gas
#: and mode tables are identical outside modes 6 and 7 -- the same measurement
#: `ukca_conden`'s capture records.
_NOOP = [(s, c) for s in (1, 3, 5, 6) for c in COMBOS]
EXPECTED_COLLISIONS: list[tuple[str, str]] = [
    *(
        (f"setup_{a}_{ca}", f"setup_{b}_{cb}")
        for i, (a, ca) in enumerate(_NOOP)
        for (b, cb) in _NOOP[i + 1 :]
    ),
    ("setup_2_default", "setup_2_dust_ageing"),
    ("setup_4_default", "setup_4_dust_ageing"),
    ("setup_2_default", "setup_8_default"),
    ("setup_2_dust_ageing", "setup_8_default"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        verify_source_structure()
        info = verify_target_modes()
        grid = build_grid(6, 12)
        print(f"leaf ageing sweep -> {args.out / ARCHIVE}")
        print(f"  rows        {grid['nd'].shape[0]:>6,}")
        for b in BLOCKS:
            print(f"    {b:<14}{int((grid['block'] == _block_code(b)).sum()):>6,}")
        print(f"  setups      {len(SETUPS):>6,} x {len(COMBOS)} combos")
        print(f"  mode pairs  {info['mode_pairs']:>6,} checked, none colliding")
        return 0

    print(f"sweeping leaf_ageing -> {args.out}")
    capture(args.out)
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
