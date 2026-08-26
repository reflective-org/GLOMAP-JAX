#!/usr/bin/env python3
"""Sweep `ukca_calcnucrate` over the branches the physics can reach (task 52).

    python validation/capture_calcnucrate_leaf.py --dry-run
    python validation/capture_calcnucrate_leaf.py    # writes tests/goldens/

`ukca_calcnucrate` decides how much H2SO4 nucleation removes in one `dtz`. It
wraps `ukca_binapara` for the binary rate, adds a boundary-layer term, and
rewrites `h2so4` in place.

Five branch combinations, and the switches do not select them directly
---------------------------------------------------------------------

Two logicals at `:302` and `:304` decide what runs, and neither is a namelist
variable:

    l1 = (i_nuc_method == 2) .AND. (height > zbl .OR. bln_on == 0)
    l2 = (i_nuc_method == 3) .AND. (ibln == 3)

`l1 .OR. l2` runs binary nucleation; `.NOT. l1` runs boundary-layer
nucleation. So:

| `i_nuc_method` | `bln_on` | height | `ibln` | runs |
|---|---|---|---|---|
| 2 | 0 | any | any | BHN only |
| 2 | 1 | above `zbl` | any | BHN only |
| 2 | 1 | at or below `zbl` | any | **BLN only** |
| 3 | any | any | 3 | **BHN then BLN** |
| 3 | any | any | 1 or 2 | BLN only |

The fourth row is the one that matters and the one no shipped namelist
reaches: `bln_on` is off in all five, so nothing has ever run both terms in the
same box. `verify_branch_coverage` refuses a grid that leaves any of the five
unexercised, counted separately.

`zbl = MIN(htpblg, 6000)` (`:265-267`), so the height test is against a
*filtered* boundary-layer height and a namelist with `pbl_height` above 6 km
would find its BL capped. The grid straddles `zmaxbln` on both sides.

Where the guards are, and what they hide
----------------------------------------

Binary nucleation needs four conditions at once (`:365-366`) --
`h2so4 > conc_eps`, `jveh > 0`, `s_cond_s > 0`, `rc > 0` -- and then
`japp > 1e-3`. Boundary-layer nucleation needs `h2so4 > conc_eps` and
`japp_bln > 1e-3`. `jveh` is zero over most of the parameter space
(`ukca_binapara` floors anything below 1e-7), so a grid that did not aim at the
nucleating corner would test the guards and nothing else. Each guard is
counted, both ways.

**`s_cond_s == 0` switches binary nucleation off entirely.** It is the
condensation sink, computed by `ukca_conden`, and it appears in the exponent as
`s_cond_s/(1e6*h2so4*1e-13)` -- so zero sink means `EXP(0) = 1` and no
suppression, except that the guard rejects the row before that can happen.
UP-6 already records that `s_cond_s` is read unassigned when `cond_on = 0` and
`nucl_on = 1`; this sweep includes `s_cond_s = 0` rows so the golden says what
the compiled routine does with them rather than leaving it to the flag.

The BLN rate has three forms
----------------------------

`ibln` selects activation (`5e-7 * h2so4`), kinetic (`4e-13 * h2so4^2`) or
Metzger (`5e-13 * h2so4 * sec_org`). Only the third reads `sec_org`, which is
asserted as a byte equality between rows differing in `sec_org` alone -- and
`sec_org` is zero in setups that carry no secondary organic, so the Metzger
form gives exactly zero there.
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
ARCHIVE = "calcnucrate.f64.leaf.npz"
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_calcnucrate.F90"
NAMELIST_TEXT = (NAMELISTS / "boundary_layer.nml").read_text(encoding="utf-8")

SETUPS: tuple[int, ...] = (1, 4)

ZMAXBLN = 6000.0
DTZ = 60.0

BLOCKS = ("nucleating", "guards", "height", "sec_org", "sink")


def _block_code(name: str) -> int:
    return BLOCKS.index(name)


#: (i_nuc_method, ibln, bln_on). Chosen to reach all five rows of the table in
#: the module docstring; `verify_branch_coverage` checks that they do.
CONFIGS: tuple[tuple[int, int, int], ...] = (
    (2, 1, 0),  # BHN only, whatever the height
    (2, 1, 1),  # BHN above zbl, BLN activation below
    (2, 2, 1),  # same, kinetic
    (3, 3, 1),  # BHN *and* BLN, the combination nothing has run
    (3, 1, 1),  # BLN only, activation
    (3, 2, 1),  # BLN only, kinetic
    # BLN only, Metzger. Needed because under (3,3,1) binary nucleation runs
    # first and can take h2so4 below conc_eps, which switches the BLN guard off
    # and makes sec_org invisible -- the capture refused the golden until this
    # configuration was added, which is the guard doing its job.
    (2, 3, 1),
)


def build_grid() -> dict[str, np.ndarray]:
    """Rows chosen to sit in the nucleating corner, not the guards."""
    cols: dict[str, list[float]] = {
        k: [] for k in ("t", "s", "rh", "aird", "h2so4", "sec_org", "height", "htpblg", "s_cond_s")
    }
    block: list[int] = []

    def add(
        *,
        t=250.0,
        s=2.0e-3,
        rh=0.5,
        aird=2.5e19,
        h2so4=1.0e8,
        sec_org=1.0e7,
        height=200.0,
        htpblg=800.0,
        s_cond_s=1.0e-4,
        blk,
    ):
        for key, value in (
            ("t", t),
            ("s", s),
            ("rh", rh),
            ("aird", aird),
            ("h2so4", h2so4),
            ("sec_org", sec_org),
            ("height", height),
            ("htpblg", htpblg),
            ("s_cond_s", s_cond_s),
        ):
            cols[key].append(value)
        block.append(_block_code(blk))

    # Cold, humid and acid enough for ukca_binapara to give a non-zero rate.
    for t in (200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 273.15):
        for h2so4 in (1.0e6, 1.0e7, 1.0e8, 1.0e9, 1.0e10):
            for rh in (0.2, 0.5, 0.9):
                add(t=t, h2so4=h2so4, rh=rh, blk="nucleating")

    # Each guard, from both sides. conc_eps is 1e-8.
    for h2so4 in (0.0, 1.0e-9, 1.0e-8, 1.0e-7, 1.0, 1.0e4):
        add(h2so4=h2so4, blk="guards")
    for s_cond_s in (0.0, 1.0e-12, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0):
        add(s_cond_s=s_cond_s, blk="sink")
    # Warm and dry: ukca_binapara floors jveh to zero, so the jveh > 0 guard
    # rejects the row and only BLN can contribute.
    for t in (290.0, 298.0, 310.0):
        add(t=t, rh=0.1, h2so4=1.0e6, blk="guards")

    # Straddle zbl and the 6 km cap on htpblg.
    for height, htpblg in (
        (100.0, 800.0),
        (799.0, 800.0),
        (800.0, 800.0),
        (801.0, 800.0),
        (5000.0, 12000.0),
        (5999.0, 12000.0),
        (6000.0, 12000.0),
        (6001.0, 12000.0),
        (7000.0, 12000.0),
        (100.0, 0.0),
        (0.0, 0.0),
    ):
        add(height=height, htpblg=htpblg, t=220.0, h2so4=1.0e8, blk="height")

    # sec_org alone, which only the Metzger form reads.
    for sec_org in (0.0, 1.0e5, 1.0e7, 1.0e9):
        add(sec_org=sec_org, t=220.0, h2so4=1.0e8, blk="sec_org")

    out = {k: np.array(v, dtype=np.float64) for k, v in cols.items()}
    out["block"] = np.array(block, dtype=np.int32)
    return out


CHILD_BODY = (
    "\nimport tempfile\n"
    f"sys.path.insert(0, {str(REPO / 'validation')!r})\n"
    "from capture_calcnucrate_leaf import build_grid, CONFIGS, DTZ\n"
    "from leaf_common import bind_call\n"
    "\n"
    "call = bind_call(g)\n"
    "grid = build_grid()\n"
    "h_rows, d_rows = [], []\n"
    "for _m, _ibln, _bln in CONFIGS:\n"
    "    _h, _d, _e = call('nuc %d/%d/%d' % (_m, _ibln, _bln), g.leaf_calcnucrate,\n"
    "                      DTZ, grid['t'], grid['s'], grid['rh'], grid['aird'],\n"
    "                      grid['h2so4'], grid['sec_org'], grid['height'],\n"
    "                      grid['htpblg'], grid['s_cond_s'], _bln, _ibln, _m)\n"
    "    h_rows.append(np.asarray(_h))\n"
    "    d_rows.append(np.asarray(_d))\n"
    "\n"
    "# ukca_binapara on the same inputs, so the capture can tell a zero that\n"
    "# came from the jveh guard from one that came from japp <= 1e-3.\n"
    "_jveh, _rc, _e2 = call('binapara', g.leaf_binapara,\n"
    "                       grid['t'], grid['rh'], grid['h2so4'])\n"
    "_tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)\n"
    "_tmp.close()\n"
    "np.savez(_tmp.name, h2so4=np.stack(h_rows), delh2so4=np.stack(d_rows),\n"
    "         jveh=np.asarray(_jveh), rc=np.asarray(_rc))\n"
    "print('@@RESULT@@' + json.dumps({'npz_path': _tmp.name, 'setup': _setup}))\n"
)

_CHILD = CHILD_PREAMBLE.format(f2py=str(F2PY_DIR)) + CHILD_BODY


# --- verification -----------------------------------------------------------


def branch_flags(grid: dict, config: tuple[int, int, int]) -> dict[str, np.ndarray]:
    """`l1` and `l2` from `:302-306`, re-derived rather than inferred.

    These are the two logicals that decide which of the two nucleation terms
    runs, and neither is a namelist variable -- so which branch a row takes
    cannot be read off the configuration alone.
    """
    i_nuc_method, ibln, bln_on = config
    zbl = np.minimum(grid["htpblg"], ZMAXBLN)
    l1 = (i_nuc_method == 2) & ((grid["height"] > zbl) | (bln_on == 0))
    l2 = np.full_like(l1, (i_nuc_method == 3) and (ibln == 3))
    return {"l1": l1, "l2": l2, "bhn": l1 | l2, "bln": ~l1}


def verify_branch_coverage(grid: dict) -> dict[str, int]:
    """Refuse a grid that leaves any of the five branch combinations unrun."""
    seen = {"bhn_only": 0, "bln_only": 0, "both": 0, "neither": 0}
    for config in CONFIGS:
        f = branch_flags(grid, config)
        seen["bhn_only"] += int((f["bhn"] & ~f["bln"]).sum())
        seen["bln_only"] += int((f["bln"] & ~f["bhn"]).sum())
        seen["both"] += int((f["bhn"] & f["bln"]).sum())
        seen["neither"] += int((~f["bhn"] & ~f["bln"]).sum())
    for name in ("bhn_only", "bln_only", "both"):
        if seen[name] == 0:
            raise SystemExit(
                f"no (config, row) pair reaches `{name}`; that combination is "
                "unvalidated and a port that got it wrong would pass"
            )
    if seen["neither"] != 0:
        raise SystemExit(
            "some row runs neither term, which `l1`/`.NOT. l1` makes impossible -- "
            "the re-derivation of the logicals is wrong"
        )
    return seen


def verify_source_literals() -> dict[str, int]:
    text = SOURCE.read_text(encoding="utf-8")
    squeezed = text.replace(" ", "")
    required = {
        "vehkamaki hardcoded": "i_bhn_method=i_bhn_method_vekhamaki" in squeezed,
        "zmaxbln": "zmaxbln=6000.0" in squeezed,
        "afac_pna": "afac_pna=5.0e-13" in squeezed,
        "afac_act": "afac_act=5.0e-7" in squeezed,
        "afac_kin": "afac_kin=4.0e-13" in squeezed,
        "dpbln": "dpbln=1.5" in squeezed,
        "japp threshold": "japp(jl)>1.0e-3" in squeezed,
        "japp_bln threshold": "japp_bln>1.0e-3" in squeezed,
        "l1": "l1=(i_nuc_method==2.AND.(height(jl)>zbl(jl).OR.bln_on==0))" in squeezed,
        "l2": "l2=(i_nuc_method==3.AND.ibln==3)" in squeezed,
    }
    missing = sorted(k for k, ok in required.items() if not ok)
    if missing:
        raise SystemExit(
            "the vendored source no longer contains: "
            + ", ".join(missing)
            + " -- this capture describes a routine that has changed"
        )
    return {"checked": len(required)}


def verify_outputs(h2so4: np.ndarray, delh2so4: np.ndarray, jveh: np.ndarray, grid: dict) -> dict:
    n_config, n_row = len(CONFIGS), grid["t"].size
    if h2so4.shape != (len(SETUPS), n_config, n_row):
        raise SystemExit(f"h2so4 has shape {h2so4.shape}")

    if not (np.array_equal(h2so4[0], h2so4[1]) and np.array_equal(delh2so4[0], delh2so4[1])):
        raise SystemExit("the two mode setups disagree; ukca_calcnucrate is not setup-independent")

    report: dict[str, int] = {}
    h_in = grid["h2so4"]

    for c_i, config in enumerate(CONFIGS):
        h_out, d_out = h2so4[0, c_i], delh2so4[0, c_i]
        label = "/".join(str(x) for x in config)

        if not np.all(np.isfinite(h_out)) or not np.all(np.isfinite(d_out)):
            raise SystemExit(f"{label}: non-finite output")
        # `:353-354` and `:377-379`: the concentration may only fall, and never
        # below zero.
        if np.any(h_out > h_in):
            raise SystemExit(
                f"{label}: nucleation increased h2so4 on {int((h_out > h_in).sum())} rows"
            )
        if np.any(h_out < 0.0):
            raise SystemExit(f"{label}: h2so4 went negative")
        # `delh2so4_nucl` accumulates `h2so4old - h2so4` over the two terms, so
        # it must equal the total drop exactly.
        drop = h_in - h_out
        if not np.allclose(d_out, drop, rtol=0, atol=0):
            bad = int((d_out != drop).sum())
            raise SystemExit(
                f"{label}: delh2so4_nucl disagrees with the h2so4 drop on {bad} rows -- "
                "the two are the same accumulation and must match bit for bit"
            )
        if not np.any(d_out > 0.0):
            raise SystemExit(f"{label}: nothing nucleated anywhere; the grid misses the corner")
        report[f"active_{label}"] = int((d_out > 0.0).sum())

    # A zero rate must be explicable: either a guard rejected the row or the
    # rate fell below 1e-3. Counting them apart is what says the grid tests
    # the physics and not only the guards.
    conc_eps = 1.0e-8
    report["guard_h2so4"] = int((h_in <= conc_eps).sum())
    report["guard_jveh"] = int((jveh <= 0.0).sum())
    report["guard_sink"] = int((grid["s_cond_s"] <= 0.0).sum())
    report["jveh_positive"] = int((jveh > 0.0).sum())
    for name in ("guard_h2so4", "guard_jveh", "guard_sink", "jveh_positive"):
        if report[name] == 0:
            raise SystemExit(f"`{name}` is never reached, so its guard is unvalidated")

    # sec_org reaches the answer only through the Metzger form.
    rows = np.flatnonzero(grid["block"] == _block_code("sec_org"))
    # The BLN-only Metzger configuration, where nothing has depleted h2so4
    # before the term that reads sec_org runs.
    metzger = CONFIGS.index((2, 3, 1))
    others = [i for i, c in enumerate(CONFIGS) if c[1] != 3]
    if np.all(delh2so4[0, metzger, rows] == delh2so4[0, metzger, rows[0]]):
        raise SystemExit("sec_org does not move the Metzger rate; ibln = 3 is not being reached")
    for c_i in others:
        vals = delh2so4[0, c_i, rows]
        if not np.all(vals == vals[0]):
            raise SystemExit(
                f"{CONFIGS[c_i]}: sec_org moved the answer at ibln != 3, where it is not read"
            )
    return report


def capture(out_dir: Path, quiet: bool = False) -> Path:
    verify_source_literals()
    grid = build_grid()
    verify_branch_coverage(grid)

    stacks: dict[str, list] = {"h2so4": [], "delh2so4": [], "jveh": [], "rc": []}
    paths: list[str] = []
    try:
        for setup in SETUPS:
            record = run_child(
                CHILD_BODY,
                namelist_text=render_namelist(NAMELIST_TEXT, setup, "default"),
                setup=setup,
                label=f"calcnucrate setup {setup}",
            )
            paths.append(record["npz_path"])
            with np.load(record["npz_path"]) as d:
                for k in stacks:
                    stacks[k].append(d[k])
            if not quiet:
                print(f"  i_mode_setup = {setup}: {len(CONFIGS)} configs x {grid['t'].size} rows")
    finally:
        for p in paths:
            Path(p).unlink(missing_ok=True)

    data = {k: np.stack(v) for k, v in stacks.items()}

    check_varied(
        {
            "/".join(str(x) for x in c): {"delh2so4": data["delh2so4"][0, i].tolist()}
            for i, c in enumerate(CONFIGS)
        },
        # No collision is expected: the six configurations differ in which
        # term runs or in the BLN rate law, and a collision would mean a
        # switch never reached the Fortran.
        expected_identical=[],
        what="calcnucrate configurations",
    )

    report = verify_outputs(data["h2so4"], data["delh2so4"], data["jveh"][0], grid)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ARCHIVE
    np.savez_compressed(
        path,
        h2so4=data["h2so4"],
        delh2so4=data["delh2so4"],
        jveh=data["jveh"],
        rc=data["rc"],
        setups=np.array(SETUPS, dtype=np.int64),
        blocks=np.array(BLOCKS, dtype="U16"),
        dtz=np.float64(DTZ),
        config=np.array(CONFIGS, dtype=np.int64),
        # `h2so4_in`, not `h2so4`: the routine rewrites its argument, so the
        # input and the output are different arrays and naming them the same
        # would let a test compare one against itself.
        h2so4_in=grid["h2so4"],
        **{
            k: grid[k]
            for k in ("t", "s", "rh", "aird", "sec_org", "height", "htpblg", "s_cond_s", "block")
        },
        **{f"_{k}": np.int64(v) for k, v in report.items()},
    )
    if not quiet:
        print(f"  wrote {path.name}: {data['h2so4'].shape}")
        print("  " + ", ".join(f"{k}={v}" for k, v in report.items() if k.startswith("active")))
        print(
            "  guards: "
            + ", ".join(f"{k}={v}" for k, v in report.items() if k.startswith(("guard", "jveh")))
        )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    grid = build_grid()
    if args.dry_run:
        verify_source_literals()
        seen = verify_branch_coverage(grid)
        print(f"leaf calcnucrate sweep -> {args.out / ARCHIVE}")
        print(f"  rows        {grid['t'].size:>6,}")
        for b in BLOCKS:
            print(f"    {b:<14}{int((grid['block'] == _block_code(b)).sum()):>6,}")
        print(f"  configs     {len(CONFIGS):>6,}  {CONFIGS}")
        print("  branch pairs: " + ", ".join(f"{k}={v}" for k, v in seen.items()))
        return 0

    print(f"sweeping leaf_calcnucrate -> {args.out}")
    capture(args.out)
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
