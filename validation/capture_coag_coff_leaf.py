#!/usr/bin/env python3
"""Sweep `ukca_coag_coff_v` over the inputs the physics can reach (task 48).

    python validation/capture_coag_coff_leaf.py --dry-run
    python validation/capture_coag_coff_leaf.py      # writes tests/goldens/

`ukca_coag_coff_v` returns one number, `kij` in cm^3 s-1: the Brownian
coagulation coefficient between a particle of radius `ri` and one of radius
`rj`. `ukca_calc_coag_kernel` calls it once per mode for the intra-modal
`kii` and once per ordered mode pair for `kij`, and `ukca_coagwithnucl` turns
those into transfer rates.

Three supported methods, one ever run
-------------------------------------

`icoag` selects between Jacobson's full transition-regime kernel (1), the
HAM/M7 mean-radius approximation (2), and the original UM sulphate scheme (3).
Every shipped namelist runs `icoag = 1`. `icoag = 4` is refused, not captured:
`:339-340` reads `mfppi` and `mfppj`, assigned only inside the `icoag == 1`
block, so it reads memory that was never written. That is UP-5, and capturing
it would commit uninitialised memory as a reference.

Which argument reaches the output, and it is not most of them
------------------------------------------------------------

The methods do not merely weight the same inputs differently -- they read
different arguments. Counting the nine array arguments:

| argument | `icoag = 1` | `icoag = 2` | `icoag = 3` |
|---|---|---|---|
| `ri`, `rj`       | read | read | read |
| `dvisc`, `t`     | read | read | read |
| `rhoi`, `rhoj`   | read | read (via `rhomid`) | **not read** |
| `mfpa`           | read | read | **not read** |
| `vi`, `vj`       | read | **not read** (`vmid` is rebuilt from `rmid`) | **not read** |

So at `icoag = 3` five of the nine are dead. The routine's own header says
`MFP=MFPA=mean free path of air=6.6e-8*p0*T/(p*T0)` for that method, but
`:317` uses the bare `ukca_mfp_ref = 6.6e-8` PARAMETER and never touches the
`mfpa` argument at all -- so the pressure and temperature scaling the header
describes is not performed, and the caller's own mean free path is discarded.

Every "not read" above is asserted as a byte equality between two rows or two
calls differing in that argument alone, and every "read" as a byte inequality.
Only the second half can fail if a call never reached the Fortran.

The kernel is byte-symmetric, and that is worth knowing
------------------------------------------------------

Swapping `(ri, rj)`, `(vi, vj)` and `(rhoi, rhoj)` together gives the same
`kij` **bit for bit**, at all three methods. It is not obvious in advance:
`termv1` at `:280` forms `SQRT(deli*deli + delj*delj)` and `dtot` at `:279`
forms `dcoefi + dcoefj`, and floating-point addition is commutative, so the
symmetry survives rounding -- but a formulation that summed three terms, or
that divided before adding, need not have.

This matters because of what phase C found about `coag_mode`: it is symmetric
on all 64 entries, so a transposed transcription of *that* table would be
byte-equal to the correct one and neither the source parse nor the capture
could catch it. The kernel being symmetric too means the `(imode, jmode)`
order still rests on `ukca_calc_coag_kernel`'s subscripts alone -- the
kernel cannot distinguish them either. Recorded here so the next task knows
its ordering is unprotected by anything downstream.

`coag_on = 0` is a branch, not an absence
-----------------------------------------

`:239-242` returns early with `kij = 0` for **every** row, masked or not. It is
the only path on which an unmasked row comes back zero, and the box model has a
`coag_on` namelist switch, so it is a supported configuration and is swept. The
three `icoag` values collide there -- all zeros -- and that collision is
declared expected rather than discovered.

Knudsen coverage
----------------

`cci = 1.0 + kni*(1.257 + 0.4*EXP(-1.1/kni))` (`:266`) is the Cunningham slip
correction, and the only transcendental in the routine. It has two limits:
`kni -> 0` gives `cci -> 1` (continuum), `kni -> inf` gives
`cci -> 1 + 1.657*kni` (free-molecular). The grid is required to span both --
`verify_knudsen_coverage` refuses a sweep whose `mfpa/ri` does not reach below
0.01 and above 100 -- because a kernel validated only in the transition regime
says nothing about either asymptote, and the modes it is applied to span five
decades of radius.
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
ARCHIVE = "coag_coff.f64.leaf.npz"

SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_coag_coff_v.F90"
NAMELIST_TEXT = (NAMELISTS / "boundary_layer.nml").read_text(encoding="utf-8")

SETUPS: tuple[int, ...] = (1, 4)

# `pi` to the digits ukca_constants.F90:37 writes it. Used only to build the
# particle volume from the radius, which is arithmetic and touches no libm.
PI = 3.14159265358979323846

# Decimal literals, never np.logspace -- see capture_cond_coff_leaf.py.
RI_AXIS = tuple(
    float(s)
    for s in (
        "5.0e-10",
        "1.0e-9",
        "3.0e-9",
        "1.0e-8",
        "3.0e-8",
        "5.0e-8",
        "1.0e-7",
        "3.0e-7",
        "1.0e-6",
        "3.0e-6",
        "5.0e-6",
        "1.0e-5",
        "2.0e-5",
    )
)
RJ_AXIS = RI_AXIS

# 1769.0 is rho_so4 and 2650.0 is the dust density; the ends bracket what
# `rhopar` can return for a mixed particle.
RHO_AXIS = tuple(
    float(s) for s in ("800.0", "1000.0", "1200.0", "1500.0", "1769.0", "2000.0", "2650.0")
)

# 6.6e-8 is `ukca_mfp_ref`, the surface value; the mean free path of air rises
# through the stratosphere, hence the decade above it.
MFPA_AXIS = tuple(
    float(s)
    for s in ("1.0e-8", "3.0e-8", "5.0e-8", "6.6e-8", "1.0e-7", "3.0e-7", "5.0e-7", "1.0e-6")
)

DVISC_AXIS = tuple(
    float(s) for s in ("1.0e-5", "1.2e-5", "1.4e-5", "1.6e-5", "1.75e-5", "1.9e-5", "2.1e-5")
)

T_AXIS = tuple(
    float(s)
    for s in (
        "180.0",
        "200.0",
        "220.0",
        "240.0",
        "260.0",
        "273.15",
        "283.0",
        "292.0",
        "303.0",
        "313.0",
        "320.0",
    )
)

BASE_RI = float("5.0e-8")
BASE_RJ = float("1.0e-7")
BASE_RHOI = float("1500.0")
BASE_RHOJ = float("1769.0")
BASE_MFPA = float("6.6e-8")
BASE_DVISC = float("1.75e-5")
BASE_T = float("283.0")

#: `vi` is multiplied by this in the decoupled block. Four, not 1.001: if
#: `icoag = 1` ever stops reading `vi` the difference must be unmissable, and
#: if `icoag = 2` or `3` ever starts, likewise.
VI_DECOUPLE = 4.0

BLOCKS = (
    "main",
    "ri",
    "rj",
    "rhoi",
    "rhoj",
    "mfpa",
    "dvisc",
    "t",
    "vi_decoupled",
    "equal_radius",
    "swap_a",
    "swap_b",
    "masked_off",
)

# (block, the icoag values at which every row of that block gives the same kij)
CONSTANT_BLOCK_AT = (
    ("rhoi", (3,)),
    ("rhoj", (3,)),
    ("mfpa", (3,)),
)

# (block, the icoag values at which it is byte-equal to the `ri` block)
DECOUPLED_EQUAL_AT = (("vi_decoupled", (2, 3)),)

ALWAYS_VARYING = ("ri", "rj", "dvisc", "t")


def _block_code(name: str) -> int:
    return BLOCKS.index(name)


def volume(r: float) -> float:
    """The wet volume of one particle of radius `r`.

    `ukca_calc_coag_kernel.F90:245-246` passes `wvol(:,imode)` alongside
    `wetdp(:,imode)/2.0`, and `ukca_volume_mode` builds `wvol` as
    `(pi/6)*wetdp**3`. With `r = wetdp/2` that is `(4/3)*pi*r**3`, written here
    as repeated multiplication because the Fortran's `**3` is an integer
    literal exponent that gfortran expands the same way.
    """
    return (4.0 / 3.0) * PI * r * r * r


def build_grid() -> dict[str, np.ndarray]:
    """Every row the sweep runs, with the block each belongs to."""
    cols: dict[str, list] = {
        k: [] for k in ("ri", "rj", "vi", "vj", "rhoi", "rhoj", "mfpa", "dvisc", "t")
    }
    mask: list[int] = []
    block: list[int] = []

    def add(
        *,
        ri,
        rj,
        rhoi=BASE_RHOI,
        rhoj=BASE_RHOJ,
        mfpa=BASE_MFPA,
        dvisc=BASE_DVISC,
        t=BASE_T,
        vi=None,
        vj=None,
        m=1,
        blk,
    ):
        cols["ri"].append(ri)
        cols["rj"].append(rj)
        cols["vi"].append(volume(ri) if vi is None else vi)
        cols["vj"].append(volume(rj) if vj is None else vj)
        cols["rhoi"].append(rhoi)
        cols["rhoj"].append(rhoj)
        cols["mfpa"].append(mfpa)
        cols["dvisc"].append(dvisc)
        cols["t"].append(t)
        mask.append(m)
        block.append(_block_code(blk))

    for ri in RI_AXIS:
        for rj in RJ_AXIS:
            add(ri=ri, rj=rj, blk="main")

    for ri in RI_AXIS:
        add(ri=ri, rj=BASE_RJ, blk="ri")
    for rj in RJ_AXIS:
        add(ri=BASE_RI, rj=rj, blk="rj")
    for rho in RHO_AXIS:
        add(ri=BASE_RI, rj=BASE_RJ, rhoi=rho, blk="rhoi")
    for rho in RHO_AXIS:
        add(ri=BASE_RI, rj=BASE_RJ, rhoj=rho, blk="rhoj")
    for mfpa in MFPA_AXIS:
        add(ri=BASE_RI, rj=BASE_RJ, mfpa=mfpa, blk="mfpa")
    for dvisc in DVISC_AXIS:
        add(ri=BASE_RI, rj=BASE_RJ, dvisc=dvisc, blk="dvisc")
    for t in T_AXIS:
        add(ri=BASE_RI, rj=BASE_RJ, t=t, blk="t")

    # Row i pairs with row i of the `ri` block: identical but for the volumes.
    for ri in RI_AXIS:
        add(
            ri=ri,
            rj=BASE_RJ,
            vi=volume(ri) * VI_DECOUPLE,
            vj=volume(BASE_RJ) * VI_DECOUPLE,
            blk="vi_decoupled",
        )

    # ri == rj, vi == vj, rhoi == rhoj: the intra-modal `kii` case, which
    # `ukca_calc_coag_kernel.F90:249` and `:301` call for every active mode.
    for ri in RI_AXIS:
        add(ri=ri, rj=ri, rhoi=BASE_RHOI, rhoj=BASE_RHOI, blk="equal_radius")

    # Asymmetric in all three pairs, so a swap that only looked symmetric
    # because two of them happened to match cannot pass.
    swap_pairs = tuple(
        (RI_AXIS[a], RI_AXIS[b]) for a, b in ((0, 6), (1, 8), (2, 11), (3, 5), (4, 12), (6, 9))
    )
    for ri, rj in swap_pairs:
        add(ri=ri, rj=rj, rhoi=1200.0, rhoj=2650.0, blk="swap_a")
    for ri, rj in swap_pairs:
        add(ri=rj, rj=ri, rhoi=2650.0, rhoj=1200.0, vi=volume(rj), vj=volume(ri), blk="swap_b")

    poison = (0.0, float("inf"), float("-inf"), float("nan"), float("1.0e308"), float("5.0e-324"))
    for p in poison:
        add(ri=p, rj=p, rhoi=p, rhoj=p, mfpa=p, dvisc=p, t=p, vi=p, vj=p, m=0, blk="masked_off")
    add(ri=BASE_RI, rj=BASE_RJ, m=0, blk="masked_off")

    out = {k: np.array(v, dtype=np.float64) for k, v in cols.items()}
    out["mask"] = np.array(mask, dtype=np.int32)
    out["block"] = np.array(block, dtype=np.int32)
    return out


ICOAGS: tuple[int, ...] = (1, 2, 3)
COAG_ONS: tuple[int, ...] = (1, 0)


def build_calls() -> list[dict]:
    return [{"icoag": ic, "coag_on": co} for co in COAG_ONS for ic in ICOAGS]


def call_label(c: dict) -> str:
    return f"icoag={c['icoag']}|coag_on={c['coag_on']}"


CHILD_BODY = (
    "\nimport tempfile\n"
    f"sys.path.insert(0, {str(REPO / 'validation')!r})\n"
    "from capture_coag_coff_leaf import build_grid, build_calls, call_label\n"
    "from leaf_common import bind_call\n"
    "\n"
    "call = bind_call(g)\n"
    "grid = build_grid()\n"
    "calls = build_calls()\n"
    "rows = []\n"
    "for _c in calls:\n"
    "    _kij, _ierr = call(call_label(_c), g.leaf_coag_coff,\n"
    "                       grid['mask'], grid['ri'], grid['rj'], grid['vi'],\n"
    "                       grid['vj'], grid['rhoi'], grid['rhoj'], grid['mfpa'],\n"
    "                       grid['dvisc'], grid['t'], _c['coag_on'], _c['icoag'])\n"
    "    rows.append(np.asarray(_kij))\n"
    "\n"
    "_tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)\n"
    "_tmp.close()\n"
    "np.savez(_tmp.name, kij=np.stack(rows))\n"
    "print('@@RESULT@@' + json.dumps({'npz_path': _tmp.name, 'setup': _setup}))\n"
)

_CHILD = CHILD_PREAMBLE.format(f2py=str(F2PY_DIR)) + CHILD_BODY


# --- verification, all of it before np.savez_compressed ---------------------


def verify_knudsen_coverage(grid: dict[str, np.ndarray]) -> tuple[float, float]:
    """The grid must reach both limits of the Cunningham correction.

    `cci = 1.0 + kni*(1.257 + 0.4*EXP(-1.1/kni))` (`:266`) tends to `1` as
    `kni -> 0` and to `1 + 1.657*kni` as `kni -> inf`. A sweep confined to the
    transition regime validates a kernel that is applied across five decades of
    radius and says nothing about either asymptote.
    """
    live = grid["mask"] == 1
    kn = np.concatenate(
        [grid["mfpa"][live] / grid["ri"][live], grid["mfpa"][live] / grid["rj"][live]]
    )
    lo, hi = float(kn.min()), float(kn.max())
    if lo > 0.01 or hi < 100.0:
        raise SystemExit(
            f"mfpa/r spans [{lo:.3g}, {hi:.3g}]; the continuum limit needs < 0.01 "
            "and the free-molecular limit > 100"
        )
    return lo, hi


def verify_source_literals() -> dict[str, int]:
    """Re-read the Fortran for every literal this capture's reasoning rests on."""
    text = SOURCE.read_text(encoding="utf-8")
    squeezed = text.replace(" ", "")
    required = {
        "ukca_mfp_ref=6.6e-8": "ukca_mfp_ref=6.6e-8" in squeezed,
        "cunningham 1.257/0.4/1.1": "1.257" in text and "0.4*EXP(-1.1/" in squeezed,
        "icoag=3 uses ukca_mfp_ref": "1.591*ukca_mfp_ref" in squeezed,
        "icoag=3 does not read mfpa": (
            "termv4(:)=1.591*ukca_mfp_ref*(1.0/ri(:)/ri(:)+1.0/rj(:)/rj(:))" in squeezed
        ),
        "icoag=4 reads mfppi": "mfppi(:)*cci(:)/ri(:)/ri(:)" in squeezed,
        "coag_on early return": "IF(coag_on==0)THEN" in squeezed,
        "header says 1.59": "assumed = 1.59 for all modes" in text,
    }
    missing = sorted(k for k, ok in required.items() if not ok)
    if missing:
        raise SystemExit(
            "the vendored source no longer contains: "
            + ", ".join(missing)
            + " -- this capture's docstring describes a routine that has changed"
        )
    return {"checked": len(required)}


def _idx(grid: dict[str, np.ndarray], block: str) -> np.ndarray:
    return np.flatnonzero(grid["block"] == _block_code(block))


def _find(calls: list[dict], icoag: int, coag_on: int) -> int:
    for i, c in enumerate(calls):
        if c["icoag"] == icoag and c["coag_on"] == coag_on:
            return i
    raise SystemExit(f"no call icoag={icoag}, coag_on={coag_on}")


def verify_outputs(kij: np.ndarray, grid: dict, calls: list[dict]) -> dict:
    """Everything the golden claims, checked against the golden itself."""
    nsetup, ncall, nrow = kij.shape
    want = (len(SETUPS), len(calls), grid["ri"].size)
    if (nsetup, ncall, nrow) != want:
        raise SystemExit(f"kij has shape {kij.shape}, expected {want}")

    report: dict[str, float] = {}

    off = _idx(grid, "masked_off")
    on = np.flatnonzero(grid["mask"] == 1)

    # 1. The mask is a select, not a multiply -- masked rows carry inf and NaN.
    bad = np.flatnonzero(kij[:, :, off] != 0.0)
    if bad.size:
        raise SystemExit(f"kij is non-zero on {bad.size} masked-off entries")
    report["masked_rows"] = int(off.size)

    # 2. coag_on = 0 returns early at `:239-242`, BEFORE the mask is consulted,
    #    so every row is zero -- the only path on which an unmasked row is.
    for i, c in enumerate(calls):
        live = kij[:, i, on]
        if c["coag_on"] == 0:
            if np.any(live != 0.0):
                raise SystemExit(
                    f"{call_label(c)}: coag_on=0 left {int((live != 0.0).sum())} rows non-zero"
                )
            continue
        if not np.all(np.isfinite(live)):
            n_bad = int((~np.isfinite(live)).sum())
            raise SystemExit(f"{call_label(c)}: kij is not finite on {n_bad} live entries")
        if not np.all(live > 0.0):
            raise SystemExit(
                f"{call_label(c)}: kij is not positive on {int((live <= 0.0).sum())} entries"
            )
    report["live_rows"] = int(on.size)

    # 3. Setup independence, measured rather than argued.
    if not np.array_equal(kij[0], kij[1]):
        raise SystemExit(
            f"i_mode_setup {SETUPS[0]} and {SETUPS[1]} disagree -- ukca_coag_coff_v "
            "reads a per-setup table after all"
        )

    live_calls = [i for i, c in enumerate(calls) if c["coag_on"] == 1]

    # 4. A block whose only varying argument is unread must be constant, and
    #    must not be at the methods that do read it.
    for block, const_icoags in CONSTANT_BLOCK_AT:
        rows = _idx(grid, block)
        for i in live_calls:
            vals = kij[0, i, rows]
            constant = bool(np.all(vals == vals[0]))
            want_const = calls[i]["icoag"] in const_icoags
            if constant != want_const:
                raise SystemExit(
                    f"{block} block under {call_label(calls[i])}: kij is "
                    f"{'constant' if constant else 'varying'}, expected "
                    f"{'constant' if want_const else 'varying'}"
                )

    # 5. `vi` and `vj` are read only by icoag = 1. The decoupled block is the
    #    `ri` block with both volumes multiplied by four and nothing else
    #    changed, so equality is exactly "the argument is dead".
    ri_rows = _idx(grid, "ri")
    for block, equal_icoags in DECOUPLED_EQUAL_AT:
        rows = _idx(grid, block)
        if rows.size != ri_rows.size:
            raise SystemExit(f"the {block} and ri blocks are not the same length")
        for i in live_calls:
            same = np.array_equal(kij[0, i, rows], kij[0, i, ri_rows])
            want_same = calls[i]["icoag"] in equal_icoags
            if same != want_same:
                raise SystemExit(
                    f"{block} under {call_label(calls[i])}: rows "
                    f"{'match' if same else 'differ from'} their consistent partners, "
                    f"expected to {'match' if want_same else 'differ'}"
                )

    # 6. Arguments read by every method.
    for block in ALWAYS_VARYING:
        rows = _idx(grid, block)
        for i in live_calls:
            vals = kij[0, i, rows]
            if np.all(vals == vals[0]):
                raise SystemExit(f"{block} block is constant under {call_label(calls[i])}")

    # 7. The kernel is byte-symmetric under a simultaneous swap of the three
    #    pairs. Not obvious in advance: it survives because every combination
    #    of i and j is an addition, and addition is commutative.
    a_rows, b_rows = _idx(grid, "swap_a"), _idx(grid, "swap_b")
    if a_rows.size != b_rows.size:
        raise SystemExit("the swap_a and swap_b blocks are not the same length")
    for i in live_calls:
        if not np.array_equal(kij[0, i, a_rows], kij[0, i, b_rows]):
            n = int((kij[0, i, a_rows] != kij[0, i, b_rows]).sum())
            raise SystemExit(
                f"{call_label(calls[i])}: kij is not symmetric under (i,j) swap on "
                f"{n} of {a_rows.size} pairs"
            )
    report["swap_pairs"] = int(a_rows.size)

    # 8. The three methods must not agree with each other, or the switch is
    #    not reaching the Fortran.
    for a in range(len(live_calls)):
        for b in range(a + 1, len(live_calls)):
            ia, ib = live_calls[a], live_calls[b]
            if np.array_equal(kij[0, ia], kij[0, ib]):
                raise SystemExit(
                    f"{call_label(calls[ia])} and {call_label(calls[ib])} are identical"
                )

    lo, hi = verify_knudsen_coverage(grid)
    report["kn_min"], report["kn_max"] = lo, hi
    return report


def capture(out_dir: Path, quiet: bool = False) -> Path:
    verify_source_literals()
    grid = build_grid()
    verify_knudsen_coverage(grid)
    calls = build_calls()

    by_setup, paths = [], []
    try:
        for setup in SETUPS:
            record = run_child(
                CHILD_BODY,
                namelist_text=render_namelist(NAMELIST_TEXT, setup, "default"),
                setup=setup,
                label=f"coag_coff setup {setup}",
            )
            paths.append(record["npz_path"])
            with np.load(record["npz_path"]) as data:
                by_setup.append(data["kij"])
            if not quiet:
                print(f"  i_mode_setup = {setup}: {len(calls)} calls x {grid['ri'].size} rows")
    finally:
        for p in paths:
            Path(p).unlink(missing_ok=True)

    kij = np.stack(by_setup)

    # The three coag_on = 0 calls return all zeros and therefore collide with
    # each other. Declared, not discovered: an unexpected collision means a
    # call never reached the Fortran, and a missing expected one means the
    # early return has changed.
    off_labels = [call_label(c) for c in calls if c["coag_on"] == 0]
    check_varied(
        {call_label(c): {"kij": kij[0, i].tolist()} for i, c in enumerate(calls)},
        expected_identical=[
            (off_labels[a], off_labels[b])
            for a in range(len(off_labels))
            for b in range(a + 1, len(off_labels))
        ],
        what="coag_coff configurations",
    )

    report = verify_outputs(kij, grid, calls)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ARCHIVE
    np.savez_compressed(
        path,
        kij=kij,
        setups=np.array(SETUPS, dtype=np.int64),
        blocks=np.array(BLOCKS, dtype="U16"),
        block=grid["block"],
        mask=grid["mask"],
        **{k: grid[k] for k in ("ri", "rj", "vi", "vj", "rhoi", "rhoj", "mfpa", "dvisc", "t")},
        call_icoag=np.array([c["icoag"] for c in calls], dtype=np.int64),
        call_coag_on=np.array([c["coag_on"] for c in calls], dtype=np.int64),
        _kn_min=np.float64(report["kn_min"]),
        _kn_max=np.float64(report["kn_max"]),
    )
    if not quiet:
        print(f"  wrote {path.name}: {kij.shape} kij, {report['live_rows']} live rows per call")
        print(f"  mfpa/r spans [{report['kn_min']:.3g}, {report['kn_max']:.3g}]")
        print(f"  symmetric on {report['swap_pairs']} swapped pairs, all methods")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="print the grids and stop")
    args = parser.parse_args(argv)

    grid = build_grid()
    calls = build_calls()

    if args.dry_run:
        verify_source_literals()
        lo, hi = verify_knudsen_coverage(grid)
        print(f"leaf coag_coff sweep -> {args.out / ARCHIVE}")
        print(f"  rows        {grid['ri'].size:>6,}  ({int((grid['mask'] == 0).sum())} masked off)")
        for b in BLOCKS:
            print(f"    {b:<18}{int((grid['block'] == _block_code(b)).sum()):>6,}")
        print(f"  calls       {len(calls):>6,}  = {len(ICOAGS)} methods x {len(COAG_ONS)} coag_on")
        print(f"  setups      {len(SETUPS):>6,}  {SETUPS}")
        print(f"  mfpa/r spans [{lo:.3g}, {hi:.3g}]")
        return 0

    print(f"sweeping leaf_coag_coff -> {args.out}")
    capture(args.out)
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
