#!/usr/bin/env python3
"""Sweep `ukca_calc_coag_kernel` over all seven supported setups (task 50).

    python validation/capture_coag_kernel_leaf.py --dry-run
    python validation/capture_coag_kernel_leaf.py     # writes tests/goldens/

`ukca_calc_coag_kernel` is a driver, not a formula. It calls
`ukca_coag_coff_v` once per active mode for the intra-modal `kii_arr` and once
per ordered mode pair for `kij_arr`, and every number it returns was already
pinned by task 48. What only this routine can settle is two things a value
comparison cannot.

**1. The slot map, which no value can check.** The kernel is byte-symmetric
under an `(i, j)` swap (task 48), and `coag_mode` is symmetric on all 64
entries (phase C). So a transposed transcription writes the *right number into
the wrong slot* and every value test passes. The only discriminator is which
entries are non-zero and which are left at the `0.0` of `:239-247`:

* `:243-262` fills `(imode, jmode)` for soluble `imode` and larger soluble
  `jmode` -- strictly upper-triangular within 1..4;
* `:276-291` fills `(imode, jmode)` for soluble `imode` and insoluble
  `jmode >= imode+4` -- upper-triangular across the 1..4 / 5..8 split;
* `:308-322` fills `(imode, jmode)` for **insoluble** `imode` and *smaller*
  soluble `jmode >= imode-2` -- LOWER-triangular.

So `kij_arr` is not triangular in either direction, and the two families are
disjoint but adjacent: `(1,5)` is written by the first family and `(5,1)`,
`(5,2)`, `(5,3)` by the third. Transpose the convention and `(5,1)` collides
with a slot the other family owns. The expected set is re-derived here from
`mode` and `modesol` and compared with what the compiled routine produced; if
the two disagree, this file has misread the loops.

**2. Which size the driver passes for an insoluble partner, and it is not
consistent.** In the soluble-`imode` loop, `:281-286` selects by `modesol`:

    IF (modesol(jmode) == 1) THEN rpj = wetdp/2, vpj = wvol
    ELSE                          rpj = drydp/2, vpj = dvol

so an insoluble `jmode` enters at its **dry** size. In the insoluble-`imode`
loop, `:297-299` and `:313-315` take `wetdp`/`wvol` unconditionally -- so the
same insoluble mode enters at its **wet** size, both for its own `kii` and for
its coagulation with smaller soluble modes.

Whether that matters in the model depends on whether `ukca_volume_mode` returns
`wetdp == drydp` for an insoluble mode, which is a separate question with its
own answer. It is not assumed here either way: the grid feeds
`drydp = 0.7*wetdp` for **every** mode, so the two paths give different
numbers, and the capture calls `ukca_coag_coff_v` directly under both
conventions and records which one each slot actually matches. The finding is a
measurement, not a reading of the source.

Why the same grid across all setups
-----------------------------------

The per-mode sizes and densities do not depend on `i_mode_setup`, so the only
thing that varies between the seven captures is which modes are active. That is
what makes the slot map comparable: a difference between setups is a difference
in `mode`, and nothing else.

Every mode carries a distinct diameter and a distinct density, strictly
increasing across the eight slots. That is deliberate -- with equal sizes,
`kij(i,j)` would equal `kij(j,i)` as *values* and a slot mix-up inside one
family would be invisible as well as one between families.
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
ARCHIVE = "coag_kernel.f64.leaf.npz"

SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_calc_coag_kernel.F90"
NAMELIST_TEXT = (NAMELISTS / "boundary_layer.nml").read_text(encoding="utf-8")

#: The setups `docs/harness.md` records as supported. 7 and 9-13 are not.
SETUPS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8)

NMODES = 8
PI = 3.14159265358979323846

# Fortran 1-based mode indices, as ukca_mode_setup.F90:86-93 declares them.
MODE_NUC_SOL, MODE_COR_SOL = 1, 4
MODE_AIT_INSOL, MODE_COR_INSOL, MODE_SUP_INSOL = 5, 7, 8

#: Wet diameter per slot, strictly increasing and distinct so that a slot
#: mix-up cannot be masked by two modes sharing a value. Spans the nucleation
#: mode to the super-coarse insoluble mode.
WETDP = tuple(
    float(s)
    for s in ("4.0e-9", "3.0e-8", "2.0e-7", "2.0e-6", "5.0e-8", "3.0e-7", "3.0e-6", "1.0e-5")
)

#: Distinct per slot for the same reason. Real values would come from
#: `rhopar`, which is a mixture density and varies by composition.
RHOPAR = tuple(
    float(s)
    for s in ("1500.0", "1600.0", "1700.0", "1800.0", "1900.0", "2000.0", "2100.0", "2200.0")
)

#: `drydp = DRY_FRACTION * wetdp` for EVERY mode, insoluble included. The
#: model would give an insoluble mode no water and hence `drydp == wetdp`; that
#: is exactly the coincidence that would hide which size the driver passes, so
#: the fixture refuses it.
DRY_FRACTION = 0.7

BASE_MFPA = float("6.6e-8")
BASE_DVISC = float("1.75e-5")
BASE_T = float("283.0")

SCALES = tuple(float(s) for s in ("0.25", "0.5", "1.0", "2.0", "4.0", "10.0"))
T_ROWS = tuple(float(s) for s in ("200.0", "240.0", "273.15", "320.0"))
MFPA_ROWS = tuple(float(s) for s in ("1.0e-8", "3.0e-7", "1.0e-6"))
DVISC_ROWS = tuple(float(s) for s in ("1.0e-5", "2.1e-5"))


def volume(dp: float) -> float:
    """`(pi/6)*dp**3`, as `ukca_volume_mode` forms `wvol` and `dvol`."""
    return (PI / 6.0) * dp * dp * dp


def build_grid() -> dict[str, np.ndarray]:
    """The box rows. Setup-independent by construction -- see the docstring."""
    wetdp: list[list[float]] = []
    drydp: list[list[float]] = []
    mfpa: list[float] = []
    dvisc: list[float] = []
    t: list[float] = []

    def add(scale: float, *, temp=BASE_T, mfp=BASE_MFPA, dv=BASE_DVISC):
        wet = [d * scale for d in WETDP]
        wetdp.append(wet)
        drydp.append([d * DRY_FRACTION for d in wet])
        mfpa.append(mfp)
        dvisc.append(dv)
        t.append(temp)

    for s in SCALES:
        add(s)
    for temp in T_ROWS:
        add(1.0, temp=temp)
    for mfp in MFPA_ROWS:
        add(1.0, mfp=mfp)
    for dv in DVISC_ROWS:
        add(1.0, dv=dv)

    wet = np.array(wetdp, dtype=np.float64)
    dry = np.array(drydp, dtype=np.float64)
    vol = np.vectorize(volume)
    return {
        "wetdp": wet,
        "drydp": dry,
        "wvol": vol(wet),
        "dvol": vol(dry),
        "rhopar": np.tile(np.array(RHOPAR, dtype=np.float64), (wet.shape[0], 1)),
        "mfpa": np.array(mfpa, dtype=np.float64),
        "dvisc": np.array(dvisc, dtype=np.float64),
        "t": np.array(t, dtype=np.float64),
    }


CALLS: tuple[tuple[int, int], ...] = ((1, 1), (2, 1), (3, 1), (1, 0))


def call_label(c: tuple[int, int]) -> str:
    return f"icoag={c[0]}|coag_on={c[1]}"


def expected_slots(mode: np.ndarray, modesol: np.ndarray) -> tuple[set[int], set[tuple[int, int]]]:
    """The `kii` modes and `kij` slots the three loops fill, re-derived.

    From `:243`, `:262`, `:276` and `:308` of the Fortran, in its own 1-based
    indices, converted to 0-based on the way out. This is the *reading* of the
    loops; `verify_outputs` compares it against what the compiled routine did,
    so a misreading fails the capture rather than becoming the port's
    specification.

    `modesol` is an argument rather than a constant even though it is
    `[1,1,1,1,0,0,0,0]` in every setup, because `:281` branches on it and a
    port that assumed the constant would be reading a different program.
    """
    active = {m for m in range(1, NMODES + 1) if mode[m - 1]}
    kii: set[int] = set()
    kij: set[tuple[int, int]] = set()

    for imode in range(MODE_NUC_SOL, MODE_COR_SOL + 1):  # `:243`
        if imode not in active:
            continue
        kii.add(imode)
        for jmode in range(imode + 1, MODE_COR_SOL + 1):  # `:262`
            if jmode in active:
                kij.add((imode, jmode))
        for jmode in range(imode + 4, MODE_SUP_INSOL + 1):  # `:276`
            if jmode in active:
                kij.add((imode, jmode))

    for imode in range(MODE_AIT_INSOL, MODE_SUP_INSOL + 1):  # `:295`
        if imode not in active:
            continue
        kii.add(imode)
        if imode < MODE_COR_INSOL:  # `:308`
            for jmode in range(imode - 2, MODE_COR_SOL + 1):  # `:309`
                if jmode in active:
                    kij.add((imode, jmode))

    del modesol  # read by the callee, not by the slot map
    return {m - 1 for m in kii}, {(i - 1, j - 1) for i, j in kij}


CHILD_BODY = (
    "\nimport tempfile\n"
    f"sys.path.insert(0, {str(REPO / 'validation')!r})\n"
    "from capture_coag_kernel_leaf import build_grid, CALLS, call_label, NMODES\n"
    "from leaf_common import bind_call\n"
    "\n"
    "call = bind_call(g)\n"
    "grid = build_grid()\n"
    "nb = grid['wetdp'].shape[0]\n"
    "args = (grid['drydp'], grid['dvol'], grid['wetdp'], grid['wvol'],\n"
    "        grid['rhopar'], grid['mfpa'], grid['dvisc'], grid['t'])\n"
    "kii_rows, kij_rows = [], []\n"
    "for _c in CALLS:\n"
    "    _kii, _kij, _e = call(call_label(_c), g.leaf_calc_coag_kernel, *args, _c[1], _c[0])\n"
    "    kii_rows.append(np.asarray(_kii))\n"
    "    kij_rows.append(np.asarray(_kij))\n"
    "\n"
    "# Every (i,j) pair through ukca_coag_coff_v directly, under BOTH size\n"
    "# conventions for the j partner, at icoag = 1. Which one each filled slot\n"
    "# matches is then a measurement rather than a reading of :281-286.\n"
    "_ones = np.ones(nb, dtype=np.int32)\n"
    "direct = {}\n"
    "for _tag, _dp, _vl in (('wetj', grid['wetdp'], grid['wvol']),\n"
    "                       ('dryj', grid['drydp'], grid['dvol'])):\n"
    "    _out = np.zeros((nb, NMODES, NMODES))\n"
    "    for _i in range(NMODES):\n"
    "        for _j in range(NMODES):\n"
    "            _k, _e2 = call('direct %s %d %d' % (_tag, _i, _j), g.leaf_coag_coff,\n"
    "                           _ones, grid['wetdp'][:, _i] / 2.0, _dp[:, _j] / 2.0,\n"
    "                           grid['wvol'][:, _i], _vl[:, _j],\n"
    "                           grid['rhopar'][:, _i], grid['rhopar'][:, _j],\n"
    "                           grid['mfpa'], grid['dvisc'], grid['t'], 1, 1)\n"
    "            _out[:, _i, _j] = np.asarray(_k)\n"
    "    direct[_tag] = _out\n"
    "\n"
    "_mode, _e3 = call('mode flags', g.wrap_mode_int, 'mode', NMODES)\n"
    "_modesol, _e4 = call('modesol flags', g.wrap_mode_int, 'modesol', NMODES)\n"
    "_tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)\n"
    "_tmp.close()\n"
    "np.savez(_tmp.name, kii=np.stack(kii_rows), kij=np.stack(kij_rows),\n"
    "         direct_wetj=direct['wetj'], direct_dryj=direct['dryj'],\n"
    "         mode=np.asarray(_mode), modesol=np.asarray(_modesol))\n"
    "print('@@RESULT@@' + json.dumps({'npz_path': _tmp.name, 'setup': _setup}))\n"
)

_CHILD = CHILD_PREAMBLE.format(f2py=str(F2PY_DIR)) + CHILD_BODY


# --- verification, all of it before np.savez_compressed ---------------------


def verify_source_structure() -> dict[str, int]:
    """Re-read the loop bounds this capture's slot map is derived from."""
    text = SOURCE.read_text(encoding="utf-8")
    squeezed = text.replace(" ", "")
    required = {
        "soluble outer loop": "DOimode=mode_nuc_sol,mode_cor_sol" in squeezed,
        "larger soluble inner": "DOjmode=(imode+1),mode_cor_sol" in squeezed,
        "larger insoluble inner": "DOjmode=(imode+4),mode_sup_insol" in squeezed,
        "insoluble outer loop": "DOimode=mode_ait_insol,mode_sup_insol" in squeezed,
        "insoluble guard": "IF(imode<mode_cor_insol)THEN" in squeezed,
        "smaller soluble inner": "DOjmode=(imode-2),mode_cor_sol" in squeezed,
        "modesol branch": "IF(modesol(jmode)==1)THEN" in squeezed,
        "insoluble imode takes wetdp": "rpi(:)=wetdp(:,imode)/2.0" in squeezed,
    }
    missing = sorted(k for k, ok in required.items() if not ok)
    if missing:
        raise SystemExit(
            "the vendored driver no longer contains: "
            + ", ".join(missing)
            + " -- this capture's slot map describes a routine that has changed"
        )
    return {"checked": len(required)}


def verify_outputs(data: dict, grid: dict) -> dict:
    """Everything the golden claims, checked against the golden itself."""
    report: dict[str, int] = {"setups": 0, "kij_slots": 0, "dryj_slots": 0, "wetj_slots": 0}
    nb = grid["wetdp"].shape[0]

    for s_i, setup in enumerate(SETUPS):
        mode = data["mode"][s_i]
        modesol = data["modesol"][s_i]
        kii = data["kii"][s_i]
        kij = data["kij"][s_i]

        if kii.shape != (len(CALLS), nb, NMODES) or kij.shape != (len(CALLS), nb, NMODES, NMODES):
            raise SystemExit(f"setup {setup}: shapes {kii.shape}, {kij.shape}")
        if modesol.tolist() != [1, 1, 1, 1, 0, 0, 0, 0]:
            raise SystemExit(
                f"setup {setup}: modesol is {modesol.tolist()}, not the structural split"
            )

        want_kii, want_kij = expected_slots(mode, modesol)

        for c_i, c in enumerate(CALLS):
            coag_on = c[1]
            got_kii = {m for m in range(NMODES) if np.any(kii[c_i, :, m] != 0.0)}
            got_kij = {
                (i, j)
                for i in range(NMODES)
                for j in range(NMODES)
                if np.any(kij[c_i, :, i, j] != 0.0)
            }
            if coag_on == 0:
                if got_kii or got_kij:
                    raise SystemExit(f"setup {setup}, {call_label(c)}: coag_on=0 left slots filled")
                continue
            if got_kii != want_kii:
                raise SystemExit(
                    f"setup {setup}, {call_label(c)}: kii modes {sorted(got_kii)} != "
                    f"expected {sorted(want_kii)}"
                )
            if got_kij != want_kij:
                raise SystemExit(
                    f"setup {setup}, {call_label(c)}: kij slots {sorted(got_kij)} != "
                    f"expected {sorted(want_kij)}"
                )
            if not np.all(np.isfinite(kij[c_i])) or not np.all(np.isfinite(kii[c_i])):
                raise SystemExit(f"setup {setup}, {call_label(c)}: non-finite output")

        # The transposition check the values cannot make: no slot is filled in
        # both orders, so `kij_arr` genuinely carries a direction.
        both = {(i, j) for (i, j) in want_kij if (j, i) in want_kij}
        if both:
            raise SystemExit(f"setup {setup}: slots filled in both orders: {sorted(both)}")

        # Which size convention each filled slot actually matches, measured.
        c_i = CALLS.index((1, 1))
        for i, j in sorted(want_kij):
            wet_ok = np.array_equal(kij[c_i, :, i, j], data["direct_wetj"][s_i][:, i, j])
            dry_ok = np.array_equal(kij[c_i, :, i, j], data["direct_dryj"][s_i][:, i, j])
            if wet_ok == dry_ok:
                raise SystemExit(
                    f"setup {setup}, slot ({i},{j}): matches neither convention "
                    f"(wet={wet_ok}, dry={dry_ok}) -- the driver passes something else"
                )
            insoluble_j = modesol[j] == 0
            if dry_ok != bool(insoluble_j):
                raise SystemExit(
                    f"setup {setup}, slot ({i},{j}): j is "
                    f"{'insoluble' if insoluble_j else 'soluble'} but the slot matches the "
                    f"{'dry' if dry_ok else 'wet'} convention"
                )
            report["dryj_slots" if dry_ok else "wetj_slots"] += 1
        report["kij_slots"] += len(want_kij)
        report["setups"] += 1

    if report["dryj_slots"] == 0:
        raise SystemExit(
            "no slot took the dry-size branch, so :281-286 was never exercised and "
            "the sweep cannot say which size an insoluble partner enters at"
        )
    return report


def capture(out_dir: Path, quiet: bool = False) -> Path:
    verify_source_structure()
    grid = build_grid()

    stacks: dict[str, list] = {
        k: [] for k in ("kii", "kij", "direct_wetj", "direct_dryj", "mode", "modesol")
    }
    paths: list[str] = []
    try:
        for setup in SETUPS:
            record = run_child(
                CHILD_BODY,
                namelist_text=render_namelist(NAMELIST_TEXT, setup, "default"),
                setup=setup,
                label=f"coag_kernel setup {setup}",
            )
            paths.append(record["npz_path"])
            with np.load(record["npz_path"]) as d:
                for k in stacks:
                    stacks[k].append(d[k])
            if not quiet:
                active = int(stacks["mode"][-1].sum())
                print(f"  i_mode_setup = {setup}: {active} active modes")
    finally:
        for p in paths:
            Path(p).unlink(missing_ok=True)

    data = {k: np.stack(v) for k, v in stacks.items()}

    # Setups whose `mode` vector is identical must give identical kernels, and
    # setups whose `mode` differs must not. Both directions, because a capture
    # that ran one setup seven times passes the first alone.
    prints = {
        str(s): {"kij": data["kij"][i].tolist(), "mode": data["mode"][i].tolist()}
        for i, s in enumerate(SETUPS)
    }
    same_mode = [
        (str(a), str(b))
        for ia, a in enumerate(SETUPS)
        for ib, b in enumerate(SETUPS)
        if ia < ib and np.array_equal(data["mode"][ia], data["mode"][ib])
    ]
    check_varied(prints, expected_identical=same_mode, what="coag_kernel setups")

    report = verify_outputs(data, grid)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ARCHIVE
    np.savez_compressed(
        path,
        setups=np.array(SETUPS, dtype=np.int64),
        call_icoag=np.array([c[0] for c in CALLS], dtype=np.int64),
        call_coag_on=np.array([c[1] for c in CALLS], dtype=np.int64),
        **data,
        **{k: grid[k] for k in ("wetdp", "drydp", "wvol", "dvol", "rhopar", "mfpa", "dvisc", "t")},
    )
    if not quiet:
        print(f"  wrote {path.name}: kij {data['kij'].shape}")
        print(
            f"  {report['kij_slots']} filled slots over {report['setups']} setups; "
            f"{report['dryj_slots']} took the dry-size branch, {report['wetj_slots']} the wet"
        )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    grid = build_grid()
    if args.dry_run:
        verify_source_structure()
        print(f"leaf coag_kernel sweep -> {args.out / ARCHIVE}")
        print(f"  boxes       {grid['wetdp'].shape[0]:>6,}")
        print(f"  calls       {len(CALLS):>6,}  {[call_label(c) for c in CALLS]}")
        print(f"  setups      {len(SETUPS):>6,}  {SETUPS}")
        all_on = np.ones(NMODES, dtype=np.int32)
        kii, kij = expected_slots(all_on, np.array([1, 1, 1, 1, 0, 0, 0, 0]))
        print(f"  slots with every mode active: {len(kii)} kii, {len(kij)} kij")
        print(f"    {sorted(kij)}")
        return 0

    print(f"sweeping leaf_calc_coag_kernel -> {args.out}")
    capture(args.out)
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
