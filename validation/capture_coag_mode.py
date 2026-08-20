#!/usr/bin/env python3
"""Capture `coag_mode`, the mode coagulation table (task 33).

    python validation/capture_coag_mode.py --dry-run
    python validation/capture_coag_mode.py
    python validation/capture_coag_mode.py --emit-literal

`coag_mode(nmodes,nmodes)` is declared at `ukca_mode_setup.F90:174` and read at
exactly one place, `ukca_coagwithnucl.F90:534-535`, where it names the mode that
receives the mass leaving `imode` when `imode` coagulates with `jmode`.

Two independent mechanical readings of the same 64 integers, because each one
misses what the other catches:

* **`parse_source`** re-parses the `RESHAPE` literal out of the vendored file.
  This is what pins the committed Python table — the port must never contain a
  number that a human typed while looking at Fortran. It cannot tell you what
  the compiler did with the literal.
* **the f2py capture** reads the constant out of the built extension, once with
  no init at all and then once per `i_mode_setup` in its own process. This is
  what pins the *claim* that the table is setup-independent.

Neither is decoration. The second is the one that would catch a `coag_mode`
that some setup routine quietly rebound; the first is the one that would catch
a mistyped transcription — with the caveat, stated here rather than discovered
later, that **the table is symmetric**, so no reading of it, mechanical or
otherwise, can detect a transposition. That is a property of the data and it is
asserted, not assumed.

The anti-tautology guard
------------------------

A capture that returns the same bytes for every setup is exactly the shape of
the failure this repo has already shipped once: a replacement no-op'd, every
setup got the same record, and every byte-equality test passed. Here the
identical bytes are the *expected* result, which makes that failure invisible —
so the archive also carries `mode`, `topmode` and `ncp` per setup, which
genuinely do vary. `tests/test_coag_mode.py` asserts both directions:
`coag_mode` identical across all seven setups, and the witness fields *not*
identical. A capture that ran setup 1 seven times fails the second, here and in
the test.

ONE SUBPROCESS PER SETUP. `ukca_mode_setup` allocates under
`IF (.NOT. ALLOCATED)` and never deallocates and the 283 `nmas*` budget indices
have no initialiser, so a second init in one process leaves stale indices; the
binding refuses it. See `docs/harness.md`.

Integers, so unlike every other golden here this one is bit-reproducible across
compilers and platforms. It still obeys ADR-005 — generated once, committed,
never regenerated in CI — because the point of the drift gate is that a change
is a finding, and here a change could only be a real one.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
F2PY_DIR = REPO / "validation" / "f2py"
NAMELISTS = REPO / "fortran" / "namelists"
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_mode_setup.F90"
DEFAULT_OUT = REPO / "tests" / "goldens"
ARCHIVE = "coagmode.f64.tables.npz"

SETUPS = (1, 2, 3, 4, 5, 6, 8)
NMODES = 8

# ---------------------------------------------------------------------------
# Reading 1: the vendored source text.
# ---------------------------------------------------------------------------

_DECL = re.compile(
    r"INTEGER,\s*PARAMETER\s*::\s*coag_mode\s*\(\s*nmodes\s*,\s*nmodes\s*\)\s*="
    r"\s*RESHAPE\s*\(\s*\[(?P<body>.*?)\]\s*,\s*&?\s*\[\s*nmodes\s*,\s*nmodes\s*\]\s*\)",
    re.DOTALL,
)


def parse_source(text: str | None = None) -> np.ndarray:
    """The `coag_mode` literal, parsed out of `ukca_mode_setup.F90`.

    Every step asserts its own match count. A regex that silently finds nothing
    — or finds two declarations and takes the first — is how a capture script
    ends up producing a golden full of the wrong data that every byte-equality
    test then passes.
    """
    if text is None:
        text = SOURCE.read_text(encoding="utf-8")

    matches = _DECL.findall(text)
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly 1 coag_mode PARAMETER declaration in {SOURCE.name}, "
            f"found {len(matches)}"
        )
    body = matches[0]

    joined, n_cont = re.subn(r"&\s*\n\s*", "", body)
    if n_cont < 1:
        raise AssertionError(
            "the coag_mode literal spans continuation lines in the vendored source; "
            f"joined {n_cont} of them, which means the regex matched something else"
        )

    tokens = [t.strip() for t in joined.split(",")]
    tokens = [t for t in tokens if t]
    if len(tokens) != NMODES * NMODES:
        raise AssertionError(f"expected {NMODES * NMODES} entries, parsed {len(tokens)}")
    for tok in tokens:
        if not re.fullmatch(r"[+-]?\d+", tok):
            raise AssertionError(f"non-integer entry in the coag_mode literal: {tok!r}")

    # Fortran RESHAPE fills column-major: the first eight entries are
    # coag_mode(1:8, 1). Written out rather than assumed.
    return np.array([int(t) for t in tokens], dtype=np.int32).reshape((NMODES, NMODES), order="F")


def emit_literal(table: np.ndarray) -> str:
    """The `_RESHAPE_SOURCE_ORDER` block for `physics/coag_mode.py`.

    Emitted one tuple per Fortran source line — i.e. per *column* — so the
    committed block can be diffed against `ukca_mode_setup.F90:174-183` by eye
    without anyone having to hold the column-major fill in their head.
    """
    lines = ["_RESHAPE_SOURCE_ORDER: tuple[tuple[int, ...], ...] = ("]
    for jmode in range(table.shape[1]):
        column = ", ".join(str(int(v)) for v in table[:, jmode])
        lines.append(f"    ({column}),")
    lines.append(")")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reading 2: the built extension, one process per setup.
# ---------------------------------------------------------------------------

_CHILD = r"""
import json, pathlib, re, sys, tempfile

f2py_dir, namelists, setup_arg = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, f2py_dir)
import glomap_f2py as g

out = {"setup": None if setup_arg == "none" else int(setup_arg)}

# coag_mode is a PARAMETER, so this is legal with no init at all -- and being
# legal with no init is the whole reason the accessor omits the guard.
nmodes = int(g.wrap_coag_nmodes())
out["nmodes"] = nmodes

table, ierr = g.wrap_coag_mode(nmodes)
assert ierr == 0, ("wrap_coag_mode", ierr)
out["coag_mode"] = [[int(v) for v in row] for row in table]

# The same constant read the way the consumer reads it, one subscript pair at a
# time, so the whole-array copy is cross-checked against an indexed read.
dest = [[0] * nmodes for _ in range(nmodes)]
for i in range(1, nmodes + 1):
    for j in range(1, nmodes + 1):
        v, e = g.wrap_coag_dest(i, j)
        assert e == 0, ("wrap_coag_dest", i, j, e)
        dest[i - 1][j - 1] = int(v)
out["coag_dest"] = dest

# The error paths must actually fire. An accessor whose guard never triggers is
# an accessor whose guard might not exist.
_, e = g.wrap_coag_mode(nmodes + 1)
assert e == 2, ("extent guard did not fire", e)
_, e = g.wrap_coag_dest(0, 1)
assert e == 2, ("imode guard did not fire", e)
_, e = g.wrap_coag_dest(1, nmodes + 1)
assert e == 2, ("jmode guard did not fire", e)
out["guards_fired"] = 3

if out["setup"] is not None:
    src = pathlib.Path(namelists) / "boundary_layer.nml"
    text = src.read_text()
    text, n = re.subn(
        r"^(\s*i_mode_setup\s*=\s*)\d+",
        lambda m: m.group(1) + str(out["setup"]),
        text,
        count=1,
        flags=re.MULTILINE,
    )
    assert n == 1, "failed to set i_mode_setup in the namelist"
    nml = pathlib.Path(tempfile.mkdtemp()) / "setup.nml"
    nml.write_text(text)

    ierr = g.wrap_init(str(nml))
    if ierr != 0:
        print("@@FAIL@@" + json.dumps({"setup": out["setup"], "ierr": int(ierr)}))
        raise SystemExit(0)

    # Read the constant AGAIN, after common_mode_setup_interface has run. If any
    # setup routine rebound a table of this name, this is where it shows.
    table2, e = g.wrap_coag_mode(nmodes)
    assert e == 0, ("wrap_coag_mode after init", e)
    out["coag_mode_after_init"] = [[int(v) for v in row] for row in table2]

    # The witness fields: these DO vary by setup, so a capture that silently ran
    # the same setup every time is detectable.
    mode, e = g.wrap_mode_int("mode", nmodes)
    assert e == 0, ("mode", e)
    out["mode"] = [int(x) for x in mode]
    out["topmode"] = int(g.wrap_topmode()[0])
    out["ncp"] = int(g.wrap_sizes()[2])

print("@@RESULT@@" + json.dumps(out))
"""


def capture_one(setup: int | None) -> dict:
    """Run one setup — or no setup at all — in its own process."""
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(F2PY_DIR), str(NAMELISTS), str(setup or "none")],
        capture_output=True,
        text=True,
    )
    label = "no init" if setup is None else f"setup {setup}"
    if proc.returncode != 0:
        raise SystemExit(f"{label} failed:\n{proc.stdout}\n{proc.stderr}")
    if "@@FAIL@@" in proc.stdout:
        payload = json.loads(proc.stdout[proc.stdout.rindex("@@FAIL@@") + 8 :])
        raise SystemExit(f"{label}: wrap_init returned ierr={payload['ierr']}")
    if "@@RESULT@@" not in proc.stdout:
        raise SystemExit(f"{label}: child produced no result:\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout[proc.stdout.rindex("@@RESULT@@") + 10 :])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--emit-literal", action="store_true")
    args = parser.parse_args(argv)

    from_source = parse_source()

    if args.emit_literal:
        print(emit_literal(from_source))
        return 0

    if args.dry_run:
        print(f"{len(SETUPS)} setups, one subprocess each, plus one with no init")
        print(f"  -> {args.out / ARCHIVE}")
        print(f"  source parse: {from_source.shape} {from_source.dtype}")
        print(f"  symmetric:    {np.array_equal(from_source, from_source.T)}")
        return 0

    arrays: dict[str, np.ndarray] = {}

    noinit = capture_one(None)
    if noinit["nmodes"] != NMODES:
        raise SystemExit(f"nmodes is {noinit['nmodes']}, expected {NMODES}")
    baseline = np.array(noinit["coag_mode"], dtype=np.int32)
    np.testing.assert_array_equal(
        baseline,
        np.array(noinit["coag_dest"], dtype=np.int32),
        err_msg="whole-array read disagrees with the indexed read",
    )
    np.testing.assert_array_equal(
        baseline, from_source, err_msg="the built extension disagrees with the source literal"
    )
    arrays["coag_mode_noinit"] = baseline
    print(f"  no init : coag_mode read, {noinit['guards_fired']} error guards fired")

    witnesses: list[tuple] = []
    for setup in SETUPS:
        rec = capture_one(setup)
        table = np.array(rec["coag_mode"], dtype=np.int32)
        after = np.array(rec["coag_mode_after_init"], dtype=np.int32)
        np.testing.assert_array_equal(
            table, after, err_msg=f"setup {setup}: coag_mode changed across wrap_init"
        )
        np.testing.assert_array_equal(
            table, baseline, err_msg=f"setup {setup}: coag_mode differs from the no-init read"
        )
        np.testing.assert_array_equal(
            table,
            np.array(rec["coag_dest"], dtype=np.int32),
            err_msg=f"setup {setup}: indexed read disagrees with the whole-array read",
        )
        arrays[f"s{setup}_coag_mode"] = table
        arrays[f"s{setup}_mode"] = np.array(rec["mode"], dtype=np.int32)
        arrays[f"s{setup}_topmode"] = np.array(rec["topmode"], dtype=np.int32)
        arrays[f"s{setup}_ncp"] = np.array(rec["ncp"], dtype=np.int32)
        witnesses.append((tuple(rec["mode"]), rec["topmode"], rec["ncp"]))
        print(
            f"  setup {setup} : identical table; "
            f"active={sum(rec['mode'])} modes topmode={rec['topmode']} ncp={rec['ncp']}"
        )

    # The guard against a capture that quietly ran one setup seven times. The
    # table being invariant is the finding; the witnesses being invariant would
    # mean the capture never varied its input.
    if len(set(witnesses)) < 2:
        raise SystemExit(
            "every setup returned the same mode/topmode/ncp -- the capture did not "
            "actually vary i_mode_setup, so the invariance of coag_mode proves nothing"
        )
    print(f"  witness : {len(set(witnesses))} distinct (mode, topmode, ncp) across setups")

    arrays["_case"] = np.array("coagmode")
    arrays["_mode"] = np.array("tables")
    arrays["_variant"] = np.array("f64")
    arrays["_setups"] = np.array(SETUPS, dtype=np.int32)
    arrays["_nmodes"] = np.array(NMODES, dtype=np.int32)
    arrays["_rows"] = np.array(len(arrays), dtype=np.int64)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ARCHIVE
    np.savez_compressed(path, **arrays)
    print(f"wrote {path.name}  {path.stat().st_size / 1e3:.1f} kB")
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
