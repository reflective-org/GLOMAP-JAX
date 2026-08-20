#!/usr/bin/env bash
# Build the in-process binding used by validation gate A.
#
#   ./validation/build_f2py.sh
#
# Two stages, and the order is not negotiable.
#
#   1. Build the 46 vendored sources with the vendored Makefile, producing
#      objects and .mod files under a build directory.
#   2. Run f2py on the WRAPPER ONLY, compiling against those .mod files and
#      linking the objects.
#
# The TOMAS precedent (three fixed-form F77 files with COMMON blocks and no
# modules) does not transfer: GLOMAP is 46 free-form .F90 with a USE DAG, and
# f2py has no notion of module dependency order. Handing it all 46 sources
# fails on the first `USE` of a module it has not compiled yet. Letting the
# Makefile do what it already does correctly, and giving f2py a single
# self-contained file to wrap, sidesteps the problem entirely.
#
# Everything is built at -fdefault-real-8. Gate A exists to reach ~1e-14, and
# ref-f32 is diagnostic only (ADR-001), so a single-precision binding would
# have nothing to compare against. The wrapper declares its interface
# REAL(KIND=8) explicitly rather than relying on an .f2py_f2cmap: f2py maps the
# *token* `real` to C float whatever the compiler flags say, which would feed
# float32 buffers into real(8) dummies and produce garbage rather than an
# error.
#
# Built from the PLAIN vendored tree, not from a patched stage. The overlays
# only add instrumentation -- verified byte-identical trajectories -- so the
# science the binding exercises is the same science that produced the goldens,
# without dragging the dump modules into the extension.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"
FORTRAN="$REPO/fortran"
OUT="$REPO/validation/f2py"
BUILD="build-f2py"
MODULE="glomap_f2py"

FCFLAGS="-O2 -ffree-line-length-none -fno-range-check -ffp-contract=off -fdefault-real-8 -fPIC"

command -v gfortran >/dev/null || { echo "gfortran not found" >&2; exit 3; }

PY="${PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

# numpy.f2py -c shells out to `meson` BY NAME through PATH -- it does not use
# the interpreter that imported numpy. Using .venv/bin/python is not enough.
for tool in meson ninja; do
  command -v "$tool" >/dev/null || {
    echo "ERROR: $tool not on PATH. numpy.f2py -c invokes it by name." >&2
    echo "  brew install meson ninja   (or: uv pip install meson ninja)" >&2
    exit 4
  }
done

echo "==> 1/2  building the vendored sources (-fdefault-real-8, -fPIC)"
make -C "$FORTRAN" BUILD="$BUILD" BIN="bin-$BUILD" \
     FCFLAGS="$FCFLAGS -J $BUILD -I $BUILD" >/dev/null

# Stage 1 also links $(BIN)/glomap_box. On a REBUILD, make finds the shimmed
# ereport_mod.o newer than its source, does not rebuild it, and relinks the
# executable with the shim inside -- producing a box model that prints
# "UKCA ERROR (shim, not fatal)" and writes a complete CSV anyway,
# indistinguishable at a glance from fortran/bin/glomap_box. Nothing here
# needs it, so it goes.
rm -rf "$FORTRAN/bin-$BUILD"

echo "==> 2/2  compiling the state module, then f2py on the wrapper only"
# The state module holds the derived-type box state. f2py is never handed it:
# f90mod_rules would try to expose TYPE(box_env_type) and abort with
# KeyError: 'void'. Compiled here and linked in as an ordinary object.
gfortran $FCFLAGS -J "$FORTRAN/$BUILD" -I "$FORTRAN/$BUILD" \
         -c "$OUT/glomap_f2py_state_mod.F90" -o "$FORTRAN/$BUILD/glomap_f2py_state_mod.o"

# The ereport shim REPLACES src/ukca/ereport_mod.o in this extension only.
# The real one does STOP 1 on a fatal error, which inside a Python extension
# terminates the interpreter with no traceback -- there are twenty reachable
# call sites, so any driver aimed near an error path takes the test session
# with it. Same module name and same signature, so every already-compiled
# caller links against it unchanged; overwriting the object and the .mod in
# place is enough. validation/build_reference.sh never sees this, so no golden
# is affected. See docs/harness.md.
gfortran $FCFLAGS -J "$FORTRAN/$BUILD" -I "$FORTRAN/$BUILD" \
         -c "$OUT/glomap_ereport_shim.F90" -o "$FORTRAN/$BUILD/ereport_mod.o"

# The program object carries a `main`, which would collide at link time.
OBJECTS=$(ls "$FORTRAN/$BUILD"/*.o | grep -v '/glomap_box\.o$' | tr '\n' ' ')

rm -rf "$OUT/_build"
mkdir -p "$OUT/_build"
cd "$OUT/_build"

# shellcheck disable=SC2086
PATH="$(dirname "$PY"):$PATH" "$PY" -m numpy.f2py \
  -c "$OUT/glomap_f2py_mod.F90" "$OUT/glomap_leaf_mod.F90" "$OUT/glomap_modes_mod.F90" \
  "$OUT/glomap_gasidx_mod.F90" \
  "$OUT/glomap_coagmode_mod.F90" \
  -m "$MODULE" \
  --f90flags="$FCFLAGS -I$FORTRAN/$BUILD" \
  -I"$FORTRAN/$BUILD" \
  $OBJECTS \
  --quiet

SO=$(ls "$MODULE".*.so 2>/dev/null | head -1)
[ -n "$SO" ] || { echo "f2py produced no extension module" >&2; exit 1; }
mv "$SO" "$OUT/$SO"
cd "$OUT" && rm -rf _build

echo "==> built $OUT/$SO"

if ! git -C "$REPO" diff --quiet -- fortran/; then
  echo "ERROR: building modified fortran/. The vendored tree is read-only." >&2
  exit 1
fi
