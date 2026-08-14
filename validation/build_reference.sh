#!/usr/bin/env bash
# Build the vendored Fortran reference in a named precision variant.
#
#   ./validation/build_reference.sh f32     # as shipped: default REAL is kind=4
#   ./validation/build_reference.sh f64     # -fdefault-real-8
#   ./validation/build_reference.sh both
#
# Two properties this script must preserve, because the whole validation
# strategy depends on them:
#
#   1. `fortran/` is never modified. Variants are produced entirely through
#      make command-line overrides (BUILD/BIN/FCFLAGS), never by editing the
#      vendored Makefile. The script verifies this before exiting.
#
#   2. `-ffp-contract=off` on every variant. gfortran defaults to `fast`, which
#      contracts a*b+c into an FMA on aarch64 but not on older x86, and
#      differently at -O0 than at -O2. Goldens are compared by content hash, so
#      a reference that changes with the host's FMA behaviour is not a
#      reference. This does not make the build portable across platforms --
#      libm still differs -- but it does make it stable across optimisation
#      levels on one machine. See docs/REFERENCE_BUILD.md and ADR-005.
#
# Why f64 at all: the shipped Fortran runs in SINGLE precision (default REAL is
# kind=4, ~6 significant digits). Comparing a float64 JAX port against that
# disagrees at ~1e-6 for reasons that are not bugs. Building both lets us
# measure the precision floor and keep it separate from port error.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"
FORTRAN="$REPO/fortran"

COMMON_FLAGS="-O2 -ffree-line-length-none -fno-range-check -ffp-contract=off"

# Staged builds. The reference is built from a COPY of the vendored tree with
# validation/patches/*.patch applied, never from fortran/ itself. That keeps
# fortran/ byte-comparable with glomap-box (and hash-checkable by the tamper
# test) while still allowing the instrumentation the harness needs: a
# high-precision dump, budget output, branch-mask output.
#
# Only src/box/ is ever patched. That is new BSD-3 code, not Crown Copyright
# UKCA, so extending it is permitted; src/ukca/ stays untouched in the stage
# too, and the stage's own copy is hash-compared against the original.
STAGE_ROOT="$REPO/.refstage"

stage_tree() {
  local stage="$STAGE_ROOT/$1"
  rm -rf "$stage"; mkdir -p "$stage"
  # -a would drag in build products; copy only what the build needs.
  for d in src namelists tools tests; do cp -R "$FORTRAN/$d" "$stage/"; done
  cp "$FORTRAN/Makefile" "$stage/"

  shopt -s nullglob
  for p in "$REPO"/validation/patches/*.patch; do
    if ! patch -d "$stage" -p1 --forward --silent < "$p"; then
      echo "ERROR: overlay failed to apply: $(basename "$p")" >&2
      return 1
    fi
  done
  shopt -u nullglob

  # The overlays must not have touched UKCA science.
  if ! diff -rq "$FORTRAN/src/ukca" "$stage/src/ukca" >/dev/null; then
    echo "ERROR: an overlay modified src/ukca/, which is read-only." >&2
    return 1
  fi
  echo "$stage"
}

build_variant() {
  local variant="$1" extra=""
  case "$variant" in
    f32) extra="" ;;
    f64) extra="-fdefault-real-8" ;;
    *) echo "unknown variant '$variant' (expected f32 or f64)" >&2; return 2 ;;
  esac

  echo "==> staging and building ref-${variant}"
  local stage; stage="$(stage_tree "$variant")" || return 1

  make -C "$stage" \
       BUILD=build BIN=bin \
       FCFLAGS="$COMMON_FLAGS $extra -J build -I build" \
       >/dev/null

  local exe="$stage/bin/glomap_box"
  [ -x "$exe" ] || { echo "build produced no executable at $exe" >&2; return 1; }
  # Stable path for callers, independent of the staging layout.
  mkdir -p "$FORTRAN/bin-ref-${variant}"
  cp "$exe" "$FORTRAN/bin-ref-${variant}/glomap_box"
  echo "    $FORTRAN/bin-ref-${variant}/glomap_box"
}

verify_tree_untouched() {
  # The vendored tree is read-only (PROVENANCE.md). If a build ever needed to
  # edit it, that is a design failure and should stop the harness, loudly.
  if ! git -C "$REPO" diff --quiet -- fortran/; then
    echo "ERROR: building modified fortran/. The vendored tree is read-only;" >&2
    echo "variants must come from make overrides only." >&2
    git -C "$REPO" diff --stat -- fortran/ >&2
    return 1
  fi
}

record_toolchain() {
  # Goldens are only meaningful alongside the toolchain that produced them.
  local out="$FORTRAN/TOOLCHAIN.txt"
  {
    echo "# Toolchain used to build the reference. Goldens are NOT portable"
    echo "# across compilers or platforms -- see docs/REFERENCE_BUILD.md."
    echo "gfortran: $(gfortran --version | head -1)"
    echo "uname:    $(uname -srm)"
    echo "flags:    $COMMON_FLAGS"
    echo "f64_flag: -fdefault-real-8"
  } > "$out"
  echo "==> recorded $out"
}

main() {
  local what="${1:-both}"
  command -v gfortran >/dev/null || { echo "gfortran not found" >&2; exit 3; }

  case "$what" in
    f32|f64) build_variant "$what" ;;
    both) build_variant f32; build_variant f64 ;;
    *) echo "usage: $0 [f32|f64|both]" >&2; exit 2 ;;
  esac

  verify_tree_untouched
  record_toolchain
  echo "==> reference build complete"
}

main "$@"
