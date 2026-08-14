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

build_variant() {
  local variant="$1" extra=""
  local build="build-ref-${variant}" bin="bin-ref-${variant}"

  case "$variant" in
    f32) extra="" ;;
    f64) extra="-fdefault-real-8" ;;
    *) echo "unknown variant '$variant' (expected f32 or f64)" >&2; return 2 ;;
  esac

  echo "==> building ref-${variant}"
  make -C "$FORTRAN" \
       BUILD="$build" BIN="$bin" \
       FCFLAGS="$COMMON_FLAGS $extra -J $build -I $build" \
       >/dev/null

  local exe="$FORTRAN/$bin/glomap_box"
  [ -x "$exe" ] || { echo "build produced no executable at $exe" >&2; return 1; }
  echo "    $exe"
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
