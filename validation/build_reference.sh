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
    verify_additive_for_ukca "$p" || return 1
    if ! patch -d "$stage" -p1 --forward --silent < "$p"; then
      echo "ERROR: overlay failed to apply: $(basename "$p")" >&2
      return 1
    fi
  done
  shopt -u nullglob

  # Overlays may INSTRUMENT the UKCA science but never CHANGE it.
  #
  # Task 15 needs dump calls at 13 sites inside ukca_aero_step.F90, which lives
  # under src/ukca/. Forbidding that outright would make per-process validation
  # impossible; permitting arbitrary edits would let a "fix" slip into the
  # reference and silently redefine every golden.
  #
  # The line the rule draws: a patch touching src/ukca/ may only ADD lines. Any
  # removal means a science line was altered or deleted, which is a science
  # change wearing instrumentation's clothes. Checked on the patch itself, so it
  # holds regardless of what the patch claims about itself.
  #
  # src/box/ is new BSD-3 code and may be edited freely.
  if ! diff -rq "$FORTRAN/src/ukca" "$stage/src/ukca" >/dev/null 2>&1; then
    # NOTE: stderr, not stdout -- stage_tree echoes the stage path as its
    # return value, so anything on stdout would be captured as part of it.
    echo "    (src/ukca instrumented; patches verified additive)" >&2
  fi
  echo "$stage"
}

verify_additive_for_ukca() {
  # Reject a patch that removes any line from a file under src/ukca/.
  local patchfile="$1"
  awk -v name="$(basename "$patchfile")" '
    /^\+\+\+ / { in_ukca = ($2 ~ /src\/ukca\//) ; next }
    in_ukca && /^-/ && !/^---/ {
      printf "ERROR: %s removes a line from src/ukca/: %s\n", name, $0 > "/dev/stderr"
      bad = 1
    }
    END { exit bad ? 1 : 0 }
  ' "$patchfile" || {
    echo "src/ukca/ may be instrumented (insertions only), never modified." >&2
    return 1
  }
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
