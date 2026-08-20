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

# ${BASH_SOURCE[0]} rather than $0, so that sourcing this file (which is how
# tests/test_reference_build.py drives the patch gates) still lands in the repo
# root instead of wherever the sourcing shell happened to be.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
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
    require_unified_diff "$p" || return 1
    verify_additive_for_ukca "$p" || return 1
    # -u pins the interpretation to unified. Without it `patch` auto-detects,
    # so a context or normal diff would be applied happily by a format the
    # additive gate below cannot read. require_unified_diff already refuses
    # those; this makes the two agree by construction rather than by review.
    if ! patch -u -d "$stage" -p1 --forward --silent < "$p"; then
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

require_unified_diff() {
  # Refuse anything that is not a pure unified diff, BEFORE the additive gate
  # below parses it.
  #
  # The phase C review demonstrated two ways past a gate that reads unified
  # syntax while `patch` reads whatever it is handed. `patch` auto-detects
  # unified, context and normal diffs, and both of the other two changed
  # `se_ins = 1.0` to `0.3` in src/ukca/ukca_conden.F90 with the additive gate
  # exiting 0:
  #
  #   context diff  `--- b/src/ukca/...` sets the path but no `+++ ` ever
  #                 arrives, so the gate never arms; removals are `! ` lines.
  #   normal diff   there are no `---`/`+++ ` headers at all. The file is named
  #                 by an `Index:` line, the removal marker is `<`, and the
  #                 `---` separator carries no trailing space.
  #
  # Teaching one awk three grammars would make the most security-relevant code
  # in the harness also the most subtle. Instead there is exactly one accepted
  # grammar, checked structurally here: prose, then one or more file sections of
  # `--- ` / `+++ ` / `@@` with hunk bodies whose length matches the counts the
  # `@@` header declares. Counting the bodies is what makes the additive gate
  # exact -- it is then never guessing whether a `---` line is a header or the
  # removal of a line whose text begins with `--`.
  local patchfile="$1"
  awk -v name="$(basename "$patchfile")" '
    function reject(why) {
      printf "ERROR: %s is not a unified diff: %s\n  line %d: %s\n",
             name, why, NR, $0 > "/dev/stderr"
      bad = 1
      exit 1
    }
    function hunk_len(field,   n, parts) {
      # "-81,26" -> 26; "-81" -> 1 (an omitted count means exactly one line).
      n = split(field, parts, ",")
      return (n == 2) ? parts[2] + 0 : 1
    }

    # Inside a hunk body every line is accounted for by the @@ counts.
    body {
      c = substr($0, 1, 1)
      if (c == "\\") next                     # "\ No newline at end of file"
      else if (c == "-") old--
      else if (c == "+") new--
      else if (c == " " || c == "") { old--; new-- }
      else reject("hunk body line starts with " c ", not one of space + - \\")
      if (old < 0 || new < 0) reject("hunk body is longer than its @@ header declares")
      if (old == 0 && new == 0) body = 0
      next
    }

    want_plus {
      if ($0 !~ /^[+][+][+] /) reject("a \"--- \" header is not followed by \"+++ \"")
      want_plus = 0; want_hunk = 1; seen_file = 1
      next
    }

    want_hunk {
      if ($0 !~ /^@@ /) reject("a \"+++ \" header is not followed by an @@ hunk")
      want_hunk = 0
      # falls through to the @@ rule
    }

    /^@@ / {
      if (!seen_file) reject("an @@ hunk appears before any \"--- \"/\"+++ \" header")
      if ($0 !~ /^@@ -[0-9]+(,[0-9]+)? [+][0-9]+(,[0-9]+)? @@/) reject("malformed @@ header")
      old = hunk_len($2); new = hunk_len($3)
      seen_hunk = 1
      body = (old > 0 || new > 0)
      next
    }

    /^--- / { want_plus = 1; next }

    # Prose, and the `diff ...` lines that separate file sections. Nothing here
    # may look to `patch` like the start of some other diff format.
    {
      if ($0 ~ /^[*][*][*]/) reject("a context-diff marker")
      if ($0 ~ /^[+][+][+] /) reject("a \"+++ \" header with no \"--- \" before it")
      if ($0 ~ /^[0-9]+(,[0-9]+)?[acd][0-9]+(,[0-9]+)?[ \t]*$/) reject("a normal-diff command")
      if ($0 ~ /^---[ \t]*$/) reject("a normal-diff separator")
      if ($0 ~ /^Index:[ \t]/) reject("an \"Index:\" header, which names a file the gates never read")
    }

    END {
      if (bad) exit 1
      if (body) { printf "ERROR: %s ends inside a hunk\n", name > "/dev/stderr"; exit 1 }
      if (want_plus || want_hunk) {
        printf "ERROR: %s ends on an incomplete file header\n", name > "/dev/stderr"
        exit 1
      }
      if (!seen_hunk) {
        printf "ERROR: %s contains no unified diff hunk at all\n", name > "/dev/stderr"
        exit 1
      }
    }
  ' "$patchfile" || {
    echo "Overlays must be unified diffs (diff -u / git diff): the additive" >&2
    echo "src/ukca/ gate can only read that one format." >&2
    return 1
  }
}

verify_additive_for_ukca() {
  # Reject a patch that removes any line from a file under src/ukca/.
  #
  # Assumes require_unified_diff has already passed, so the hunk counts can be
  # trusted: the body rule below knows exactly which lines are hunk content and
  # never has to disambiguate a `---` header from a removal by regex.
  local patchfile="$1"
  # Every path the patch names is inspected -- the `diff` line, --- and +++ --
  # and each is normalised first. The phase B review found two ways past a check
  # that looked only at +++ verbatim: `+++ b/src/box/../ukca/ukca_conden.F90` is
  # applied by `patch -p1` to src/ukca/ but does not match the pattern, and a
  # file deletion puts /dev/null on +++ with the UKCA path on ---.
  awk -v name="$(basename "$patchfile")" '
    function norm(p,   out, n, i, parts, stack, top) {
      n = split(p, parts, "/"); top = 0
      for (i = 1; i <= n; i++) {
        if (parts[i] == "." || parts[i] == "") continue
        if (parts[i] == "..") { if (top > 0) top--; continue }
        stack[++top] = parts[i]
      }
      out = ""
      for (i = 1; i <= top; i++) out = out "/" stack[i]
      return out
    }
    function hunk_len(field,   n, parts) {
      n = split(field, parts, ",")
      return (n == 2) ? parts[2] + 0 : 1
    }

    body {
      c = substr($0, 1, 1)
      if (c == "\\") next
      else if (c == "-") {
        old--
        if (in_ukca) {
          printf "ERROR: %s removes a line from src/ukca/: %s\n", name, $0 > "/dev/stderr"
          bad = 1
        }
      }
      else if (c == "+") new--
      else { old--; new-- }
      if (old <= 0 && new <= 0) body = 0
      next
    }

    /^diff / {
      hint = 0
      for (i = 2; i <= NF; i++) if (norm($i) ~ /src\/ukca\//) hint = 1
      next
    }
    /^--- / { from = norm($2); next }
    /^[+][+][+] / {
      in_ukca = (norm($2) ~ /src\/ukca\// || from ~ /src\/ukca\// || hint)
      from = ""; hint = 0
      next
    }
    /^@@ / { old = hunk_len($2); new = hunk_len($3); body = (old > 0 || new > 0); next }

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

# Sourcing the script defines the gates without building anything, so
# tests/test_reference_build.py can drive require_unified_diff and
# verify_additive_for_ukca directly over fixture patches. Without this hook the
# file has no testable surface that does not need gfortran, which is how the
# additive gate reached phase C with no test at all.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  main "$@"
fi
