#!/usr/bin/env bash
# Verify that src/ukca/ equals upstream UKCA plus exactly the patches in
# patches/, and nothing else.
#
# Usage:  UKCA_ROOT=/path/to/ukca ./tools/verify_vendor.sh
#         make verify-vendor UKCA_ROOT=/path/to/ukca
#
# Exit 0 only if every vendored file either matches upstream byte-for-byte, or
# differs solely by an applied patch in patches/.

set -uo pipefail
cd "$(dirname "$0")/.."

UKCA_ROOT="${UKCA_ROOT:-../ukca}"
if [ ! -d "$UKCA_ROOT/src" ]; then
  echo "error: UKCA_ROOT=$UKCA_ROOT does not look like a ukca checkout" >&2
  echo "usage: UKCA_ROOT=/path/to/ukca $0" >&2
  exit 2
fi

echo "upstream: $UKCA_ROOT"
if git -C "$UKCA_ROOT" rev-parse HEAD >/dev/null 2>&1; then
  echo "upstream commit: $(git -C "$UKCA_ROOT" rev-parse HEAD)"
fi
echo

# Reconstruct pristine upstream for the vendored files, then re-apply patches,
# and compare the result against what is actually in src/ukca/.
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/src/ukca"

identical=0
patched=0
missing=0
differs=0

for f in src/ukca/*.F90; do
  base="$(basename "$f")"
  src="$(find "$UKCA_ROOT/src" -name "$base" -type f -print -quit 2>/dev/null)"
  if [ -z "$src" ]; then
    echo "MISSING UPSTREAM: $base"
    missing=$((missing + 1))
    continue
  fi
  cp "$src" "$work/src/ukca/$base"
done

# Apply every patch to the reconstructed upstream tree.
shopt -s nullglob
for p in patches/*.patch; do
  if ! (cd "$work" && patch -p1 --forward --silent < "$OLDPWD/$p"); then
    echo "ERROR: patch does not apply cleanly to upstream: $p"
    exit 1
  fi
done
shopt -u nullglob

for f in src/ukca/*.F90; do
  base="$(basename "$f")"
  [ -f "$work/src/ukca/$base" ] || continue
  if cmp -s "$f" "$work/src/ukca/$base"; then
    if cmp -s "$f" "$(find "$UKCA_ROOT/src" -name "$base" -type f -print -quit)"; then
      identical=$((identical + 1))
    else
      patched=$((patched + 1))
      echo "PATCHED (expected): $base"
    fi
  else
    differs=$((differs + 1))
    echo "UNEXPLAINED DIFFERENCE: $base"
    diff -u "$work/src/ukca/$base" "$f" | head -20
  fi
done

echo
echo "identical to upstream : $identical"
echo "differ only by patches: $patched"
echo "missing upstream      : $missing"
echo "unexplained differences: $differs"

if [ "$differs" -ne 0 ] || [ "$missing" -ne 0 ]; then
  echo "FAIL: src/ukca/ contains changes not accounted for by patches/"
  exit 1
fi
echo "PASS: src/ukca/ is upstream + patches/ exactly"
