#!/usr/bin/env python3
"""Emit the Fortran source files under src/ in a valid compilation order.

Fortran module files must be compiled before anything that USEs them. This
script parses `MODULE` and `USE` statements, builds the file-level dependency
graph and topologically sorts it, so the Makefile never needs a hand-written
ordering.

Usage:
    python3 tools/gen_build_order.py [src_dir]        # print ordered files
    python3 tools/gen_build_order.py --check          # report external modules
    python3 tools/gen_build_order.py --make BUILDDIR  # emit make dependencies
"""

import os
import re
import sys
from collections import defaultdict

MOD_RE = re.compile(r"^\s*MODULE\s+([A-Za-z_]\w*)\s*$", re.I)
END_RE = re.compile(r"^\s*END\s*MODULE", re.I)
USE_RE = re.compile(r"^\s*USE\s*(?:,\s*INTRINSIC\s*::)?\s*([A-Za-z_]\w*)", re.I)

# Modules provided by the compiler / language, never by a source file.
INTRINSIC = {"iso_fortran_env", "iso_c_binding", "ieee_arithmetic",
             "ieee_exceptions", "ieee_features", "omp_lib"}


def scan(src_dir):
    """Return (module -> defining file, file -> set of USEd modules)."""
    mod_of = {}
    uses_of = {}
    for root, _, files in os.walk(src_dir):
        for name in sorted(files):
            if not name.endswith((".F90", ".f90")):
                continue
            path = os.path.join(root, name)
            uses = set()
            with open(path, errors="replace") as fh:
                for line in fh:
                    if line.lstrip().startswith("!"):
                        continue
                    m = MOD_RE.match(line)
                    if m:
                        mod_of[m.group(1).lower()] = path
                    u = USE_RE.match(line)
                    if u:
                        uses.add(u.group(1).lower())
            uses_of[path] = uses
    return mod_of, uses_of


def order(mod_of, uses_of):
    """Topologically sort files so providers precede consumers."""
    deps = {}
    for path, uses in uses_of.items():
        deps[path] = {mod_of[u] for u in uses
                      if u in mod_of and mod_of[u] != path}

    done, result = set(), []
    remaining = set(uses_of)
    while remaining:
        ready = sorted(p for p in remaining if deps[p] <= done)
        if not ready:
            # Circular USE dependency: report it rather than emit a bad order.
            cycle = "\n  ".join(sorted(remaining))
            sys.exit(f"error: circular module dependency among:\n  {cycle}")
        for path in ready:
            result.append(path)
            done.add(path)
            remaining.discard(path)
    return result


def externals(mod_of, uses_of):
    """Modules USEd but defined nowhere in the tree (excluding intrinsics)."""
    missing = defaultdict(list)
    for path, uses in uses_of.items():
        for u in uses:
            if u not in mod_of and u not in INTRINSIC:
                missing[u].append(path)
    return missing


def emit_make(mod_of, uses_of, build_dir):
    """Emit make rules: each object depends on its source and on the objects
    that produce the .mod files it needs. This gives make enough information
    for correct ordering under -j as well as incremental rebuilds."""
    def obj(path):
        return os.path.join(build_dir,
                            os.path.splitext(os.path.basename(path))[0] + ".o")

    objs = [obj(p) for p in sorted(uses_of)]
    print("OBJS :=", " \\\n\t".join(objs))
    print()
    for path in sorted(uses_of):
        providers = sorted({obj(mod_of[u]) for u in uses_of[path]
                            if u in mod_of and mod_of[u] != path})
        deps = " ".join([path] + providers)
        # The order-only dependency on the build directory is required, not
        # cosmetic: gfortran writes each module as <name>.mod0 and renames it
        # into the -J directory. If that directory does not exist the rename
        # fails with a confusing "Cannot rename module file" error rather than
        # a missing-directory one.
        print(f"{obj(path)}: {deps} | {build_dir}")
        print("\t$(COMPILE)")
        print()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src_dir = args[0] if args else "src"

    if "--make" in sys.argv:
        # --make BUILDDIR [src_dir]
        build_dir = args[0] if args else "build"
        src_dir = args[1] if len(args) > 1 else "src"
        mod_of, uses_of = scan(src_dir)
        emit_make(mod_of, uses_of, build_dir)
        return 0

    mod_of, uses_of = scan(src_dir)

    if "--check" in sys.argv:
        missing = externals(mod_of, uses_of)
        if not missing:
            print(f"OK: {len(uses_of)} files, no external module dependencies")
            return 0
        print("external module dependencies (need stubs):")
        for mod, users in sorted(missing.items()):
            print(f"  {mod}  <- {', '.join(sorted(users))}")
        return 1

    print("\n".join(order(mod_of, uses_of)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
