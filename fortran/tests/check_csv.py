#!/usr/bin/env python3
"""Assertions over a box-model CSV output file.

Usage:
    check_csv.py FILE finite
    check_csv.py FILE constant  COLPREFIX  [rtol]
    check_csv.py FILE decreases COLPREFIX
    check_csv.py FILE increases COLPREFIX
    check_csv.py FILE exceeds   COLPREFIX  VALUE
    check_csv.py FILE sum_constant COLPREFIX [rtol]

COLPREFIX selects columns by prefix match on the header name, so `N_aitsol`
matches the Aitken number column and `M_` matches every component mass column.
"""

import csv
import sys


def load(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    header = [h.strip() for h in rows[0]]
    data = [[float(v) for v in r] for r in rows[1:] if r]
    return header, data


def cols(header, prefix):
    idx = [i for i, h in enumerate(header) if h.startswith(prefix)]
    if not idx:
        sys.exit(f"FAIL: no column with prefix '{prefix}' in {header[:6]}...")
    return idx


def series(header, data, prefix):
    idx = cols(header, prefix)
    return [sum(row[i] for i in idx) for row in data]


def main():
    path, check = sys.argv[1], sys.argv[2]
    header, data = load(path)
    if len(data) < 2:
        sys.exit(f"FAIL: {path} has fewer than 2 data rows")

    if check == "finite":
        for r, row in enumerate(data):
            for i, v in enumerate(row):
                if v != v or abs(v) == float("inf"):
                    sys.exit(f"FAIL: non-finite value in {header[i]} at row {r}")
                if header[i].startswith(("N_", "M_", "Ddry", "Dwet", "rhop")) and v < 0:
                    sys.exit(f"FAIL: negative {header[i]} = {v} at row {r}")
        print("  ok: all values finite and non-negative")
        return

    prefix = sys.argv[3]
    vals = series(header, data, prefix)
    first, last = vals[0], vals[-1]

    if check == "constant":
        rtol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-10
        scale = max(abs(first), 1e-300)
        worst = max(abs(v - first) / scale for v in vals)
        if worst > rtol:
            sys.exit(f"FAIL: {prefix} not constant: {first:.6e} -> {last:.6e} "
                     f"(rel dev {worst:.3e} > {rtol:.3e})")
        print(f"  ok: {prefix} constant at {first:.6e} (max rel dev {worst:.2e})")

    elif check == "sum_constant":
        rtol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-6
        scale = max(abs(first), 1e-300)
        worst = max(abs(v - first) / scale for v in vals)
        if worst > rtol:
            sys.exit(f"FAIL: sum({prefix}) not conserved: {first:.6e} -> "
                     f"{last:.6e} (rel dev {worst:.3e} > {rtol:.3e})")
        print(f"  ok: sum({prefix}) conserved to {worst:.2e} relative")

    elif check == "decreases":
        if not last < first:
            sys.exit(f"FAIL: {prefix} did not decrease: {first:.6e} -> {last:.6e}")
        print(f"  ok: {prefix} decreased {first:.6e} -> {last:.6e}")

    elif check == "increases":
        if not last > first:
            sys.exit(f"FAIL: {prefix} did not increase: {first:.6e} -> {last:.6e}")
        print(f"  ok: {prefix} increased {first:.6e} -> {last:.6e}")

    elif check == "exceeds":
        target = float(sys.argv[4])
        peak = max(vals)
        if not peak > target:
            sys.exit(f"FAIL: max({prefix}) = {peak:.6e} did not exceed {target:.6e}")
        print(f"  ok: max({prefix}) = {peak:.6e} exceeds {target:.6e}")

    else:
        sys.exit(f"unknown check '{check}'")


if __name__ == "__main__":
    main()
