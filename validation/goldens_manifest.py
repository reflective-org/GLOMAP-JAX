#!/usr/bin/env python3
"""Content manifest for the golden archives, and the gate that checks it.

    python validation/goldens_manifest.py --check     # exit 1 on drift
    python validation/goldens_manifest.py --write     # regenerate

A golden that changes silently is worse than no golden: the suite goes on
passing and the reference it is passing against is no longer the one anybody
reviewed. The manifest closes that by recording, for every archive, each array's
**name, dtype, shape and content hash**, plus the provenance the capture tool
embeds and the toolchain that produced it.

Three distinct failures, deliberately reported separately, because they mean
different things and call for different responses:

    drift    a listed archive's contents changed  -> investigate, do not re-bless
    orphan   an archive exists that nothing lists -> someone captured and forgot
    missing  a listed archive is gone             -> a partial checkout or a
                                                     deleted fixture

**Array contents are hashed, not the `.npz` file.** `np.savez_compressed` writes
a zip, and zip entries carry a modification timestamp, so the same data written
twice gives two different file hashes. A file-level hash would fail on every
regeneration and be loosened away within a week. Hashing the decoded arrays also
means the manifest says something a human can act on — *which* array moved.

The manifest must pass with **zero** fixtures. It lands before the goldens do
(task 17 before task 19), so an empty goldens directory with no manifest is a
valid, passing state rather than an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GOLDENS = REPO / "tests" / "goldens"
MANIFEST = GOLDENS / "MANIFEST.json"
TOOLCHAIN = REPO / "fortran" / "TOOLCHAIN.txt"

# Keys the capture tool embeds. Recorded separately from the array hashes so a
# regenerated-against-a-different-namelist golden reports as exactly that,
# rather than as an unexplained content change.
PROVENANCE_KEYS = ("_case", "_mode", "_variant", "_rows", "_namelist_sha256")


def hash_array(name: str, array: np.ndarray) -> str:
    """Hash an array's identity and contents together.

    Name, dtype and shape go into the digest alongside the bytes so that a
    renamed array, a widened dtype or a reshaped one is a change even when the
    underlying bytes happen to be identical -- which is exactly the case for an
    int32 column silently becoming int64 on another platform.
    """
    h = hashlib.sha256()
    h.update(name.encode())
    h.update(array.dtype.str.encode())
    h.update(repr(array.shape).encode())
    h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def describe(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            name: {
                "dtype": data[name].dtype.str,
                "shape": list(data[name].shape),
                "sha256": hash_array(name, data[name]),
            }
            for name in sorted(data.files)
        }
        provenance = {
            key: data[key].item() if data[key].dtype.kind != "U" else str(data[key])
            for key in PROVENANCE_KEYS
            if key in data.files
        }
    content = hashlib.sha256(
        "".join(f"{n}:{a['sha256']}" for n, a in arrays.items()).encode()
    ).hexdigest()
    return {"content_sha256": content, "provenance": provenance, "arrays": arrays}


def read_toolchain() -> dict:
    """The toolchain block, copied into the manifest at generation time.

    `fortran/TOOLCHAIN.txt` is a build product and is gitignored, so without
    this the committed goldens would carry no record of what produced them --
    and goldens are not portable across compilers or platforms.
    """
    if not TOOLCHAIN.is_file():
        return {}
    out = {}
    for line in TOOLCHAIN.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


def build(goldens: Path = GOLDENS) -> dict:
    return {
        "toolchain": read_toolchain(),
        "goldens": {p.name: describe(p) for p in sorted(goldens.glob("*.npz"))},
    }


def load(manifest: Path = MANIFEST) -> dict:
    if not manifest.is_file():
        return {"toolchain": {}, "goldens": {}}
    return json.loads(manifest.read_text(encoding="utf-8"))


def write(goldens: Path = GOLDENS, manifest: Path = MANIFEST) -> dict:
    data = build(goldens)
    goldens.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def verify(goldens: Path = GOLDENS, manifest: Path = MANIFEST) -> list[str]:
    """Return one message per problem; an empty list means the goldens are intact."""
    recorded = load(manifest)["goldens"]
    present = {p.name for p in goldens.glob("*.npz")}
    problems: list[str] = []

    for name in sorted(present - set(recorded)):
        problems.append(
            f"orphan: {name} is not in MANIFEST.json. "
            f"Run `python validation/goldens_manifest.py --write` if it is meant to be a golden."
        )
    for name in sorted(set(recorded) - present):
        problems.append(f"missing: MANIFEST.json lists {name} but it is not in {goldens}")

    for name in sorted(set(recorded) & present):
        actual = describe(goldens / name)
        expected = recorded[name]
        if actual["content_sha256"] == expected["content_sha256"]:
            continue
        problems.extend(_explain_drift(name, expected, actual))
    return problems


def _explain_drift(name: str, expected: dict, actual: dict) -> list[str]:
    """Name what moved. "the hash changed" is not an actionable message."""
    out = []
    exp_arrays, act_arrays = expected["arrays"], actual["arrays"]
    for key in sorted(set(exp_arrays) - set(act_arrays)):
        out.append(f"drift: {name} lost array {key!r}")
    for key in sorted(set(act_arrays) - set(exp_arrays)):
        out.append(f"drift: {name} gained array {key!r}")
    for key in sorted(set(exp_arrays) & set(act_arrays)):
        e, a = exp_arrays[key], act_arrays[key]
        if e["dtype"] != a["dtype"]:
            out.append(f"drift: {name}[{key}] dtype {e['dtype']} -> {a['dtype']}")
        elif e["shape"] != a["shape"]:
            out.append(f"drift: {name}[{key}] shape {e['shape']} -> {a['shape']}")
        elif e["sha256"] != a["sha256"]:
            out.append(f"drift: {name}[{key}] values changed")
    if expected["provenance"] != actual["provenance"]:
        out.append(
            f"drift: {name} provenance {expected['provenance']} -> {actual['provenance']} "
            f"-- regenerated from a different namelist or variant"
        )
    return out or [f"drift: {name} content hash changed"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="regenerate the manifest")
    parser.add_argument("--goldens", type=Path, default=GOLDENS)
    args = parser.parse_args(argv)

    manifest = args.goldens / "MANIFEST.json"
    if args.write:
        data = write(args.goldens, manifest)
        print(f"wrote {manifest} ({len(data['goldens'])} archive(s))")
        return 0

    problems = verify(args.goldens, manifest)
    for problem in problems:
        print(problem)
    if problems:
        print(f"\n{len(problems)} problem(s). A golden that moved is a finding, not a knob.")
        return 1
    print(f"{len(load(manifest)['goldens'])} archive(s) intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
