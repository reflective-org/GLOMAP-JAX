# Provenance of the vendored Fortran

`fortran/` is the reference implementation this port is validated against. It is
kept as close to upstream as possible so that any disagreement between JAX and
Fortran is attributable to the port, never to a local edit.

| item | value |
|---|---|
| ultimate upstream | `https://github.com/MetOffice/ukca` |
| upstream commit | `387c5bb0f1166e67f029930ba624bf159bc68627` |
| intermediate | `https://github.com/reflective-org/glomap-box` @ `2befe04` |
| UKCA files vendored | 46 (44 byte-identical to upstream, 2 patched) |
| licence | BSD 3-Clause, Crown Copyright Met Office (see `LICENCE`) |

## What is and is not editable

**`fortran/src/ukca/` is read-only.** These are the 46 UKCA science and
infrastructure files. They are never edited in place; the only permitted
divergence is a patch in `fortran/patches/`, each with a written rationale and a
demonstration that release-build output is unchanged. `tests/test_vendored_tree.py`
enforces this with a content hash.

**`fortran/src/box/` may be extended.** The box driver is new BSD-3 code written
for `glomap-box`, not Crown Copyright, so the reference harness lives here: the
high-precision state dump, the budget dump, the branch-mask dump, and the
multi-box benchmark driver. Extensions arrive as overlays in
`validation/patches/` so the committed tree stays comparable with `glomap-box`.

## Applied patches

Two upstream defects, both verified to leave release-build results unchanged:

* **`0001-guard-msec_org-zero-index.patch`** — `ukca_aero_step.F90` indexes
  `condensable_choice(msec_org)` and `mm_gas(msec_org)` three lines before the
  `IF (msec_org > 0)` guard that protects them. `msec_org = 0` is legitimate for
  every mode setup without a secondary-organic component, including the default
  `i_mode_setup = 1`, so those reads are out of bounds. Inert when optimised (the
  value is discarded) but fatal under bounds checking, which makes debug builds
  unusable for the default configuration.
* **`0002-ereport-nonzero-exit-status.patch`** — `ereport_mod.F90` terminates a
  fatal error with a bare `STOP`, which exits with status **0**, so every UKCA
  abort looked like success to the shell. Only reachable on the abort path.

Both are reported upstream — see `docs/UPSTREAM_DEFECTS.md` and issue #1.

## Refreshing against a newer UKCA

1. Update `glomap-box` first and re-run its `make verify-vendor`.
2. Re-vendor `fortran/` from it and update the commit hashes above.
3. Re-apply `patches/`, dropping any upstream has since fixed.
4. Regenerate goldens on the pinned toolchain (see `docs/REFERENCE_BUILD.md`) —
   they are not portable across compilers or platforms.
5. `make test-all`.
