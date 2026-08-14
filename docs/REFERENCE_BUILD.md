# Building the Fortran reference

`validation/build_reference.sh [f32|f64|both]` produces the reference binaries
the JAX port is validated against.

## Two precisions, and why both

| variant | flags | role |
|---|---|---|
| `ref-f32` | as shipped (default `REAL` is kind=4) | diagnostic only |
| `ref-f64` | `-fdefault-real-8` | **the actual reference** |

The shipped box model runs in single precision. A float64 JAX port compared
against it disagrees for reasons that are not bugs, and the size of that
disagreement is not small: **measured 3.7×10⁻⁴** over a 48-step boundary-layer
run. That is four orders of magnitude above any tolerance worth gating on, so
`ref-f32` cannot serve as a validation target. It survives as the measurement
that explains why `ref-f64` is required.

## Staged builds

The reference is built from a **copy** of `fortran/`, never from `fortran/`
itself:

```
fortran/  --copy-->  .refstage/<variant>/  --apply validation/patches/*-->  build
```

This keeps `fortran/` byte-comparable with `glomap-box` and hash-checkable by
`tests/test_vendored_tree.py`, while still allowing the instrumentation the
harness needs. Only `src/box/` is ever patched — that is new BSD-3 code, not
Crown Copyright UKCA. The staging step verifies `src/ukca/` is untouched in the
stage as well, and the script verifies the working tree is unmodified before it
exits.

An overlay that fails to apply is a hard error, not a warning: a silently
skipped overlay would produce a reference that looks right and is missing its
instrumentation.

## Overlays

| patch | purpose |
|---|---|
| `0001-high-precision-output.patch` | `ES14.6` → `ES24.16` |

`ES14.6` carries 7 significant digits. The port is gated at `RTOL_STEP = 1e-11`
and `RTOL_ALGEBRAIC = 1e-13`, so without this a double-precision reference is
truncated at output to the same 7 digits as the single-precision one and Gate C
cannot be met at any useful tolerance. With the overlay the reference carries 17.

## Reproducibility — read this before trusting a golden

**Goldens are not portable across compilers or platforms.** Two reasons:

1. **FMA contraction.** gfortran defaults to `-ffp-contract=fast`, which
   contracts `a*b+c` into a fused multiply-add on aarch64 but not on older x86,
   and differently at `-O0` than `-O2`. Every build here passes
   `-ffp-contract=off`, which makes the reference stable across optimisation
   levels on one machine. It does **not** make it portable.
2. **libm.** `EXP`, `LOG`, `ERF` and `x**y` come from the platform maths
   library, and glibc, Apple libm and a Homebrew gfortran's libm do not agree to
   the last bit.

So: goldens are generated **once**, on a pinned toolchain, and committed.
`fortran/TOOLCHAIN.txt` records the compiler version, flags, OS and
architecture used. `make goldens` is deliberately **never** run in CI — a job
that regenerated them would fail the drift gate on the first PR from a different
machine, and the gate would then get loosened, which the tolerance policy
forbids. See ADR-005.

A CI job that rebuilds the Fortran must compare at tolerance, never by hash.

## Requirements

`gfortran` (16.1.0 on the pinned toolchain). For the in-process f2py comparison
(Gate A) also `meson` and `ninja`, both in the `dev` extra — and note
`numpy.f2py -c` shells out to `meson` **by name** through `PATH`, so
`.venv/bin` must be on `PATH`, not merely used to invoke Python.
