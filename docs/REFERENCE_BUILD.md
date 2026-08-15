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
harness needs. The script verifies the working tree is unmodified before it
exits.

`src/box/` is new BSD-3 code and overlays may edit it freely. `src/ukca/` is
Crown Copyright UKCA, and per-process validation is impossible without reaching
into it — the dumps at tasks 15 and 15b need hooks at call sites and inside the
science routines themselves. The rule that keeps that safe is mechanical rather
than declared: **an overlay touching `src/ukca/` may only ADD lines**, checked
on the patch text itself so it holds regardless of what the patch claims about
itself. A removal means a science line was altered or deleted, which is a
science change wearing instrumentation's clothes, and the build rejects it.

An overlay that fails to apply is a hard error, not a warning: a silently
skipped overlay would produce a reference that looks right and is missing its
instrumentation.

## Overlays

Overlays are **ordered**. Each is generated against a stage with the earlier
ones already applied, so a new one must be produced the same way or it will not
apply.

| patch | purpose | namelist key |
|---|---|---|
| `0001-high-precision-output.patch` | `ES14.6` → `ES24.16` | — |
| `0002-dump-budgets.patch` | the 283 per-process mass fluxes in `bud_aer_mas` | `budget_file` |
| `0003-dump-state.patch` | state snapshot after each of the 13 process calls | `state_file` |
| `0004-dump-branches.patch` | the predicates the science branches on (gate 0) | `branch_file` |
| `0005-unique-state-keys.patch` | per-step call counter + the gas-phase fields | `state_file` |

Each dump is written only when its key is set to a non-empty path in
`&box_run`, so the reference binary is a single build regardless of which
instrumentation a given run wants.

On `0001`: `ES14.6` carries 7 significant digits. The port is gated at
`RTOL_STEP = 1e-11` and `RTOL_ALGEBRAIC = 1e-13`, so without this a
double-precision reference is truncated at output to the same 7 digits as the
single-precision one and Gate C cannot be met at any useful tolerance. With the
overlay the reference carries 17.

On `0004`: this code diverges by flipped predicates, not by precision drift.
About ten sites compare a computed float against a threshold and then select a
*different closed form*, so a disagreement there is O(1) between two
individually correct float64 implementations and no trajectory tolerance can
attribute it. `branch_file` records, per box and per substep, which mask each
site produced and — for `ukca_solvecoagnucl_v`, where the branch is a five-way
select rather than a single test — an integer naming the form that ran. See
`docs/porting-notes.md` for what the shipped fixtures do and do not reach.

## Capturing goldens

```sh
./validation/build_reference.sh both          # once
python validation/capture_reference.py --dry-run
python validation/capture_reference.py        # writes tests/goldens/*.npz
```

`--dry-run` prints the capture matrix and needs no toolchain, so the plan is
inspectable before anything is built. The matrix is deliberately **not** a cross
product of cases × modes × variants: `ref-f32` exists only to measure the
precision floor against `ref-f64`, and that floor is a property of the
trajectory, so the trajectory is captured in both variants and the three dumps
in `f64` only. `--mode`, `--case`, `--variant` and `--steps` narrow it;
`--steps` in particular gives a cheap smoke capture without a 24-hour run.

Each archive is one `.npz` named `<case>.<variant>.<mode>.npz`. The two wide
streams store a float64 table plus its column names; the two long-format streams
store integer columns with their string labels factorised into a codebook, and
branch values as `int8`. Every archive carries `_case`, `_mode`, `_variant`,
`_rows` and a SHA-256 of the exact namelist that produced it, so a stale golden
is distinguishable from a regenerated one.

**The full golden set is 0.80 MB** across all four cases and all four modes —
roughly 90× smaller than the CSV the reference emits, mostly because the state
dump's 318k rows per case compress to ~0.15 MB once the labels are codes. Git
LFS is not needed at this size (task 18 records the decision).

## The drift / orphan gate

```sh
python validation/goldens_manifest.py --check    # exit 1 on any problem
python validation/goldens_manifest.py --write    # re-bless, deliberately
make goldens                                     # build + capture + write
```

`tests/goldens/MANIFEST.json` records every array's **name, dtype, shape and
content hash**, the provenance keys the capture tool embeds, and the toolchain
block from `TOOLCHAIN.txt` — which is a gitignored build product, so copying it
into the manifest is the only way the committed goldens carry a record of what
produced them. `tests/test_goldens_manifest.py` runs the check in CI.

Three problems are reported separately, because they mean different things:

| | meaning |
|---|---|
| `drift` | a listed archive's contents changed — investigate, do not re-bless |
| `orphan` | an archive exists that nothing lists — captured and forgotten |
| `missing` | a listed archive is gone — partial checkout, or a deleted fixture |

Drift messages name *what* moved (`a.npz[values] values changed`,
`a.npz[step] dtype <i4 -> <i8`, or a provenance change meaning the golden was
regenerated from a different namelist), because "the hash changed" is not
something anyone can act on.

**Array contents are hashed, not the `.npz` file.** `np.savez_compressed` writes
a zip and zip entries carry timestamps, so the same data written twice gives two
different file hashes. A file-level hash would fail on every regeneration and be
loosened away within a week.

Capture does **not** update the manifest. Auto-blessing a capture would make the
gate silent the one time it matters, so `--write` is always an explicit act.

## The in-process binding (gate A)

```sh
./validation/build_f2py.sh          # needs meson + ninja on PATH
```

Gate A is the only mechanism here that reaches machine precision: it calls the
Fortran in-process, so a routine can be driven with chosen inputs and read back
at full double precision, with no text file and no accumulated trajectory in
between. The build is two stages, and the order is not negotiable — the
vendored Makefile builds the 46 sources first, then f2py wraps a **single**
self-contained file against those `.mod`s. Handing f2py all 46 fails on the
first `USE` of a module it has not compiled yet; it has no notion of module
dependency order. (The TOMAS precedent does not transfer: three fixed-form F77
files with COMMON blocks and no modules.)

Four things that cost time to learn, recorded so they are not relearned:

| | |
|---|---|
| **`meson` is invoked by name** | `numpy.f2py -c` shells out through `PATH`; using `.venv/bin/python` is not enough |
| **module-level derived types abort the build** | f2py exposes *every* module variable, and `TYPE(box_env_type)` has no C mapping → `KeyError: 'void'`. The state lives in a module f2py is never handed, reached by `USE` from file-scope subroutines |
| **`REAL(KIND=8)`, never bare `REAL`** | f2py maps the *token* `real` to C float whatever `-fdefault-real-8` says, silently feeding float32 buffers into real(8) dummies. Being explicit removes the need for an `.f2py_f2cmap` |
| **setter signatures are asymmetric** | f2py infers array dimensions from input arrays and no directive stops it (`required` is ignored). Getters keep their explicit sizes because `INTENT(OUT)` gives nothing to infer from |

That last one matters more than it looks. f2py only guarantees `n1`/`n2` agree
with the array it was handed — it knows nothing about `nmodes`, so a
wrong-width array arrives as a self-consistent pair and reaches Fortran
unchallenged. The comparison against the module's own sizes inside each
accessor is the only real check.

**The binding reproduces the committed goldens bit for bit, as far as it is
compared.** The assertion is exact — `==`, not a tolerance — but it covers one
chemistry step of one case (`marine_bcoc`) on `nd`, `drydp` and `rhopar`, i.e.
15 of the trajectory's 39 columns and one of its 49 rows. Widening it to the
mass and gas columns and to multiple steps is worth doing before phase C leans
on it. It is built from
the plain vendored tree while the goldens came from the fully patched stage, so
that agreement says three things at once: the wrapper's transcription of the
driver is faithful, the `ES24.16` overlay round-trips float64 without loss, and
the overlays really are instrumentation rather than science.

**One process per setup.** `ukca_mode_setup` allocates under
`IF (.NOT. ALLOCATED)` and never deallocates, and the 283 `nmas*` budget
indices have no initialiser, so a second `init_ukca_for_box` leaves stale
indices — and since `nbudaer` changes too (8 vs 138) a stale index can be out
of bounds. `wrap_init` refuses a different setup, and because
`read_box_namelist` must run before the setup is even knowable, a refusal also
poisons the process: the config scalars have already been replaced, leaving the
new switches paired with the old mode setup. Every entry point then returns 1
until the process restarts. `tests/test_f2py_binding.py` runs each case in a
subprocess accordingly.

**The `ereport` shim.** `glomap_ereport_shim.F90` replaces
`src/ukca/ereport_mod.F90` in this extension only, so a fatal error records
itself and returns instead of stopping the interpreter. Same module name and
signature, so already-compiled callers link against it unchanged. The reference
build never sees it. `wrap_init` and `wrap_step` return `ierr = 5` if a fatal
fired during the call — see `docs/harness.md` for why that check is not
optional.

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
