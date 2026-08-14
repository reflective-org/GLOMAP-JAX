# GLOMAP-JAX — binding rules

These are not style preferences. Each one exists because breaking it produces a
result that looks plausible and is wrong.

## Precision

**float64 everywhere.** `jax_enable_x64` is set in exactly one place,
`src/glomap_jax/__init__.py`. Do not add redundant config calls, and never
disable it.

This is a correctness requirement, not accuracy tuning. `ukca_solvecoagnucl_v`
selects among five closed-form solutions by comparing a discriminant against
`eps_d = 1e-40`, and mode `num_eps` values reach `1e-20`. In float32 both
underflow to zero and **a different branch runs** — a different answer, not a
noisier one.

float32 is permitted only as an explicitly labelled benchmark mode, never as a
validated trajectory.

## Fidelity to the Fortran

**Port from the code, not the comments.** Several headers contradict their own
implementation — `ukca_solvecoagnucl_v` swaps arctan and log relative to what it
computes, and `ukca_conden`'s header claims `se_ins = 0.3` while the live value
is `1.0`. The code is the specification.

**Every upstream quirk gets a `FidelityConfig` flag whose default reproduces the
Fortran**, plus a `docs/fidelity.md` entry and a test at both settings. Getting a
default backwards silently changes results. "Obviously a bug, so I fixed it" is
how a port stops being a port — see `ukca_ageing`'s `naged` no-op, where the
naive fix loses mass.

**`fortran/src/ukca/` is read-only.** Divergence only via `fortran/patches/`,
each with a rationale and a demonstration that release output is unchanged.
`fortran/src/box/` is new BSD-3 code and may be extended.

## JAX

**No Python branching on traced values.** Use `jnp.where`. `lax.cond` is rare
here and `lax.select` unused.

**Every guarded division needs the double-where idiom:**

```python
safe_den = jnp.where(cond, den, 1.0)
out = jnp.where(cond, num / safe_den, 0.0)
```

A single `where` around a division still evaluates the unsafe branch and gives
NaN cotangents under reverse-mode AD. `test_grad_finite` guards this.

**Mask before reducing, never multiply.** Use `jnp.where(mask, term, 0.0)` ahead
of a sum, not `mask * term`: several GLOMAP quantities are evaluated over the
full array extent and `0.0 * inf = NaN`.

**Port first, `jit` second, in separate commits.** float64, no `jit`, no `vmap`;
validate against the reference; only then optimise. The eager driver stays
permanently — it is the debugger, because it can raise real exceptions.

**Five routines need a sequential `lax.scan`, not `vmap`**, because they are
loop-carried: `ukca_remode` over modes, `ukca_ageing` over modes (7→4 and 8→4
collide) and over `jv`, `ukca_coagwithnucl`'s `icp` loop, and
`ukca_water_content_v`'s 12 ion pairs. "Compute all deltas, then apply" changes
the answer. Each has a test asserting the broadcast version *differs*, so the
scan cannot be silently regressed.

## Validation

**No physics commit before the reference that validates it.** This is the
ordering rule most likely to break under time pressure, and breaking it means
writing code with no way to know whether it is right.

**Tolerances live in `tests/conftest.py`.** Loosening one to make a test pass is
a finding to investigate, not a knob to turn. Per-test overrides are
review-blocking.

**Goldens are generated once on a pinned toolchain and committed.** The Fortran
is not bit-reproducible across platforms: `-ffp-contract` defaults to `fast`, so
FMA contraction differs by architecture, and `EXP`/`LOG`/`ERF` come from the
platform libm. `make goldens` is never run in CI.

**Fortran-dependent tests carry the `fortran` marker** and skip cleanly when the
toolchain is absent. CI has no gfortran.

## Failing loudly

**Never skip or hide a failure.** A crashed run must fail the suite, not produce
an empty figure or a silently skipped scenario. The harness self-check exists
because the suite once reported `passed: 8 failed: 0` against a binary that never
ran.

**Surface caps and clamps.** If a substep count is capped or a value clamped,
report it. Silent truncation reads as success.

## Working practice

One commit per task, each with an acceptance criterion, repo green at every
commit. Problems become GitHub issues, not paragraphs. Each phase closes with an
adversarial agent review of its diff against the Fortran; findings become issues
before the phase is closed.
