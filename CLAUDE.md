# GLOMAP-JAX — binding rules

These are not style preferences. Each one exists because breaking it produces a
result that looks plausible and is wrong.

## Precision

**float64 everywhere.** `jax_enable_x64` is set in exactly one place,
`src/glomap_jax/__init__.py`. Do not add redundant config calls, and never
disable it.

This is a correctness requirement, not accuracy tuning — but **not** for the
underflow reason this file gave for three reviews running. `1e-20` is a normal
float32 number, `1e-40` is subnormal but non-zero, and `ref-f32` and `ref-f64`
produce **identical** form-code histograms in the branch dump. See ADR-001; do
not reintroduce the underflow argument.

The measured reasons: the f32 reference is a *different trajectory*, not a
noisier one — for `marine_bcoc` the column-scaled f32-vs-f64 gap is **0.80**
(issue #14) — and even away from that the floor is **3.7e-4**, four orders
above any tolerance worth gating on.

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
NaN cotangents under reverse-mode AD.
`test_numerics.py::test_safe_divide_masks_without_poisoning_the_gradient`
guards this. (It was cited here as `test_grad_finite`, which has never
existed.)

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
the answer.

None of the five is ported yet, so none has such a test. **Each must get one
asserting the broadcast version *differs*** — written as part of the port, not
after it, since a scan that was never wrong is a scan nobody can show is
needed. This paragraph previously stated that guarantee in the present tense.

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


## Repository layout

```
src/glomap_jax/
  config/     model.py (what to run) + fidelity.py (how faithfully)
  core/       state.py, numerics.py, constants.py — machinery every process
              needs and none of them owns
  physics/    one module per UKCA routine, ported in the order the Fortran
              forces (see its docstring — it is not the naive order)
  drivers/    eager + scan timestepping, which must agree to RTOL_JIT_VS_EAGER
  utils/      helpers with no physics in them
fortran/      vendored UKCA, READ-ONLY. src/ukca/ is Crown Copyright.
inputs/       namelists this repo added; the shipped three stay in fortran/
outputs/      run scratch, gitignored. NOTHING here is a golden.
tests/        including goldens/ + MANIFEST.json — the committed reference
validation/   the harness: build scripts, overlays, f2py binding, capture
docs/         harness.md is the map; porting-notes.md has every measurement
benchmarks/   throughput, orders 2-3 (not created yet)
figures/      generated plots, gitignored by default
```

`validation/` is what `docs/harness.md` calls the harness. Not renamed to
`harness/` because ~30 references point at it and the gain is cosmetic.

## No hard-coded values

Every physical constant and numerical threshold lives in
`core/constants.py`, and `tests/test_constants.py` **re-parses the vendored
Fortran and compares**. Do not type a constant from memory into a physics
module; import it.

The reason is arithmetic, not tidiness. UKCA carries `avogadro = 6.022e23`
where CODATA says `6.02214076e23` — 2.3e-5 relative, eight orders of magnitude
above `RTOL_ALGEBRAIC`. "Correcting" it invalidates every golden downstream of
a concentration conversion, and the failure surfaces in whichever routine
happens to use it first.

Derived quantities are **not** cached there. `mm_da = avogadro*boltzmann/rgas`
is computed where it is used; a derived value in a constants table is a second
source of truth.

## Numerics: three rules, all measured

`core/numerics.py` is not a convenience wrapper. Three primitives must be
written a specific way, and the obvious way is silently wrong — wrong as in
flipping a branch and moving a trajectory by O(1), not losing a digit.

Measured over 15,382 points against the Fortran itself
(`validation/capture_leaf.py`, asserted by `tests/test_numerics*.py`):

| | JAX vs gfortran |
|---|---|
| `erf`, `log`, `1/x`, `x**(1/3)` | **bit-identical on the capture platform** (arm64); `erf` and the powers drift ≤2 ulp on x86_64 |
| `exp` | 456/3199 differ, max 2.1e-16 — 1 ulp, inside tolerance |
| `jnp.cbrt` | 1763/1865 differ on arm64, 1793 on x86_64, max 1.3e-14 — **do not use** |
| `jnp.round` | 64 of 129 ties differ — **do not use** |

**Cube root is `x ** (1.0/3.0)`, never `jnp.cbrt`.** `cubrt_v` is literally that
expression. `cbrt` is a genuinely better cube root and returns a real root for
negatives where the power form gives NaN — which is exactly why it cannot be
the faithful path. Its output is `drydp`, compared against `dp_thresh1` (merge
or not) and `ddplim0*0.1` (rewrite `md`/`mdt` or not); both are step changes, so
1.3e-14 flips them. Available as `FidelityConfig.cbrt_exact`, default `False`.

**Rounding is `numerics.nint`, never `jnp.round`.** Fortran `NINT` rounds half
away from zero. The live consumer is `ukca_vapour.F90:226`, `(NINT(wts/5))*5`,
whose result **indexes a lookup table** — a tie that rounds the other way picks
a different entry, not a nearby number. Note `sign(x)*floor(|x|+0.5)` is also
wrong, at `x = ±0.49999999999999994`.

**`0.0 * inf` is `NaN`.** Mask before reducing — `numerics.masked_sum`, i.e.
`jnp.where(mask, term, 0.0)`, never `mask * term`. Bites at
`ukca_ageing.F90:308`, an unmasked whole-array `SUM` gating a transfer block.

**Divide with `numerics.safe_divide`.** Single-`where` division gives a NaN
cotangent; reverse mode differentiates the branch not taken.

**XLA flushes subnormal arithmetic results to zero**, gfortran does not. Latent
(`num_eps` bottoms at 1e-20, `eps_d` at 1e-40, both normal in float64) but it
would present as a gate-0 disagreement with no arithmetic explanation. Issue
#15. Re-measure on CUDA.

## What the harness can and cannot tell you

Four gates, none subsuming another — `docs/harness.md` has the full map.

* **Gate 0** — did the same *predicate* go the same way? Catches nothing that
  is not a branch.
* **Gate A** — does one *routine* agree at machine precision? In-process f2py.
  Says nothing about sequencing. Currently compares 15 of 39 columns on one row
  of one case (#17), and dies on 19 of 20 error paths (#16).
* **Gate B** — which *call* diverged? Per-process dumps, keyed
  `(step, seq, imts, izts)`. `seq` is load-bearing: without it the key is not
  unique, because `drydiam` and `volume_mode` each run twice per `imts`.
* **Gate C** — does the *run* agree? Committed goldens. Cannot say where.

Setups **1, 2, 3, 4, 5, 6, 8** only. Mode slots **1–7**; slot 8
`mode_sup_insol` needs setup 12 or 13, which the box model does not implement.
`modesol = [1,1,1,1,0,0,0,0]` in every setup — the soluble/insoluble split is
structural; only `mode_choice` varies.

## Two failure modes this repo actually has

Both have recurred across every review so far. Assume they are present now.

**Tests that cannot fail.** Nine found so far, across two reviews — one with no
assertion at all, one asserting `float(f"{v:.16E}") == v` (true of every
double), one substring-matching a script that names no source file. Before
writing a test, name the mutation that would fail it; then apply that mutation
and check it does.

**Claims that do not survive checking.** Twenty-one wrong statements found in
one review, including an ADR still carrying a justification an earlier review
had already recorded as false, and a fidelity flag documented backwards. If a
doc states a number, it must be reproducible from committed data.

## Working practice

One commit per task, each with an acceptance criterion, repo green at every
commit. Problems become GitHub issues, not paragraphs. Each phase closes with an
adversarial agent review of its diff against the Fortran; findings become issues
before the phase is closed.
