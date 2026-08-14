# Key decisions (ADRs)

Decisions with a rationale, so they are not re-litigated. Each states what was
decided, why, and what evidence would change it.

## ADR-001 — float64 only; float32 is not a supported trajectory mode

**Decided.** `jax_enable_x64` is set once, in `src/glomap_jax/__init__.py`.

This is correctness, not accuracy. `ukca_solvecoagnucl_v` selects among five
closed-form solutions by comparing a discriminant against `eps_d = 1e-40`, and
mode `num_eps` values reach `1e-20`. float32 flushes both to zero and **a
different branch executes**. `carma-jax` reached the same conclusion
independently for its own `SMALL_PC = 1e-50`.

float32 is permitted only as an explicitly labelled benchmark mode.

## ADR-002 — one compiled kernel for all seven mode setups

**Decided**, with a caveat to resolve in phase C.

`nmodes = 8` and `ncp_max = 10` are Fortran PARAMETERs, and all seven supported
setups have `ncp = 6`, so arrays are allocated at full extent and masked. What
varies is `nchemg`, `nadvg` and `nbudaer`; those are padded to their maxima.

**Open:** the 283 `nmas*` budget index scalars are per-setup integers. Either
they are static config — which means recompiling per setup, contradicting the
above — or they are traced, making every budget write a dynamic scatter.
Recommendation is traced, since budgets are diagnostics and one kernel is worth
more than a marginally faster scatter. To be settled with a measurement in
phase C.

## ADR-003 — `jax-metal` is out of scope

**Decided.** Incomplete op coverage and weak-to-absent float64, which ADR-001
rules out on its own. Order 3 targets NVIDIA CUDA on cloud or cluster hardware,
and specifically datacenter parts: float64 is ½ FP32 rate on A100/H100/V100 but
**1/32 on consumer RTX 30xx/40xx**, so consumer-card numbers would be
meaningless. Tracked as issue #4.

## ADR-004 — diffrax is order 2 only, scoped to the `nzts` interior

**Decided.** The faithful path uses no ODE library.

`ukca_solvecoagnucl_v` is an *exact closed-form integral*, not a numerical step,
so a solver would be slower and less accurate. `aer3d` and `vbs-jax`, the two
newest faithful ports in this family, use no ODE library at all; `tomas-jax`
abandoned `Tsit5` for forward Euler because "higher-order methods amplify N²
coagulation rates", which is exactly this system; and `mam4-jax` had to relax
its acceptance bar from `1e-6` to "3% over 24h" to adopt diffrax.

Where diffrax *does* fit: inside the `nzts` loop, sizes and coagulation kernels
are never updated, so that block is a genuine constant-coefficient ODE. It
cannot absorb the `nmts` boundary, where `ukca_remode` is a discontinuous remap
and the `deltas0g` reconciliation is a projection.

**Would change this:** a diffrax configuration that matches the faithful path to
`RTOL_STEP` on the shipped cases.

## ADR-005 — goldens are generated once on a pinned toolchain

**Decided.** `make goldens` is never run in CI.

The Fortran reference is not bit-reproducible across platforms: the vendored
Makefile leaves `-ffp-contract` at gfortran's default `fast`, so FMA contraction
differs by architecture and optimisation level, and `EXP`/`LOG`/`ERF` come from
the platform libm. A CI job regenerating goldens would fail the drift gate on
the first PR from another machine, and the gate would then get loosened — which
the tolerance policy forbids.

Goldens are committed with the compiler version, flags (including
`-ffp-contract=off`), OS and architecture recorded in the manifest.

## ADR-006 — consistency routines are mostly not ported

**Decided.** `ukca_check_md_nd` declares its state arguments `INTENT(IN)` and
only prints, so omitting it is exact. `ukca_mode_check_artefacts` has no caller
in the box model. `ukca_mode_check_mdt` is gated on `iextra_checks > 1`, which
the box model leaves at 0, and it mutates state, so it raises rather than being
silently skipped.

The one always-active state mutation, `ukca_calc_drydiam`'s undersize reset, IS
ported — see `drydiam_undersize_reset` in `fidelity.md`.

## ADR-007 — goldens are committed as plain files; no Git LFS

**Decided.** `tests/goldens/*.npz` go into the repository directly.

The concern was real before it was measured: 283 budgets × 48 steps × several
namelists × 13 hook points is hundreds of megabytes captured naively, and the
per-substep state dump is the largest stream by an order of magnitude. So the
question was settled with numbers rather than a guess.

Measured, at the namelists' own 48 steps:

| | one case, as CSV | full set, as `.npz` |
|---|---|---|
| trajectory | 0.05 MB | |
| budgets | 0.13 MB | |
| state | 15.90 MB | |
| branches | 3.82 MB | |
| **total** | **19.89 MB** per case | **0.83 MB** for all four cases and all four modes |

A factor of about 96. Three things do the work, and all three are properties of
this data rather than of compression in general: the long-format dumps repeat a
small vocabulary of `site`, `field` and `tag` labels over hundreds of thousands
of rows, so factorising them into integer codes removes most of the bytes before
zlib sees them; branch values are 0/1 and 0–7 and are stored as `int8`; and the
state dump's float64 values contain long runs of exact zeros for inactive modes
and components.

LFS would buy nothing at this size and cost a great deal: contributors need a
`git lfs install` step, CI needs LFS quota, and a shallow clone without LFS
gives pointer files that fail as *corrupt fixtures* rather than as missing ones
— the worst possible failure mode for a validation gate.

**Would change this:** any single archive above 5 MB, or the set above 25 MB.
Both are asserted in `tests/test_goldens_manifest.py`, so the decision is
re-opened by a failing test rather than by someone noticing. The likely trigger
is not more namelists — it is capturing a multi-box case, where every stream
scales with `nbox`.
