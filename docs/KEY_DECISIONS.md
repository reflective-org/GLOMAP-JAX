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
