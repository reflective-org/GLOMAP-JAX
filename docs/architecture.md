# Architecture

How the port is laid out, and why. The short version: **configuration is
static, state is traced, and the reference decides everything else.**

## Layout

```
src/glomap_jax/
  config/     model.py    — what to run: setup, process switches, timestep
              fidelity.py — how faithfully: one flag per upstream defect
  core/       state.py     — traced pytrees carried through a step
              numerics.py  — the gfortran-compatibility primitives
              constants.py — physical constants, extracted from the Fortran
  physics/    one module per UKCA routine
  drivers/    eager + scan timestepping
  utils/      helpers with no physics in them
```

Four top-level directories carry data rather than code: `inputs/` (namelists
this repo added), `outputs/` (run scratch, gitignored — nothing here is a
golden), `tests/goldens/` (the committed reference, manifest-guarded), and
`figures/`. `fortran/` is the vendored reference and is read-only;
`validation/` is the harness that builds and captures from it.

## Static config, traced state

`ModelConfig` and `FidelityConfig` are `@dataclass(frozen=True)`, so they are
hashable and can be passed to `jax.jit` as static arguments. Every switch then
becomes a compile-time Python branch with no runtime cost, and no traced
`cond`. The price is one compilation per distinct configuration, which is the
right trade for a model run at one configuration for thousands of steps.

State is `NamedTuple`, which JAX registers as a pytree without ceremony. No
`flax.struct`, no `chex`, no manual registration — matching every sibling port
in this tree.

The split is not cosmetic. If a value can change during a run it is state; if
it cannot it is config. `i_mode_setup` is config; `nd` is state. The mode
tables are config *derived from* config, which is why `physics/modes.py`
returns a frozen dataclass rather than a pytree.

## Two kinds of configuration flag

`ModelConfig` is what to run. `FidelityConfig` is where the port deliberately
reproduces upstream behaviour that is wrong.

The test for which one a switch belongs to: **does it select between a defect
and its correction?** `l_fix_nacl_density` does — 1600 kg m⁻³ against the
correct 2165 — so it is a fidelity flag. `l_radaer`, `i_tune_bc` and
`l_dust_mp_ageing` change the mode tables without any of them being wrong, so
they are model configuration.

Every fidelity flag defaults to **reproducing the Fortran**, with one
instructive exception: `l_fix_nacl_density` defaults `True`, because the box
model sets it on and the reference this port is validated against therefore
uses the corrected density. "Reproduce the reference" and "reproduce the
literal in the source" are not always the same instruction.

## Faithfulness is a layering constraint, not a style

`core/numerics.py` exists because three primitives must be written a specific
way to match gfortran and the obvious way is silently wrong. It does not import
`config`: a caller passes `exact=fidelity.cbrt_exact` rather than the primitive
reaching for a flag. Primitives stay dependency-free so they can be tested
against the reference in isolation.

`core/constants.py` holds no derived quantities. `mm_da = avogadro·boltzmann/rgas`
is computed where it is used, because a derived value in a constants table is a
second source of truth that can drift from the first.

Both are machine-checked against the vendored Fortran — `constants.py` by
re-parsing, `physics/_mode_literals.py` by re-extraction. Several hundred
numbers typed by hand is how a wrong model comes to look plausible.

## What "ported" means here

A module is ported when it reproduces the reference **byte for byte** on the
committed goldens, with derived quantities recomputed rather than copied.
Copying a derived table out of the golden produces a module that passes every
test and implements nothing.

Byte equality rather than a tolerance because this code branches on computed
floats. `drydp` is compared against `dp_thresh1` and `ddplim0·0.1`, both step
changes, so one ulp does not stay one ulp — it flips a branch and the
trajectories separate by O(1). A tolerance at this layer defers the failure to
somewhere it cannot be diagnosed.

Getting the mode tables byte-equal took four corrections, none of them a
mistake about the physics:

| | what differed |
|---|---|
| `d**3` vs `d*d*d` | gfortran expands an integer literal exponent; numpy calls `pow()` |
| factor order | reassociating a product of four terms changes the last bit |
| switch order | a density patched after the masses derive from it, not before |
| `.AND.` read as two knobs | a nested condition treated as independent |

Each is pinned as its own test with a guard that the hazard still exists, so a
later simplification fails with a reason rather than an anonymous mismatch.

## Where the boundaries are

`validation/` may read `fortran/` and write `tests/goldens/`. `src/` may read
neither: the port never touches the reference at runtime, and a test that
compares them loads a committed archive rather than running Fortran. That is
what lets the whole suite run in CI, where there is no gfortran.

The f2py binding is the one place the two meet, and it lives in `validation/`
for exactly that reason.
