# The validation harness

Four gates, each answering a question the others cannot. They are not
redundant, and none of them subsumes another — which is the point of writing
them down together.

| gate | question | mechanism | status |
|---|---|---|---|
| **0 — branch** | did the same *predicate* go the same way? | `branch_file` dump | built (task 15b) |
| **A — in-process** | does one *routine* agree at machine precision? | f2py binding | built (task 20/21) |
| **B — per-process** | which *call* did the two sides start to disagree at? | `budget_file`, `state_file` dumps | built (tasks 14/15) |
| **C — trajectory** | does the *run* agree over time? | committed `.npz` goldens | built (task 19) |

Nothing in the port exists yet, so all four currently gate the reference
against itself. That is not vacuous — it is how the harness was debugged, and
three of the findings in [porting notes](porting-notes.md) came out of it before
a line of JAX was written.

## Gate 0 — branch agreement

The gate the plan did not originally have, and the highest-value one.

GLOMAP-mode does not diverge by precision drift. It diverges by **flipped
predicates**: roughly ten sites compare a computed float against a threshold and
then select a *different closed form*. Two individually correct float64
implementations that disagree at one of those comparisons produce an O(1)
trajectory difference, and no tolerance on the trajectory can say which
comparison it was.

So the reference dumps the predicates themselves — every mask, per box, per
substep, plus an integer branch code for `ukca_solvecoagnucl_v` where the choice
is a five-way select rather than a single test. A trajectory that then diverges
is attributable to a specific flipped predicate rather than to "precision".

**What it cannot catch:** anything that is not a branch. Two implementations can
agree on every predicate and still differ in the arithmetic between them.

**Already earned its keep**, in both directions:

* it *confirmed* UP-1 fires every substep of every shipped namelist, for the top
  soluble mode, in the default setup — stronger than the defect note claimed;
* it *excluded* a branch explanation for the `marine_bcoc` f32 divergence
  (issue #14), where 60 of 108,432 records differ and all of them start 25 steps
  after the trajectory does. Ruling a hypothesis out is worth as much as
  confirming one;
* it *measured* what the shipped fixtures never reach — half of
  `solvecoagnucl_v`'s branches, every mode merge, the undersize reset. That is
  now issues #12 and #13 rather than a surprise in phase I.

## Gate A — in-process, per routine

The only mechanism here that reaches machine precision. `validation/f2py/`
builds an extension module that calls the vendored Fortran directly, so a
routine can be driven with **chosen inputs** and read back at full double
precision — no text file, no accumulated trajectory.

Two shapes:

* **whole-step** (`wrap_init`, `wrap_step`, accessors) — reproduces the
  committed trajectory goldens *bit for bit* — exactly, not at a tolerance —
  which simultaneously validates the binding, the `ES24.16` overlay's
  round-trip, and that the instrumentation overlays really are instrumentation.
  Note the comparison currently covers 15 of 39 columns on one row of one case;
  the assertion is exact but narrow;
* **leaf drivers** (`leaf_erf`, `leaf_cubrt`, …) — one entry point per routine,
  with the input grids in Python. This is the pattern for every per-routine
  fixture from phase C onwards.

**What it cannot catch:** anything about sequencing. It says a routine agrees on
the inputs you chose; it says nothing about whether the driver calls it at the
right time with the right state.

**Constraint that shapes everything around it:** the UKCA mode tables are
built **once per process**, and `wrap_init` refuses any re-init that would need
them rebuilt — which means any change to `i_mode_setup`, `l_radaer`,
`i_tune_bc`, `l_fix_nacl_density`, `l_fix_ukca_hygroscopicities` or
`l_dust_mp_ageing`. Re-running an *identical* namelist is fine and is how a
driver resets between cases; anything else needs a fresh process.

Keying that guard on `i_mode_setup` alone was a real defect, found by the phase
B review: the other five were read from the new namelist and then ignored,
because `common_mode_setup_interface` was never re-called. Measured 2.3e-5 in
`drydp`, reported as `ierr = 0`, in the one mechanism whose whole purpose is
comparing at machine precision.

The underlying reason a rebuild is impossible: `ukca_mode_setup` allocates under `IF (.NOT. ALLOCATED)` and
never deallocates, and the 283 `nmas*` budget indices have no initialiser, so a
second init leaves stale indices — and since `nbudaer` also changes (8 vs 138) a
stale index can be out of bounds. One process per setup, enforced rather than
documented. See [reference build](REFERENCE_BUILD.md).

## Gate B — per-process, per-substep

`ukca_aero_step` calls thirteen process routines per chemistry step, fifteen
substeps deep. Comparing only the end state tells you *that* a port diverged;
comparing after each call tells you *where*.

Two dumps, because neither is sufficient:

* `budget_file` — the 283 per-process mass fluxes in `bud_aer_mas`;
* `state_file` — `nd`, `mdt`, `md`, `drydp`, `wetdp`, `mdwat`, `rhopar` after
  each of the thirteen calls, tagged `(step, seq, imts, izts)`, plus `h2so4`,
  `delh2so4_nucl`, `sec_org` and `s_cond_s` at the nucleation call.

Two things about that key are worth knowing, because both were wrong until the
phase B review. `seq` is a per-step call counter and it is **load-bearing**:
`calc_drydiam` and `volume_mode` each run twice per `imts`, so without it the
key is not unique — a committed golden carried 397 keys with two different
values, separable only by file row order. And the gas fields exist because
`ukca_calcnucrate` writes no aerosol array at all, so its snapshot used to be a
byte-for-byte copy of the preceding `conden` one; 21.7% of the dump was repeats,
and a wrong nucleation rate would first have surfaced at `coagwithnucl`.

Budgets carry **mass fluxes only** — no number, no diameters — so they cannot
localise a divergence in coagulation or mode merging, which is exactly where
divergences happen. Hence both.

**What it cannot catch:** a divergence inside a single routine. It bounds the
error to one call; gate A opens the call up.

## Gate C — trajectory

The committed `.npz` goldens: four cases × four streams in `f64`, plus the `f32`
trajectory for the precision comparison, and the numerics leaf sweep.
21 archives totalling 1.03 MB, guarded by a
content manifest.

**Read the tolerance policy in `tests/conftest.py` before using this.** The
headline number is that the primary gate is a *bounded* number of steps from a
golden state at `RTOL_TRAJECTORY`, and the 24-hour run is a **soak** at
`RTOL_SOAK = 1e-6`. An earlier draft gated 24 hours at 1e-9, which was never
achievable.

**What it cannot catch:** where. That is the whole reason gates 0, A and B
exist.

## What the harness deliberately does not do

* **It does not regenerate goldens in CI.** The Fortran reference is not
  bit-reproducible across platforms — FMA contraction and libm both differ — so
  a CI job that rebuilt them would fail the drift gate on the first PR from
  another machine, and the gate would then get loosened. ADR-005.
* **It does not modify `fortran/`.** Instrumentation is applied to a staged
  copy, and a patch touching `src/ukca/` may only *add* lines, checked
  mechanically on the patch text.
* **It does not auto-bless a capture.** `goldens_manifest.py --write` is always
  an explicit act, because auto-blessing would make the drift gate silent the
  one time it matters.

### The one deliberate divergence: the `ereport` shim

Gate A links `validation/f2py/glomap_ereport_shim.F90` in place of
`src/ukca/ereport_mod.F90`, **for the extension module only**. The real routine
handles a fatal error with `STOP 1`, which is correct in an executable — it is
what `fortran/patches/0002` exists to guarantee — and terminates the
interpreter inside a Python extension, with no traceback and no way to say which
of twenty reachable call sites fired. The shim records the call and returns.

Three things keep that honest:

1. **It never reaches the reference.** `build_reference.sh` does not mention it,
   so no golden and no committed number is affected. Asserted by
   `test_the_shim_is_not_linked_into_the_reference_build`, which also checks the
   vendored `ereport` still contains `STOP 1`.
2. **The binding checks for itself.** Letting a caller continue past a fatal
   error is what makes the error visible — but the caller then computes
   something, and that something looks like a number. Leaving the check to the
   caller would convert a loud crash into a silent wrong answer, which is
   strictly worse. So `wrap_init` and `wrap_step` record the shim's fatal count
   before the call and return `ierr = 5` if it moved. This was not hypothetical:
   before the check existed, `wrap_init` on a nonexistent namelist returned 0.
3. **Every leaf driver must do the same.** Call `wrap_ereport_count()` after any
   call that could reach an error path and discard the result if it is non-zero.

This is what unblocks issue #13 — `ukca_solvecoagnucl_v`'s error branch (code 4)
calls `ereport`, so it cannot be swept without the shim.

## Running it

```sh
./validation/build_reference.sh both     # ref-f32 and ref-f64
./validation/build_f2py.sh               # the gate-A binding (needs meson+ninja)
make goldens                             # all of the above, then capture + manifest
make test                                # everything; Fortran-dependent tests skip
```

Tests that need a toolchain carry the `fortran` marker and skip cleanly without
one. Everything that reads a committed golden does **not** — that is what makes
the port's own tests runnable in CI, where there is no gfortran at all.
