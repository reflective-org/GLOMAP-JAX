# Patches to the vendored UKCA sources

`src/ukca/` is otherwise a byte-for-byte copy of upstream UKCA (see
`../PROVENANCE.md`). Anything in this directory is an explicit, auditable
delta. The rules:

* A patch lands here only for a genuine **upstream defect**, never to change
  science, tune a result, or work around a driver limitation.
* Every patch must be demonstrated **bit-identical in release builds**, so no
  published result can depend on whether it was applied.
* `make verify-vendor UKCA_ROOT=/path/to/ukca` checks that `src/ukca/` equals
  upstream plus exactly these patches, and nothing more.

Applied patches are already present in `src/ukca/`; the files here exist so
the delta is reviewable and can be sent upstream.

---

## 0001-guard-msec_org-zero-index.patch

**Defect.** `ukca_aero_step.F90`, in the nucleation block, reads

```fortran
cp_sec_org        = condensable_choice(msec_org)
s0g_to_gc_sec_org = mm_gas(msec_org)/mm(cp_sec_org)

IF (msec_org > 0) THEN
  ...
```

The two array reads use `msec_org` as a subscript *before* the
`IF (msec_org > 0)` guard three lines below. `msec_org` is legitimately `0`
("secondary organics not present in this configuration") for every mode setup
without an SO component — including `suss_4mode`, which is `i_mode_setup = 1`.

So for those configurations the code reads `condensable_choice(0)` and
`mm_gas(0)`, both out of bounds, and then `mm(cp_sec_org)` with a wild
subscript.

**Impact.** Latent in an ordinary optimised build: the garbage
`s0g_to_gc_sec_org` is computed but never consumed, because the `ELSE` branch
is the one that executes when `msec_org == 0`. It becomes fatal the moment
bounds checking is enabled, which makes `make debug` unusable for the default
configuration — exactly the build you want when investigating anything.

**Fix.** Move the two reads inside the existing guard. No logic change.

**Verification.** With the patch applied, all six shipped cases run clean
under `-fcheck=bounds,pointer -ffpe-trap=invalid,zero,overflow
-finit-real=snan`. Release-build CSV output is byte-identical before and
after the patch for all three example namelists, confirming the discarded
value never influenced results.

**Status.** Should be reported upstream to MetOffice/ukca. Present in
`387c5bb`.

---

## 0002-ereport-nonzero-exit-status.patch

**Defect.** `ereport_mod.F90` handles a fatal error (`error_status > 0`) with a
bare `STOP`, which in Fortran terminates with exit status **0**. Every UKCA
abort therefore looked like success to the shell:

```console
$ ./bin/glomap_box /nonexistent.nml; echo $?
UKCA ERROR in READ_BOX_NAMELIST: cannot open namelist file /nonexistent.nml
0
```

**Impact.** Severe for a standalone tool. Any script, `make` target or CI job
that tests the exit status treats a crashed run as a passing one. It defeated
this repository's own test suite, which could report `passed: 8 failed: 0`
against a binary that never ran.

Inside the UM this is harmless — the UM aborts through its own MPI-aware error
path and never relies on `ereport`'s exit status — which is presumably why it
was never noticed. `src/control/legacy/ereport_mod.F90` is a UKCA-supplied
standalone shim, not UM code and not science, so the correct exit status is
squarely its responsibility.

**Fix.** `STOP 1` instead of `STOP`, preceded by `FLUSH(6)` so the diagnostic
is not lost when output is redirected.

**Verification.** Fatal errors now exit 1, successful runs exit 0. No effect on
numerical results — the statement is only reachable on the abort path.

**Status.** Should be reported upstream. Present in `387c5bb`.

---

### Related, not patched

The same shape appears in `ukca_coarse_no3_mod.F90:211` and
`ukca_fine_no3_mod.F90:207-208`, which read `mm_gas(mhno3)` / `mm_gas(mnh3)`
without a local guard. These are only reachable when the nitrate/ammonium
production switches are on, and in the mode setups that enable them those
indices are non-zero. The box model keeps those switches off, so they are
left untouched rather than patched speculatively. Revisit if Phase 4.4 of
`../PLAN.md` is implemented.
