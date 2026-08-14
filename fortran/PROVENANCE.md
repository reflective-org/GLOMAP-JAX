# Provenance of the vendored UKCA sources

The `src/ukca/` directory contains files copied from the UKCA repository,
retaining their original copyright headers. 44 of the 46 are **byte-for-byte
identical** to upstream; two carry documented upstream bug fixes (see
[Patches](#patches) below).

Verify this at any time against a UKCA checkout:

```sh
make verify-vendor UKCA_ROOT=/path/to/ukca
# => identical to upstream : 44
#    differ only by patches: 2
#    unexplained differences: 0
#    PASS: src/ukca/ is upstream + patches/ exactly
```

| item | value |
|---|---|
| upstream repository | `https://github.com/MetOffice/ukca` |
| upstream commit | `387c5bb0f1166e67f029930ba624bf159bc68627` |
| upstream tag/branch | `main` (release 2026.7.1 lineage) |
| licence | BSD 3-Clause, Crown Copyright Met Office (see `LICENCE`) |
| files vendored | 46 (44 byte-identical, 2 patched) |

Individual GLOMAP science routines additionally carry
`(c) [University of Leeds] [2008]`, licensed to the Met Office under the UKCA
collaboration agreement. Those headers are preserved in every file.

## How this file set was chosen

It is the exact transitive `USE` closure of `ukca_aero_step`, computed
mechanically rather than assembled by hand:

```sh
python3 tools/gen_build_order.py --check src
# => OK: 51 files, no external module dependencies
```

The closure of `ukca_aero_step` is 45 files, counting `ukca_aero_step.F90`
itself. One further file is vendored deliberately:
`common_mode_setup_interface_mod.F90`, which dispatches `i_mode_setup` to the
right `ukca_mode_*` routine — re-used rather than reimplemented. All of its
own dependencies were already in the closure, so it contributes only itself.
That gives 46 files.

`ukca_aero_step` is the GLOMAP-mode microphysics driver. It is already written
against an array of `nbox` independent grid boxes with all environmental
fields passed as arguments, which is why a box model needs no science changes
at all — only a driver that fills those arguments.

Notably the closure needs **no UM infrastructure stubs**. The UKCA repository
already ships standalone `parkind1`, `yomhook`, `umPrintMgr`, `ereport_mod`
and `errormessagelength_mod` under `src/control/legacy/`, so the box model
links with zero external dependencies.

## Patches

`patches/` holds the complete, reviewable delta against upstream. Two patches,
both fixing genuine upstream defects and neither changing any numerical result:

* **`0001-guard-msec_org-zero-index.patch`** — `ukca_aero_step.F90` indexes
  `condensable_choice(msec_org)` and `mm_gas(msec_org)` before the
  `IF (msec_org > 0)` guard three lines below. Any mode setup lacking secondary
  organics — including the default `i_mode_setup = 1` — therefore reads out of
  bounds. Inert when optimised (the value is discarded), fatal under bounds
  checking.
* **`0002-ereport-nonzero-exit-status.patch`** — `ereport_mod.F90` terminates a
  fatal error with a bare `STOP`, which exits with status **0**, so every UKCA
  abort looked like success to the shell. Only reachable on the abort path.

Full rationale and verification for each are in `patches/README.md`. Both
should be reported upstream.

## Refreshing against a newer UKCA

1. Point `UKCA_ROOT` at an updated checkout.
2. Re-run the closure computation there to get the current file list.
3. Copy the listed files over `src/ukca/`, update the commit hash above.
4. Re-apply `patches/*.patch`, dropping any that upstream has since fixed.
5. `make clean && make && make test && make verify-vendor`.

Test 1 (`all processes off => state is invariant`) and test 2 (mass
conservation under coagulation) are the ones most likely to catch an
incompatible upstream change.

## File-by-file origin

| vendored file | origin in MetOffice/ukca |
|---|---|
| `src/ukca/common_mode_setup_interface_mod.F90` | `src/control/core/interface/common_mode_setup_interface_mod.F90` |
| `src/ukca/ereport_mod.F90` | `src/control/legacy/ereport_mod.F90` |
| `src/ukca/errormessagelength_mod.F90` | `src/control/legacy/errormessagelength_mod.F90` |
| `src/ukca/parkind1.F90` | `src/control/legacy/parkind1.F90` |
| `src/ukca/ukca_aero_step.F90` | `src/science/core/aerosols/glomap/ukca_aero_step.F90` |
| `src/ukca/ukca_ageing.F90` | `src/science/core/aerosols/glomap/ukca_ageing.F90` |
| `src/ukca/ukca_binapara_mod.F90` | `src/science/core/aerosols/glomap/ukca_binapara_mod.F90` |
| `src/ukca/ukca_calc_coag_kernel.F90` | `src/science/core/aerosols/glomap/ukca_calc_coag_kernel.F90` |
| `src/ukca/ukca_calc_drydiam.F90` | `src/science/core/aerosols/glomap/ukca_calc_drydiam.F90` |
| `src/ukca/ukca_calcminmaxgc.F90` | `src/science/core/aerosols/glomap/ukca_calcminmaxgc.F90` |
| `src/ukca/ukca_calcminmaxndmdt.F90` | `src/science/core/aerosols/glomap/ukca_calcminmaxndmdt.F90` |
| `src/ukca/ukca_calcnucrate.F90` | `src/science/core/aerosols/glomap/ukca_calcnucrate.F90` |
| `src/ukca/ukca_check_md_nd.F90` | `src/science/core/aerosols/glomap/ukca_check_md_nd.F90` |
| `src/ukca/ukca_cloudproc.F90` | `src/science/core/aerosols/activation/ukca_cloudproc.F90` |
| `src/ukca/ukca_coag_coff_v.F90` | `src/science/core/aerosols/glomap/ukca_coag_coff_v.F90` |
| `src/ukca/ukca_coagwithnucl.F90` | `src/science/core/aerosols/glomap/ukca_coagwithnucl.F90` |
| `src/ukca/ukca_coarse_no3_mod.F90` | `src/science/core/aerosols/glomap/ukca_coarse_no3_mod.F90` |
| `src/ukca/ukca_cond_coff_v.F90` | `src/science/core/aerosols/glomap/ukca_cond_coff_v.F90` |
| `src/ukca/ukca_conden.F90` | `src/science/core/aerosols/glomap/ukca_conden.F90` |
| `src/ukca/ukca_config_constants_mod.F90` | `src/control/core/interface/ukca_config_constants_mod.F90` |
| `src/ukca/ukca_config_specification_mod.F90` | `src/control/core/interface/ukca_config_specification_mod.F90` |
| `src/ukca/ukca_constants.F90` | `src/science/core/chemistry/ukca_constants.F90` |
| `src/ukca/ukca_dcoff_par_av_k.F90` | `src/science/core/aerosols/deposition/ukca_dcoff_par_av_k.F90` |
| `src/ukca/ukca_ddepaer_coeff_mod.F90` | `src/science/core/aerosols/deposition/ukca_ddepaer_coeff_mod.F90` |
| `src/ukca/ukca_ddepaer_incl_sedi_mod.F90` | `src/science/core/aerosols/deposition/ukca_ddepaer_incl_sedi_mod.F90` |
| `src/ukca/ukca_ddepaer_mod.F90` | `src/science/core/aerosols/deposition/ukca_ddepaer_mod.F90` |
| `src/ukca/ukca_error_mod.F90` | `src/control/shared/ukca_error_mod.F90` |
| `src/ukca/ukca_fine_no3_mod.F90` | `src/science/core/aerosols/glomap/ukca_fine_no3_mod.F90` |
| `src/ukca/ukca_impc_scav.F90` | `src/science/core/aerosols/deposition/ukca_impc_scav.F90` |
| `src/ukca/ukca_impc_scav_dust_mod.F90` | `src/science/core/aerosols/deposition/ukca_impc_scav_dust_mod.F90` |
| `src/ukca/ukca_missing_data_mod.F90` | `src/control/core/misc/ukca_missing_data_mod.F90` |
| `src/ukca/ukca_mode_check_artefacts_mod.F90` | `src/science/core/aerosols/glomap/ukca_mode_check_artefacts_mod.F90` |
| `src/ukca/ukca_mode_setup.F90` | `src/science/core/aerosols/glomap/ukca_mode_setup.F90` |
| `src/ukca/ukca_rainout.F90` | `src/science/core/aerosols/deposition/ukca_rainout.F90` |
| `src/ukca/ukca_remode.F90` | `src/science/core/aerosols/glomap/ukca_remode.F90` |
| `src/ukca/ukca_setup_indices.F90` | `src/science/core/aerosols/glomap/ukca_setup_indices.F90` |
| `src/ukca/ukca_solvecoagnucl_v.F90` | `src/science/core/aerosols/glomap/ukca_solvecoagnucl_v.F90` |
| `src/ukca/ukca_types_mod.F90` | `src/control/core/misc/ukca_types_mod.F90` |
| `src/ukca/ukca_um_legacy_mod.F90` | `src/control/legacy/ukca_um_legacy_mod.F90` |
| `src/ukca/ukca_vapour.F90` | `src/science/core/aerosols/glomap/ukca_vapour.F90` |
| `src/ukca/ukca_vgrav_av_k.F90` | `src/science/core/aerosols/deposition/ukca_vgrav_av_k.F90` |
| `src/ukca/ukca_volume_mode.F90` | `src/science/core/aerosols/glomap/ukca_volume_mode.F90` |
| `src/ukca/ukca_water_content_v.F90` | `src/science/core/aerosols/glomap/ukca_water_content_v.F90` |
| `src/ukca/ukca_wetox.F90` | `src/science/core/aerosols/glomap/ukca_wetox.F90` |
| `src/ukca/umprintmgr.F90` | `src/control/legacy/umprintmgr.F90` |
| `src/ukca/yomhook.F90` | `src/control/legacy/yomhook.F90` |