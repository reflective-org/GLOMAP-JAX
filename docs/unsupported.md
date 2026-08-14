# What this does not do

Stated up front, because a public port invites the assumption of parity with UM
GLOMAP and there is none.

## Processes compiled in but switched off

The box model these results are validated against runs a single well-mixed
parcel, which carries no column, surface or cloud information. So:

| process | why off |
|---|---|
| wet oxidation of SO₂ | needs cloud water and oxidant fields |
| cloud processing / activation | off via `iactmethod = 0`, which gates the call |
| dry deposition, sedimentation | needs surface and layer-depth data |
| nucleation and impaction scavenging | needs a precipitation profile |
| nitrate / ammonium production | needs HNO₃ and NH₃ chemistry |

Enabling one is a matter of supplying fields, not writing science.

## Switches not supported

* **`icoag = 4`** — broken upstream. `ukca_coag_coff_v.F90:339-340` reads
  `mfppi`/`mfppj`, assigned only inside the mutually exclusive `icoag == 1`
  block, so it always reads undefined memory. There is no correct reference to
  validate against, so this raises rather than producing plausible garbage.
* **`i_nuc_method = 1`** — does not exist. Upstream `ereport`s for anything
  outside 2–3 and the header marks it "Do not use!!".
* **Kulmala (1998) BHN** — dead code upstream. `i_bhn_method` is hard-coded to
  Vehkamäki in source (`ukca_calcnucrate.F90:256-257`), so the Kulmala branch is
  unreachable and is not ported.
* **`iextra_checks > 1`** — activates `ukca_mode_check_mdt`, which zeroes number
  concentration for out-of-range modes and so changes mass budgets. Not ported.

## Configuration, not UM defaults

This port reproduces the **box model's** configuration, not the UM's. In
particular `l_fix_ukca_water_content`, `l_fix_neg_pvol_wat` and
`l_fix_ukca_hygroscopicities` are true, `ntype = 1`, and `iactmethod = 0`.
Mode setups 10, 12 and 13 (the `ncp = 9`/`10` configurations) are rejected.
