# inputs

Namelists and initial conditions driving both the Fortran reference and the
JAX port. Keeping them in one place is what makes a comparison meaningful: the
two sides must be given the same case, not two cases that resemble each other.

* `namelists/` — cases this repository added. The seven shipped cases live in
  `fortran/namelists/` and stay there, because they are part of the vendored
  tree and are hash-checked by the tamper test.

`bl_nmts3.nml` is the only case anywhere with `nmts > 1`, so it is the only one
that exercises the nested outer/inner substep structure at all.
