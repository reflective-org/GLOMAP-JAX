# Copyright and attribution

This repository contains code under two licences. `NOTICE` is the
distribution-facing summary; this file is the map.

## New Python/JAX code — `src/`, `tests/`, `validation/`, `docs/`

Copyright (c) 2026 Reflective, released under the **Apache License 2.0**
(`LICENCE`).

The port is written from the vendored Fortran — the repo's own rule is "port
from the code, not the comments" — so it is treated as a derivative work of
the BSD-licensed source, and every distribution of it (source or wheel)
carries the BSD notice via `NOTICE`. BSD 3-Clause permits derivatives under a
different licence provided the notice, conditions and disclaimer are
retained; that is what `tests/test_licensing.py` enforces.

## Vendored Fortran — `fortran/`

Crown Copyright (c) Met Office, with contributions by the University of
Leeds, under the **BSD 3-Clause licence** (`fortran/LICENCE`).

`fortran/` is a copy of the UKCA GLOMAP-mode aerosol microphysics and a
standalone box driver, taken from
[MetOffice/ukca](https://github.com/MetOffice/ukca) via
[reflective-org/glomap-box](https://github.com/reflective-org/glomap-box). Every
file retains its original copyright header. The GLOMAP science routines
additionally carry `(c) [University of Leeds] [2008]`, licensed to the Met
Office under the UKCA collaboration agreement.

`fortran/src/box/` and the overlay extensions in `validation/patches/` are
new code, Copyright (c) 2026 Reflective, kept under the **same BSD 3-Clause
licence** deliberately: the box driver stays comparable with `glomap-box`,
and overlays are drafted for upstreaming there.

See `PROVENANCE.md` for the exact upstream commit and the file-by-file origin.

## No endorsement

BSD 3-Clause clause 3 prohibits using the copyright holder's name to endorse
derived work. Accordingly:

> **Not affiliated with, endorsed by, or an official product of the Met Office, NCAS, or the University of Leeds.**

It is an independent port. Do not describe it as an official UKCA product, and
do not use Met Office, NCAS or University of Leeds branding in connection with
it. Results produced by this code are the responsibility of this project, not of
the UKCA authors.
