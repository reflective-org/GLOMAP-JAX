# Copyright and attribution

This repository contains code from two sources, both under the BSD 3-Clause
licence in `LICENCE`.

## Vendored Fortran — `fortran/`

Crown Copyright (c) Met Office, with contributions by the University of Leeds.

`fortran/` is a copy of the UKCA GLOMAP-mode aerosol microphysics and a
standalone box driver, taken from
[MetOffice/ukca](https://github.com/MetOffice/ukca) via
[reflective-org/glomap-box](https://github.com/reflective-org/glomap-box). Every
file retains its original copyright header. The GLOMAP science routines
additionally carry `(c) [University of Leeds] [2008]`, licensed to the Met
Office under the UKCA collaboration agreement.

See `PROVENANCE.md` for the exact upstream commit and the file-by-file origin.

## New Python/JAX code — `src/`, `tests/`, `validation/`, `scripts/`, `benchmarks/`

Copyright (c) 2026 Reflective, released under the same BSD 3-Clause licence.

This is a reimplementation, not a translation of copied text: it is written
against the Fortran's published behaviour and validated against it numerically.

## No endorsement

BSD 3-Clause clause 3 prohibits using the copyright holder's name to endorse
derived work. Accordingly:

> **Not affiliated with, endorsed by, or an official product of the Met Office, NCAS, or the University of Leeds.**

It is an independent port. Do not describe it as an official UKCA product, and
do not use Met Office, NCAS or University of Leeds branding in connection with
it. Results produced by this code are the responsibility of this project, not of
the UKCA authors.
