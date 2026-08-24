# GLOMAP-JAX

Differentiable GLOMAP-mode aerosol microphysics in Python/JAX — a faithful port
of the UKCA GLOMAP-mode box model.

Active processes: nucleation (binary homogeneous H₂SO₄–H₂O, optional
boundary-layer), condensation of H₂SO₄ and secondary organics, intra- and
inter-modal coagulation, ageing of insoluble into soluble modes, mode merging,
and equilibrium water uptake.

## Status

Order 1 (faithful port) in progress. See `PROGRESS.md` in the repository root.

## Start here

* [Porting notes](porting-notes.md) — the traps, and why they are traps
* [Validation harness](harness.md) — the four gates, and what each cannot catch
* [Fidelity flags](fidelity.md) — where the port deliberately reproduces
  upstream bugs
* [Unsupported](unsupported.md) — what this does **not** do
* [Upstream defects](UPSTREAM_DEFECTS.md) — what we found in UKCA

## Licence

Apache 2.0 for the port; vendored Fortran is Crown Copyright Met Office with
University of Leeds contributions under BSD 3-Clause. See `COPYRIGHT.md`.

**Not affiliated with, endorsed by, or an official product of the Met Office,
NCAS, or the University of Leeds.**
