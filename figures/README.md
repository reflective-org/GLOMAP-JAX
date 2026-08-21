# figures

Plots from `benchmarks/` and the validation gates. Generated, gitignored by
default; commit one explicitly when a document references it.

## What is here

    extract_figure_data.py   pulls the plotted series out of the committed goldens
    measure_hazards.py       re-measures the numerical hazards against the running JAX
    make_report.py           renders report.html from both

    python figures/extract_figure_data.py
    python figures/measure_hazards.py
    python figures/make_report.py

`data.json` and `hazards.json` are intermediates and are gitignored;
`report.html` is committed because documents reference it.

The split is deliberate. Everything in `data.json` comes from a **committed
golden**, so those figures are reproducible without a Fortran toolchain and
move when the goldens move. Everything in `hazards.json` is a property of the
**running JAX and CPU** and is re-measured every time, because three of the
four hazards are version- or platform-dependent — a figure that pinned them as
constants would be wrong the next time somebody upgraded, which is exactly how
the phase-D interpreter mix-up went unnoticed.

## What is deliberately not here

**Performance benchmarks.** `benchmarks/` is empty and ADR-008's timing table
cannot be re-derived from anything in the repository — that is issue #21. None
of the figures here are timings, and none should be read as any.
