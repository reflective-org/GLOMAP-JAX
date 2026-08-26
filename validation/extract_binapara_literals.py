#!/usr/bin/env python3
"""Extract `ukca_binapara`'s polynomial coefficients into a generated module.

    python validation/extract_binapara_literals.py --check   # CI / staleness
    python validation/extract_binapara_literals.py           # rewrite

The Vehkamaki (2002) parameterisation is four expressions carrying **113
literal coefficients between them** -- 10, 50, 50 and 3 -- most written to
sixteen significant digits. Retyping them is not an option: phase C's mode tables were 1,351
numbers and the reason they were machine-extracted is that a mistyped digit
gives a plausible table and a quietly wrong model. Here it would give a
plausible nucleation rate.

What is extracted is not just the numbers but the **term structure and its
order**. Fortran sums left to right, so

    a + b - c + d   is   ((a + b) - c) + d

and a port that grouped the terms differently -- by power of `logrh`, say, or
by collecting the `1/termx` terms -- would be algebraically identical and
numerically different. Each expression is therefore stored as an ordered list
of terms, and the port sums them in that order and no other.

Term grammar, which the parser asserts it has covered completely:

    [sign] COEF                                 a bare constant
    [sign] COEF * FACTOR [* FACTOR ...]         a monomial
    [sign] COEF / termx(jl)                     a bare constant over termx
    [sign] ( COEF * FACTOR [...] ) / termx(jl)  a monomial over termx

`FACTOR` is one of `tdegk`, `tdegk2`, `tdegk3`, `logrh`, `logrh2`, `logrh3`,
`logh2so4`, `logh2so42`, `logh2so43`, or -- in `rc` alone -- `termx(jl)` and
`ntot(jl)`. Anything else raises, rather than being silently dropped: a parser
that skips a term it does not recognise produces a shorter polynomial that
still runs.

The round trip is asserted both ways. The parsed terms are re-rendered into
Fortran source text and compared, token for token, against the whitespace-
normalised original -- so a coefficient that was read with a digit missing, or
a factor attached to the wrong term, fails here rather than in a golden.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_binapara_mod.F90"
TARGET = REPO / "src" / "glomap_jax" / "physics" / "_binapara_literals.py"

EXPRESSIONS = ("termx", "jveh", "ntot", "rc")

#: Every name that may appear as a factor. `termx(jl)` and `ntot(jl)` are
#: values computed earlier in the same loop iteration, not inputs.
FACTORS = (
    "tdegk3",
    "tdegk2",
    "tdegk",
    "logrh3",
    "logrh2",
    "logrh",
    "logh2so43",
    "logh2so42",
    "logh2so4",
    "termx(jl)",
    "ntot(jl)",
)

_NUMBER = r"\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?"


def joined_source() -> str:
    """The routine's body with continuations joined and comments stripped."""
    text = SOURCE.read_text(encoding="utf-8")
    text = re.sub(r"&\s*\n\s*", "", text)
    return re.sub(r"!.*", "", text)


def expression_text(body: str, name: str) -> str:
    """The right-hand side of `name(jl) = ...`, whitespace-normalised."""
    pattern = rf"^\s*{name}\(jl\)\s*=\s*(.*?)(?=\n\s*(?:[a-z_0-9]+\(jl\)\s*=|END DO|CALL|IF))"
    match = re.search(pattern, body, re.S | re.M | re.I)
    if not match:
        raise SystemExit(f"no assignment to {name}(jl) in {SOURCE.name}")
    return " ".join(match.group(1).split())


def split_terms(expr: str) -> list[tuple[str, str]]:
    """Top-level `+`/`-` split, respecting parentheses.

    Signs inside an exponent (`1.5e-6`) must not split the expression, which is
    why the scan skips a `+`/`-` immediately preceded by `e` or `E`.
    """
    terms: list[tuple[str, str]] = []
    depth = 0
    start = 0
    sign = "+"
    # `ntot` and `rc` open with a unary minus. Consumed here rather than left
    # to the scan, which only treats a sign as a separator when something
    # precedes it.
    if expr[:1] in "+-":
        sign = expr[0]
        expr = expr[1:].lstrip()
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch in "+-" and depth == 0 and i > 0 and expr[i - 1] not in "eE":
            chunk = expr[start:i].strip()
            if chunk:
                terms.append((sign, chunk))
            sign = ch
            start = i + 1
    tail = expr[start:].strip()
    if tail:
        terms.append((sign, tail))
    if not terms:
        raise SystemExit(f"no terms parsed from {expr[:60]!r}")
    return terms


def parse_term(sign: str, chunk: str) -> tuple[str, str, tuple[str, ...], bool]:
    """One term as `(sign, coefficient, factors, over_termx)`."""
    over = False
    body = chunk
    if body.endswith("/termx(jl)"):
        over = True
        body = body[: -len("/termx(jl)")].strip()
        if body.startswith("(") and body.endswith(")"):
            body = body[1:-1].strip()

    parts = [p.strip() for p in body.split("*") if p.strip()]
    # `termx(jl)` and `ntot(jl)` contain no `*`, so splitting is safe.
    if not parts:
        raise SystemExit(f"empty term {chunk!r}")

    if not re.fullmatch(_NUMBER, parts[0]):
        raise SystemExit(f"term {chunk!r} does not start with a numeric coefficient")
    coeff = parts[0]

    factors = tuple(parts[1:])
    for f in factors:
        if f not in FACTORS:
            raise SystemExit(f"unknown factor {f!r} in term {chunk!r}")
    return sign, coeff, factors, over


def parse_expression(expr: str) -> list[tuple[str, str, tuple[str, ...], bool]]:
    return [parse_term(sign, chunk) for sign, chunk in split_terms(expr)]


def render_term(sign: str, coeff: str, factors: tuple[str, ...], over: bool) -> str:
    inner = "*".join((coeff, *factors))
    if over:
        inner = f"({inner})/termx(jl)" if factors else f"{inner}/termx(jl)"
    return f"{sign}{inner}"


def assert_round_trip(name: str, expr: str, terms: list) -> None:
    """Re-render and compare against the source, token for token.

    Both directions matter. Comparing only the count would pass a term whose
    coefficient lost a digit; comparing only the values would pass a term
    parsed with the wrong sign. This compares the rendered string with every
    space and redundant parenthesis removed, which is exactly the information
    the port will use and nothing else.
    """
    rendered = "".join(render_term(*t) for t in terms)
    if rendered.startswith("+"):
        rendered = rendered[1:]
    normal = expr.replace(" ", "")
    # The source writes `(c*f)/termx(jl)` for monomials and `c/termx(jl)` for
    # bare constants; render_term reproduces both, so the only remaining
    # difference is a leading `+` the source never writes.
    if rendered != normal:
        for i, (a, b) in enumerate(zip(rendered, normal)):
            if a != b:
                raise SystemExit(
                    f"{name}: round trip diverges at character {i}\n"
                    f"  source:   ...{normal[max(0, i - 40) : i + 40]}\n"
                    f"  rendered: ...{rendered[max(0, i - 40) : i + 40]}"
                )
        raise SystemExit(f"{name}: round trip differs in length ({len(rendered)} vs {len(normal)})")


def parse_clamps(body: str) -> dict[str, tuple[float, float]]:
    """The input bounds at `:106-118`, which are part of the specification.

    `WHERE (t(:) < 190.15) t(:)=190.15` and the five like it. Parsed rather
    than typed for the same reason as everything else here, and because the
    *port* must clamp before taking the logarithm -- `LOG(0.0)` otherwise.
    """
    out: dict[str, tuple[float, float]] = {}
    for var in ("t", "rh", "h2so4"):
        lo = re.search(
            rf"WHERE\s*\(\s*{var}\(:\)\s*<\s*({_NUMBER})\s*\)\s*{var}\(:\)\s*=\s*({_NUMBER})", body
        )
        hi = re.search(
            rf"WHERE\s*\(\s*{var}\(:\)\s*>\s*({_NUMBER})\s*\)\s*{var}\(:\)\s*=\s*({_NUMBER})", body
        )
        if not lo or not hi:
            raise SystemExit(f"could not find both clamps for {var}")
        for m in (lo, hi):
            if float(m.group(1)) != float(m.group(2)):
                raise SystemExit(f"{var}: clamp tests {m.group(1)} but assigns {m.group(2)}")
        out[var] = (float(lo.group(1)), float(hi.group(1)))
    return out


def parse_output_limits(body: str) -> dict[str, float]:
    """The three post-exponential rules at `:249-283`."""
    want = {
        "ntot_min": r"IF\s*\(\s*ntot_out\(jl\)\s*<\s*({n})\s*\)".format(n=_NUMBER),
        "t_cold": r"IF\s*\(\s*t\(jl\)\s*<\s*({n})\s*\)".format(n=_NUMBER),
        "j_cold": r"jveh\(jl\)\s*=\s*({n})\s*\n?\s*ELSE".format(n=_NUMBER),
        "j_floor": r"IF\s*\(\s*jveh\(jl\)\s*<\s*({n})\s*\)".format(n=_NUMBER),
        "j_ceiling": r"IF\s*\(\s*jveh\(jl\)\s*>\s*({n})\s*\)".format(n=_NUMBER),
    }
    out: dict[str, float] = {}
    for key, pattern in want.items():
        m = re.search(pattern, body)
        if not m:
            raise SystemExit(f"could not find {key} in {SOURCE.name}")
        out[key] = float(m.group(1))
    return out


def build() -> dict:
    body = joined_source()
    terms: dict[str, list] = {}
    for name in EXPRESSIONS:
        expr = expression_text(body, name)
        parsed = parse_expression(expr)
        assert_round_trip(name, expr, parsed)
        terms[name] = parsed
    return {
        "terms": terms,
        "clamps": parse_clamps(body),
        "limits": parse_output_limits(body),
    }


def render(data: dict) -> str:
    lines = [
        '"""Generated by `validation/extract_binapara_literals.py`. Do not edit.',
        "",
        "`ukca_binapara`'s Vehkamaki (2002) coefficients: 113 literals across four",
        "expressions, machine-extracted and never retyped.",
        "",
        "Each expression is an ORDERED tuple of `(sign, coefficient, factors,",
        "over_termx)`. The order is the Fortran's, and it is load-bearing: Fortran",
        "sums left to right, so regrouping the terms is algebraically identical and",
        "numerically different.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
    ]
    for name in EXPRESSIONS:
        lines.append(f"{name.upper()}_TERMS = (")
        for sign, coeff, factors, over in data["terms"][name]:
            # Double quotes and an explicit trailing comma so the emitted file
            # is already in `ruff format`'s normal form. Otherwise `make fmt`
            # rewrites it and `--check` reports the generated file as stale
            # against its own generator -- a staleness gate that fires on
            # formatting is a gate nobody will trust.
            rendered = (
                "("
                + ", ".join(f'"{f}"' for f in factors)
                + ("," if len(factors) == 1 else "")
                + ")"
            )
            lines.append(f'    ("{sign}", {coeff}, {rendered}, {over}),')
        lines.append(")")
        lines.append("")
    lines.append("#: `WHERE` bounds applied to the inputs before any logarithm (`:106-118`).")
    lines.append("CLAMPS = {")
    for var, (lo, hi) in data["clamps"].items():
        lines.append(f'    "{var}": ({lo!r}, {hi!r}),')
    lines.append("}")
    lines.append("")
    lines.append("#: The post-exponential rules at `:249-283`.")
    lines.append("LIMITS = {")
    for key, value in data["limits"].items():
        lines.append(f'    "{key}": {value!r},')
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the target is stale")
    args = parser.parse_args(argv)

    fresh = render(build())
    if args.check:
        if not TARGET.is_file():
            print(f"{TARGET} does not exist; run without --check", file=sys.stderr)
            return 1
        if TARGET.read_text(encoding="utf-8") != fresh:
            print(f"{TARGET} is stale; re-run without --check", file=sys.stderr)
            return 1
        print(f"{TARGET.name} is up to date")
        return 0

    TARGET.write_text(fresh, encoding="utf-8")
    counts = {k: len(v) for k, v in build()["terms"].items()}
    print(f"wrote {TARGET} -- terms per expression: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
