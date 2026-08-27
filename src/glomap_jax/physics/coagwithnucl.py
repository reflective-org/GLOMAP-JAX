"""`ukca_coagwithnucl`: coagulation and nucleation applied to number and mass.

Ported from `fortran/src/ukca/ukca_coagwithnucl.F90`. Builds `A`, `B` and `C`
per mode, calls `solvecoagnucl` for the number change, accumulates the
per-component mass transfers in `mtran`, redistributes them through
`coag_mode`, and writes 53 budget diagnostics.

The `icp` loop is loop-carried, and this port says so out loud
--------------------------------------------------------------

`:544-575` is one of the five loops CLAUDE.md names as needing sequential
treatment::

    DO icp=1,ncp
      ...
      WHERE (mask1(:) .AND. (mdcpnew(:) < 0.0))
        nd(:,imode)=0.0
        mask1(:)=.FALSE. ! set false so not used for other icp values
      END WHERE

`mask1` is mutated *inside* the loop. Once any component's new mass goes
negative the mode's number is zeroed and every **later** component is skipped,
so the components are not independent and cannot be evaluated together.

`broadcast=True` implements the wrong version deliberately -- every component
against the original mask -- and
`tests/test_coagwithnucl.py::test_the_broadcast_form_differs` requires it to
disagree with the reference. CLAUDE.md asks for that test to be written *as part
of* the port, because a scan that was never wrong is a scan nobody can show is
needed.

It is a Python loop rather than a `lax.scan`, and that is a deliberate order-1
choice: `ncp` is static, so the loop unrolls at trace time and the answer is
identical. `lax.scan` buys compile time, not correctness, and CLAUDE.md's
"port first, `jit` second, in separate commits" puts it in the second commit.

`bterm` carries across loop iterations
--------------------------------------

`:277` zeroes `bterm` once for the whole call and `:314` assigns it only inside
`IF (interoff /= 1)` and only under `mask2`. `:322` then reads it unmasked. So
`bterm` genuinely holds values from the previous `(imode, jmode)` pair on the
rows `mask2` excludes -- benign, because every consumer of `xxx` is itself
masked by `mask2`, and with `interoff = 1` it stays at the zero from `:277` and
`mtran` comes back zero. Reproduced as a carried array rather than recomputed,
because "benign" is a property of the current masks and not of the structure.

Where the mass goes is `coag_mode`'s decision
---------------------------------------------

`:534` scatters `mtran(:,icp,imode,jmode)` into
`mtrantoi(:,coag_mode(imode,jmode),icp)`, so several `(imode, jmode)` pairs
accumulate into one destination. Phase C measured `coag_mode` symmetric on all
64 entries and warned that a transposed transcription would be undetectable in
the table; this is the routine where the subscript order has consequences.

Not ported
----------

`iextra_checks > 1` calls `ukca_mode_check_mdt` at `:583`, which
`docs/unsupported.md` records as not ported because it zeroes number
concentration and changes mass budgets. Raises rather than silently ignoring.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from glomap_jax.config.fidelity import FidelityConfig
from glomap_jax.core import numerics
from glomap_jax.core.constants import CONC_EPS, DN_EPS, NMOL, XXX_EPS
from glomap_jax.physics._coagwithnucl_literals import COAG_BUDGET_SITES
from glomap_jax.physics.budget_indices import NOT_CARRIED
from glomap_jax.physics.solvecoagnucl import solvecoagnucl

__all__ = ["CoagResult", "coagwithnucl"]

MODE_COR_INSOL = 6  # 0-based; `imode < mode_cor_insol` at `:463`


class CoagResult(tuple):
    """`(nd, md, mdt, bud_aer_mas, ageterm2, clamped, failed)`.

    `clamped` and `failed` are `solvecoagnucl`'s, per mode: the Fortran drops
    the first entirely and reports the second by stopping the process.
    """

    __slots__ = ()

    def __new__(cls, nd, md, mdt, bud_aer_mas, ageterm2, clamped, failed):
        return super().__new__(cls, (nd, md, mdt, bud_aer_mas, ageterm2, clamped, failed))

    nd = property(lambda self: self[0])
    md = property(lambda self: self[1])
    mdt = property(lambda self: self[2])
    bud_aer_mas = property(lambda self: self[3])
    ageterm2 = property(lambda self: self[4])
    clamped = property(lambda self: self[5])
    failed = property(lambda self: self[6])


def _transfer(mdold_c: Array, ndold_i: Array, xxx: Array, mask2: Array) -> Array:
    """`:325-334`. The exponential is evaluated only where it is "worth it".

    `mask4 = ABS(xxx) > xxx_eps` selects `1 - EXP(-xxx)`; below the tolerance
    the linear `xxx` stands in. Both arms are masked by `mask2`, so a row the
    caller excluded stays at whatever `mtran` already held -- zero.
    """
    mask4 = jnp.abs(xxx) > XXX_EPS
    expform = mdold_c * ndold_i * (1.0 - jnp.exp(-xxx))
    linform = mdold_c * ndold_i * xxx
    return jnp.where(mask2, jnp.where(mask4, expform, linform), 0.0)


def coagwithnucl(
    tables,
    coag_mode_table: Array,  # ZERO-based; see the note at the scatter below
    budget,
    gas,
    nd: Array,
    md: Array,
    mdt: Array,
    delgc_nucl: Array,
    kii_arr: Array,
    kij_arr: Array,
    bud_aer_mas: Array,
    *,
    dtz: float,
    intraoff: int,
    interoff: int,
    iextra_checks: int = 0,
    broadcast: bool = False,
    fidelity: FidelityConfig = FidelityConfig(),
) -> CoagResult:
    """One coagulation/nucleation step.

    `broadcast=True` evaluates the `icp` loop's components against the original
    mask instead of sequentially. It exists for the test that shows the
    sequential form is required, and must never be used for a trajectory.
    """
    if iextra_checks > 1:
        raise ValueError(
            f"iextra_checks={iextra_checks} activates ukca_mode_check_mdt "
            "(ukca_coagwithnucl.F90:583), which is not ported: it zeroes number "
            "concentration for out-of-range modes and so changes mass budgets. "
            "See docs/unsupported.md."
        )

    nd, md, mdt = jnp.asarray(nd), jnp.asarray(md), jnp.asarray(mdt)
    kii_arr, kij_arr = jnp.asarray(kii_arr), jnp.asarray(kij_arr)
    bud_aer_mas = jnp.asarray(bud_aer_mas)
    nbox, nmodes = nd.shape
    ncp = md.shape[2]
    topmode = int(tables.topmode)
    mode = [bool(m) for m in tables.mode]
    component = [[bool(tables.component[i][c]) for c in range(ncp)] for i in range(nmodes)]

    ndold = nd
    mdold = md  # `:281-286` transposes; indexing directly is the same read.
    mtran = jnp.zeros((nbox, ncp, nmodes, nmodes), dtype=nd.dtype)
    ageterm2 = jnp.zeros((nbox, 4, 4, ncp), dtype=nd.dtype)
    # `:277`, zeroed ONCE for the whole call and carried across every pair.
    bterm = jnp.zeros(nbox, dtype=nd.dtype)

    if gas.mh2so4 >= 0:
        delh2so4 = jnp.asarray(delgc_nucl)[:, gas.mh2so4]
    else:
        delh2so4 = jnp.zeros(nbox, dtype=nd.dtype)

    clamped = jnp.zeros((nbox, nmodes), dtype=bool)
    failed = jnp.zeros((nbox, nmodes), dtype=bool)

    def eps(i: int) -> float:
        return float(tables.num_eps[i])

    def reset_small(md_in, mdt_in, imode, mask1):
        """`:377-384`. Where the number is insignificant, the composition goes
        back to the mode's initial fractions rather than being left alone."""
        for icp in range(ncp):
            if component[imode][icp]:
                md_in = md_in.at[:, imode, icp].set(
                    jnp.where(
                        ~mask1,
                        float(tables.mmid[imode]) * float(tables.mfrac_0[imode][icp]),
                        md_in[:, imode, icp],
                    )
                )
        mdt_in = mdt_in.at[:, imode].set(
            jnp.where(~mask1, float(tables.mmid[imode]), mdt_in[:, imode])
        )
        return md_in, mdt_in

    def inter_pair(mtran_in, bterm_in, imode, jmode, gate_component):
        """One `(imode, jmode)` inter-modal pair: the `B` contribution and the
        per-component mass transfer."""
        kij = kij_arr[:, imode, jmode]
        mask2 = (ndold[:, imode] > eps(imode)) & (ndold[:, jmode] > eps(jmode))
        if interoff != 1:
            bterm_in = jnp.where(mask2, -kij * ndold[:, jmode], bterm_in)
            b_add = jnp.where(mask2, -kij * ndold[:, jmode], 0.0)
        else:
            b_add = jnp.zeros(nbox, dtype=nd.dtype)
        xxx = -bterm_in * dtz
        for icp in range(ncp):
            if gate_component and not component[jmode][icp]:
                continue
            mtran_in = mtran_in.at[:, icp, imode, jmode].set(
                _transfer(mdold[:, imode, icp], ndold[:, imode], xxx, mask2)
            )
        return mtran_in, bterm_in, b_add

    # --- the soluble modes, `:288-404` ---
    for imode in range(4):
        if not mode[imode]:
            continue
        a = jnp.where(
            (ndold[:, imode] > eps(imode)) & (intraoff != 1), -0.5 * kii_arr[:, imode], 0.0
        )
        b = jnp.zeros(nbox, dtype=nd.dtype)
        mask1 = ndold[:, imode] > eps(imode)

        for jmode in range(imode + 1, 4):
            if mode[jmode]:
                # `:324`: the component gate is on JMODE here.
                mtran, bterm, b_add = inter_pair(mtran, bterm, imode, jmode, True)
                b = b + b_add
        for jmode in range(imode + 4, topmode):
            if mode[jmode]:
                # `:366`: NO component gate on the insoluble arm -- every
                # component is transferred, whether the destination carries it
                # or not. The asymmetry with the soluble arm is the Fortran's.
                mtran, bterm, b_add = inter_pair(mtran, bterm, imode, jmode, False)
                b = b + b_add

        md, mdt = reset_small(md, mdt, imode, mask1)

        c = jnp.zeros(nbox, dtype=nd.dtype)
        if imode == 0:
            # `:389-391`. Two divisions by scalar constants, so two
            # `true_divide`s: `delh2so4/dtz/nmol` is `(x/dtz)/nmol`.
            nucleating = delh2so4 > CONC_EPS
            c = jnp.where(
                nucleating,
                numerics.true_divide(numerics.true_divide(delh2so4, dtz), NMOL),
                0.0,
            )
            mask1a = mask1 | nucleating
        else:
            mask1a = mask1

        result = solvecoagnucl(mask1a, a, b, c, ndold[:, imode], dtz=dtz, fidelity=fidelity)
        clamped = clamped.at[:, imode].set(result.clamped)
        failed = failed.at[:, imode].set(result.failed)
        # `:399-401`: NOT masked by mask1 -- deln is already zero outside mask1a.
        nd = nd.at[:, imode].set(
            jnp.where(jnp.abs(result.deln) > DN_EPS, ndold[:, imode] + result.deln, nd[:, imode])
        )

    # --- the insoluble modes, `:427-518` ---
    for imode in range(4, topmode):
        if not mode[imode]:
            continue
        # `:430-440`: ageterm2 takes what the SOLUBLE modes already transferred
        # into this insoluble mode, so it must run after the loop above.
        for jmode in range(4):
            if not mode[jmode]:
                continue
            for jcp in range(ncp):
                if component[jmode][jcp]:
                    ageterm2 = ageterm2.at[:, jmode, imode - 4, jcp].set(
                        mtran[:, jcp, jmode, imode]
                    )

        a = jnp.where(
            (ndold[:, imode] > eps(imode)) & (intraoff != 1), -0.5 * kii_arr[:, imode], 0.0
        )
        b = jnp.zeros(nbox, dtype=nd.dtype)
        mask1 = ndold[:, imode] > eps(imode)

        if imode < MODE_COR_INSOL:
            for jmode in range(imode - 2, 4):
                if mode[jmode]:
                    # `:487`: the component gate is on IMODE here, not jmode.
                    kij = kij_arr[:, imode, jmode]
                    mask2 = (ndold[:, imode] > eps(imode)) & (ndold[:, jmode] > eps(jmode))
                    if interoff != 1:
                        bterm = jnp.where(mask2, -kij * ndold[:, jmode], bterm)
                        b = b + jnp.where(mask2, -kij * ndold[:, jmode], 0.0)
                    xxx = -bterm * dtz
                    for icp in range(ncp):
                        if component[imode][icp]:
                            mtran = mtran.at[:, icp, imode, jmode].set(
                                _transfer(mdold[:, imode, icp], ndold[:, imode], xxx, mask2)
                            )

        result = solvecoagnucl(
            mask1, a, b, jnp.zeros(nbox), ndold[:, imode], dtz=dtz, fidelity=fidelity
        )
        clamped = clamped.at[:, imode].set(result.clamped)
        failed = failed.at[:, imode].set(result.failed)
        # `:504-506`: masked by mask1 here, unlike the soluble arm.
        nd = nd.at[:, imode].set(
            jnp.where(
                mask1 & (jnp.abs(result.deln) > DN_EPS),
                ndold[:, imode] + result.deln,
                nd[:, imode],
            )
        )
        md, mdt = reset_small(md, mdt, imode, mask1)

    # --- redistribute, `:520-542` ---
    mtranfmi = jnp.zeros((nbox, nmodes, ncp), dtype=nd.dtype)
    mtrantoi = jnp.zeros((nbox, nmodes, ncp), dtype=nd.dtype)
    for imode in range(nmodes):
        if not mode[imode]:
            continue
        for icp in range(ncp):
            if not component[imode][icp]:
                continue
            for jmode in range(nmodes):
                if not mode[jmode]:
                    continue
                mtranfmi = mtranfmi.at[:, imode, icp].add(mtran[:, icp, imode, jmode])
                # ZERO-BASED, as `coag_mode.COAG_MODE` stores it. The Fortran
                # table and the golden's copy of it are 1-based, and passing
                # one of those straight in shifts every destination by a mode:
                # mass from the nucleation mode lands in the Aitken mode. The
                # first draft did exactly that, and `md` was wrong on 62 of
                # setup 1's entries with no obviously wrong-looking number.
                dest = int(coag_mode_table[imode][jmode])
                mtrantoi = mtrantoi.at[:, dest, icp].add(mtran[:, icp, imode, jmode])

    # --- the loop-carried icp update, `:544-580` ---
    for imode in range(nmodes):
        if not mode[imode]:
            continue
        mask1 = nd[:, imode] > eps(imode)
        mdt = mdt.at[:, imode].set(0.0)

        mdcpnew = {}
        for icp in range(ncp):
            if not component[imode][icp]:
                continue
            net = mtrantoi[:, imode, icp] - mtranfmi[:, imode, icp]
            if imode == 0 and icp == 0:
                # `:557-559`: the nucleated mass joins the sulfate component.
                net = net + delh2so4
            mdcpnew[icp] = ndold[:, imode] * mdold[:, imode, icp] + net

        for icp in range(ncp):
            if icp not in mdcpnew:
                continue
            neg = mask1 & (mdcpnew[icp] < 0.0)
            nd = nd.at[:, imode].set(jnp.where(neg, 0.0, nd[:, imode]))
            if not broadcast:
                # `:568`: "set false so not used for other icp values".
                mask1 = mask1 & ~neg
            mask3 = mask1 & (mdcpnew[icp] >= 0.0)
            grown = numerics.safe_divide(mdcpnew[icp], nd[:, imode], mask3)
            md = md.at[:, imode, icp].set(jnp.where(mask3, grown, md[:, imode, icp]))
            mdt = mdt.at[:, imode].set(
                jnp.where(mask3, mdt[:, imode] + md[:, imode, icp], mdt[:, imode])
            )

        # --- the budget diagnostics, `:585-884` ---
        for site_i, site_j, site_cp, name in COAG_BUDGET_SITES:
            if site_i - 1 != imode:
                continue
            icp = site_cp - 1
            if icp >= ncp or not component[imode][icp] or not mode[site_j - 1]:
                continue
            slot = budget.slot(name)
            if slot == NOT_CARRIED:
                continue
            mask3 = mask1 & (md[:, imode, icp] >= 0.0)
            bud_aer_mas = bud_aer_mas.at[:, slot].add(
                jnp.where(mask3, mtran[:, icp, imode, site_j - 1], 0.0)
            )

        md, mdt = reset_small(md, mdt, imode, mask1)

    return CoagResult(nd, md, mdt, bud_aer_mas, ageterm2, clamped, failed)
