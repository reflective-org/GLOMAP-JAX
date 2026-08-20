! *****************************COPYRIGHT*******************************
! (c) 2026. Validation instrumentation for the GLOMAP-JAX port.
! New code, BSD 3-Clause (see LICENCE). Not part of UKCA.
! *****************************COPYRIGHT*******************************
!
! Description:
!   Accessors for the gas-phase index tables in `ukca_setup_indices`
!   (phase C, task 31).
!
!   `glomap_box_config_mod`'s `init_indices` calls TWO routines per setup: a
!   gas-phase one and a mode one. `glomap_modes_mod.F90` reads what the mode
!   side built; this file reads what the gas side built -- 174 integer scalars
!   plus four arrays of nchemgmax, which between them say where each gas
!   species sits in `s0g`, which of them condense, into which aerosol
!   component, and with what molar mass and molecular diameter.
!
!   Indices come back EXACTLY AS FORTRAN HOLDS THEM: 1-based, with 0 meaning
!   "not present in this setup" (the value every use site guards with
!   `IF (mxxx > 0)`). The 1-based-to-0-based conversion and the -1 sentinel are
!   the port's job, in `physics/gas_indices.py`; doing it here would put the
!   conversion inside the reference the conversion is checked against.
!
!   Typed dispatch by name, exactly as glomap_modes_mod does, and for a
!   stronger reason: 176 named scalars is 176 entry points otherwise. The
!   SELECT CASE below was GENERATED from the same extraction the port uses, so
!   a label and its right-hand side cannot drift apart by a typo -- and
!   `tests/test_gas_indices.py` re-parses this file and asserts that each
!   dispatch label names the variable it reads -- the mutation that would
!   otherwise pass every numeric test on both sides.
!
!   LOGICALs come back as INTEGER 0/1 (`condensable`). MERGE(1, 0, x) is
!   kind-agnostic; f2py's mapping of LOGICAL is not worth relying on.
!
!   REAL(KIND=8), never bare REAL. `mm_gas` and `dimen` are declared bare
!   `REAL` in the vendored module and the whole tree is built at
!   -fdefault-real-8, so they ARE real(8) -- but f2py maps the token `real` to
!   C float whatever the compiler flags say. See glomap_f2py_mod.F90's header.
!
!   Error codes are the binding's usual ones: 0 ok, 1 must restart, 2 shape
!   mismatch, 3 unknown field, 4 not initialised.
!
!   NOT exposed, deliberately: `budget`, `nbudget`, `traqu` and `ntraqu` are
!   assigned only in `ukca_indices_traqu38` / `ukca_indices_traqu9`, which
!   `init_indices` never calls, and `idustdep`, `ndustdep` and `nbudaertot` are
!   assigned nowhere in the tree at all. They have no defined value on any box
!   path, so reading them would capture whatever the loader left in memory and
!   commit it as a golden.
!
! ---------------------------------------------------------------------------
SUBROUTINE wrap_gas_scalar(field, out, ierr)
! One named gas-phase index or count. See the header for why the value is
! 1-based and why 0 is not an index.
USE ukca_setup_indices
USE glomap_f2py_state, ONLY: is_initialised, must_restart
IMPLICIT NONE
CHARACTER(LEN=*), INTENT(IN)  :: field
INTEGER,          INTENT(OUT) :: out
INTEGER,          INTENT(OUT) :: ierr

out = 0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF

ierr = 0
SELECT CASE (TRIM(field))
  CASE ('nchemg');     out = nchemg
  CASE ('ichem');      out = ichem
  CASE ('noffox');     out = noffox
  CASE ('nbudchem');   out = nbudchem
  CASE ('gasbudget');  out = gasbudget
  CASE ('ngasbudget'); out = ngasbudget
  CASE ('nadvg');      out = nadvg
  CASE ('ntrag');      out = ntrag
  CASE ('mox');        out = mox
  CASE ('mnox');       out = mnox
  CASE ('mn2o5');      out = mn2o5
  CASE ('mhno4');      out = mhno4
  CASE ('mhno3');      out = mhno3
  CASE ('mh2o2');      out = mh2o2
  CASE ('mch4');       out = mch4
  CASE ('mco');        out = mco
  CASE ('mch2o');      out = mch2o
  CASE ('mmhp');       out = mmhp
  CASE ('mhono');      out = mhono
  CASE ('mc2h6');      out = mc2h6
  CASE ('metooh');     out = metooh
  CASE ('mmecho');     out = mmecho
  CASE ('mpan');       out = mpan
  CASE ('mc3h8');      out = mc3h8
  CASE ('mpnooh');     out = mpnooh
  CASE ('mpiooh');     out = mpiooh
  CASE ('metcho');     out = metcho
  CASE ('mme2co');     out = mme2co
  CASE ('mmecoh');     out = mmecoh
  CASE ('mppan');      out = mppan
  CASE ('mmeno3');     out = mmeno3
  CASE ('moxs');       out = moxs
  CASE ('mnoys');      out = mnoys
  CASE ('misop');      out = misop
  CASE ('mc2h4');      out = mc2h4
  CASE ('mc2h2');      out = mc2h2
  CASE ('misooh');     out = misooh
  CASE ('mison');      out = mison
  CASE ('mmacr');      out = mmacr
  CASE ('mmacrooh');   out = mmacrooh
  CASE ('mmpan');      out = mmpan
  CASE ('mhacet');     out = mhacet
  CASE ('mmgly');      out = mmgly
  CASE ('mnald');      out = mnald
  CASE ('mhcooh');     out = mhcooh
  CASE ('mmeco3h');    out = mmeco3h
  CASE ('mmeco2h');    out = mmeco2h
  CASE ('mmeoh');      out = mmeoh
  CASE ('mnh3');       out = mnh3
  CASE ('msotwo');     out = msotwo
  CASE ('mmesme');     out = mmesme
  CASE ('mh2so4');     out = mh2so4
  CASE ('mdmso');      out = mdmso
  CASE ('mmsa');       out = mmsa
  CASE ('mcs2');       out = mcs2
  CASE ('mh2s');       out = mh2s
  CASE ('mcos');       out = mcos
  CASE ('mmonoter');   out = mmonoter
  CASE ('msec_org');   out = msec_org
  CASE ('msec_orgi');  out = msec_orgi
  CASE ('mh2o2f');     out = mh2o2f
  CASE ('mq3d');       out = mq3d
  CASE ('mpt');        out = mpt
  CASE ('no');         out = no
  CASE ('no1d');       out = no1d
  CASE ('no3');        out = no3
  CASE ('nno');        out = nno
  CASE ('nno3');       out = nno3
  CASE ('nno2');       out = nno2
  CASE ('nn2o5');      out = nn2o5
  CASE ('nhno4');      out = nhno4
  CASE ('nhno3');      out = nhno3
  CASE ('noh');        out = noh
  CASE ('nho2');       out = nho2
  CASE ('nh2o2');      out = nh2o2
  CASE ('nch4');       out = nch4
  CASE ('nco');        out = nco
  CASE ('nch2o');      out = nch2o
  CASE ('nmeoo');      out = nmeoo
  CASE ('nh2o');       out = nh2o
  CASE ('nmhp');       out = nmhp
  CASE ('nhono');      out = nhono
  CASE ('nc2h6');      out = nc2h6
  CASE ('netoo');      out = netoo
  CASE ('netooh');     out = netooh
  CASE ('nmecho');     out = nmecho
  CASE ('nmeco3');     out = nmeco3
  CASE ('npan');       out = npan
  CASE ('nc3h8');      out = nc3h8
  CASE ('npnoo');      out = npnoo
  CASE ('npioo');      out = npioo
  CASE ('npnooh');     out = npnooh
  CASE ('npiooh');     out = npiooh
  CASE ('netcho');     out = netcho
  CASE ('netco3');     out = netco3
  CASE ('nme2co');     out = nme2co
  CASE ('nmecoo');     out = nmecoo
  CASE ('nmecoh');     out = nmecoh
  CASE ('nppan');      out = nppan
  CASE ('nmeno3');     out = nmeno3
  CASE ('nos');        out = nos
  CASE ('no1ds');      out = no1ds
  CASE ('no3s');       out = no3s
  CASE ('nnoxs');      out = nnoxs
  CASE ('nhno3s');     out = nhno3s
  CASE ('nnoys');      out = nnoys
  CASE ('nisop');      out = nisop
  CASE ('nc2h4');      out = nc2h4
  CASE ('nc2h2');      out = nc2h2
  CASE ('niso2');      out = niso2
  CASE ('nisooh');     out = nisooh
  CASE ('nison');      out = nison
  CASE ('nmacr');      out = nmacr
  CASE ('nmacro2');    out = nmacro2
  CASE ('nmacrooh');   out = nmacrooh
  CASE ('nmpan');      out = nmpan
  CASE ('nhacet');     out = nhacet
  CASE ('nmgly');      out = nmgly
  CASE ('nnald');      out = nnald
  CASE ('nhcooh');     out = nhcooh
  CASE ('nmeco3h');    out = nmeco3h
  CASE ('nmeco2h');    out = nmeco2h
  CASE ('nmeoh');      out = nmeoh
  CASE ('nnh3');       out = nnh3
  CASE ('nsotwo');     out = nsotwo
  CASE ('nmesme');     out = nmesme
  CASE ('nh2so4');     out = nh2so4
  CASE ('ndmso');      out = ndmso
  CASE ('nmsa');       out = nmsa
  CASE ('ncs2');       out = ncs2
  CASE ('nh2s');       out = nh2s
  CASE ('ncos');       out = ncos
  CASE ('nmonoter');   out = nmonoter
  CASE ('nsec_org');   out = nsec_org
  CASE ('nh2o2f');     out = nh2o2f
  CASE ('no3f');       out = no3f
  CASE ('nohf');       out = nohf
  CASE ('nno3f');      out = nno3f
  CASE ('nq3d');       out = nq3d
  CASE ('npt');        out = npt
  CASE ('ndmsemoc');   out = ndmsemoc
  CASE ('ndmstend');   out = ndmstend
  CASE ('nso2eman');   out = nso2eman
  CASE ('nso2embm');   out = nso2embm
  CASE ('nso2emvl');   out = nso2emvl
  CASE ('nso2tend');   out = nso2tend
  CASE ('nso2ddep');   out = nso2ddep
  CASE ('nso2wdep');   out = nso2wdep
  CASE ('nh2so4tend'); out = nh2so4tend
  CASE ('nh2so4ddep'); out = nh2so4ddep
  CASE ('ncoseman');   out = ncoseman
  CASE ('ncosemoc');   out = ncosemoc
  CASE ('ncostend');   out = ncostend
  CASE ('ncs2eman');   out = ncs2eman
  CASE ('ncs2emoc');   out = ncs2emoc
  CASE ('ncs2tend');   out = ncs2tend
  CASE ('ndmsotend');  out = ndmsotend
  CASE ('ndmsoddep');  out = ndmsoddep
  CASE ('nmsatend');   out = nmsatend
  CASE ('nmsaddep');   out = nmsaddep
  CASE ('nterp_em');   out = nterp_em
  CASE ('nterp_tend'); out = nterp_tend
  CASE ('nterp_ddep'); out = nterp_ddep
  CASE ('nsorg_tend'); out = nsorg_tend
  CASE ('nsorg_ddep'); out = nsorg_ddep
  CASE ('nsorg_wdep'); out = nsorg_wdep
  CASE ('iohdms1');    out = iohdms1
  CASE ('iohdms2');    out = iohdms2
  CASE ('ino3dms');    out = ino3dms
  CASE ('idmsooh1');   out = idmsooh1
  CASE ('idmsooh2');   out = idmsooh2
  CASE ('ics2oh');     out = ics2oh
  CASE ('ih2soh');     out = ih2soh
  CASE ('icosoh');     out = icosoh
  CASE ('ntraer');     out = ntraer
  CASE ('nbudaer');    out = nbudaer
CASE DEFAULT;        ierr = 3
END SELECT

END SUBROUTINE wrap_gas_scalar

! ---------------------------------------------------------------------------
SUBROUTINE wrap_gas_real(field, n, out, ierr)
! `mm_gas` and `dimen`, both dimension(nchemgmax). Full width on purpose: the
! entries past nadvg are dummies the Fortran still allocates and still reads
! past in principle, and trimming them would renumber every index.
USE ukca_setup_indices, ONLY: mm_gas, dimen, nchemgmax
USE glomap_f2py_state,  ONLY: is_initialised, must_restart
IMPLICIT NONE
CHARACTER(LEN=*), INTENT(IN)  :: field
INTEGER,          INTENT(IN)  :: n
REAL(KIND=8),     INTENT(OUT) :: out(n)
INTEGER,          INTENT(OUT) :: ierr

out = 0.0_8
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (n /= nchemgmax) THEN
  ierr = 2
  RETURN
END IF

ierr = 0
SELECT CASE (TRIM(field))
CASE ('mm_gas'); out = mm_gas
CASE ('dimen');  out = dimen
CASE DEFAULT;    ierr = 3
END SELECT

END SUBROUTINE wrap_gas_real

! ---------------------------------------------------------------------------
SUBROUTINE wrap_gas_int(field, n, out, ierr)
! `condensable_choice` (a 1-based AEROSOL COMPONENT index, 0 = does not
! condense) and `condensable`, the LOGICAL derived from it, as 0/1.
USE ukca_setup_indices, ONLY: condensable, condensable_choice, nchemgmax
USE glomap_f2py_state,  ONLY: is_initialised, must_restart
IMPLICIT NONE
CHARACTER(LEN=*), INTENT(IN)  :: field
INTEGER,          INTENT(IN)  :: n
INTEGER,          INTENT(OUT) :: out(n)
INTEGER,          INTENT(OUT) :: ierr

out = 0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (n /= nchemgmax) THEN
  ierr = 2
  RETURN
END IF

ierr = 0
SELECT CASE (TRIM(field))
CASE ('condensable_choice'); out = condensable_choice
CASE ('condensable');        out = MERGE(1, 0, condensable)
CASE DEFAULT;                ierr = 3
END SELECT

END SUBROUTINE wrap_gas_int

! ---------------------------------------------------------------------------
SUBROUTINE wrap_nchemgmax(out, ierr)
! A PARAMETER, so it needs no initialisation -- but it is the width every
! caller must allocate to, and hard-coding 50 in Python would be a second
! source of truth for it.
USE ukca_setup_indices, ONLY: nchemgmax
IMPLICIT NONE
INTEGER, INTENT(OUT) :: out, ierr
out  = nchemgmax
ierr = 0
END SUBROUTINE wrap_nchemgmax
