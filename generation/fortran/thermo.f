      SUBROUTINE COMPUT(NFREQ,V,AICUBE,CM,SIGMA,NOC,delh0)
c     thermo_grimme
      IMPLICIT REAL*8 (A-H,O-Z)
      include 'sizes'
      include 'tstate.dc'
      real*8 V(100)
      integer*4 nfreq
      real*8 amoment
      DATA CONVR3/4.578687075E-03/
      DATA CONVRT/1.660531E-24/
      DATA N,T1,TSTEP/11,0.0E+00,1.0E+02/
      real*8 r
c     ideal gas constant
      r=1.987165d0

c     WRITE(6,200)
c     WRITE(10,200)
      200 FORMAT(/5X,'   T   ',4X,'CP (CAL)',5X,'H (KCAL)',4X,'S (CAL)',3X,
     $ 'DH (KCAL)')
      T=T1
      CP=0.0E0
      H=0.0E0
      S=0.0E0
      nend=nfreq
      nstrt=7
      if(v(1).lt.0.0d0) nstrt=8
      if(v(1).gt.0.0d0) then
      if(nconf.gt.1) nend=nfreq-1
      endif
      II=0
      t=300.0d0
c     DO 100 I=1,N
      II=II+1
      cp=0
      h=0
      s=0
      delh=delh0
      IF(T.EQ.0.0E0) GO TO 95
C     TRANSLATION & ROTATION TERMS
      CP=7.948662E0
      H=7.948662E0*T
      S=6.863426E0*LOG10(CM)+18.302469E0*LOG10(T)-4.575617E0*
     $ LOG10(SIGMA)+2.287809E0*LOG10(AICUBE*CONVR3)-2.349265E0
C     VIBRATION TERMS
      DO 50 J=nstrt,nend
      IF(dabs(V(J)).lt.1.0E0) GO TO 50
c     conversion from kcal/mol to cm-1
      kcal_cm=349.755091d0
      write(*,*) 1.0d0/(kcal_cm*r*0.001d0)
      U=abs(1.43879E0*V(J)/T)
      EU=EXP(-U)
      EUI=1.0E0/(1.0E0-EU)
      CP=CP+r*U*U*EU*EUI*EUI
      H=H+r*U*EU*EUI*T
      S=S+r*(U*EU*EUI-LOG(1.0E0-EU))
   50 CONTINUE
      H=H*1.0E-03
      DELH0=DELH0+H
   95 continue
      RETURN
      END

