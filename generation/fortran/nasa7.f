c     reads gaussian output and summarizes thermodynamic parameters
c     Cp in particular. A 4th order polynomial fit of Cp/R is used
c     to obtain the first 5 of the Nasa7 parameters
c     Nasa parameters 6 and 7 are the enthalpy/R and entropy/R
      implicit none
      real*8 s,ssum(100),r,h(100),g(100),z(100),t(100),sg(100),st(100)
      real*8 sgh(100),sqh(100)
      real*8 ecal(100),cpcal(100),scal(100),cvcal
      real*8 sn(100),sqt(100),vibs(100),sqr(100),sgrm(100),hz(100),sh
      real*8 zC(100),hC(100),gC(100),lnQH(100),lnQG(100)
      real*8 freq(3000),q,sq,sqdiff,sC,sqC,sqdifG,SGrim,ezero
      real*8 redmas(3000),force(3000),compfreq,ak,Total
      real*8 dlnqdtH,dlnqdtG,stmp,gmg0
      real*8 h300,s300
      common/big/freq
      integer*4 n,nt,nfreq,i,n0,nfreqs,nmass,nfrc
      character a*160
      character freqs*15
      character vib*12
      character redmass*15
      character frcconst*15
      character enthalpy*41
      character freeE*44
      character zeroP*42
      character temp*13
      character vibal*16
      character zeroPC*33
      character enthC*32
      character freeC*40
      character cv*47
      data vib/' Vibration  '/
      data freqs/' Frequencies --'/
      data redmass/' Red. masses --'/
      data frcconst/' Frc consts  --'/
      data vibal/' Vibrational    '/
      data enthalpy/' Sum of electronic and thermal Enthalpies'/
      data freeE/' Sum of electronic and thermal Free Energies'/
      data zeroP/' Sum of electronic and zero-point Energies'/
      data zeroPC/' Zero-point correction=          '/
      data  enthC/' Thermal correction to Enthalpy='/
      data  freeC/' Thermal correction to Gibbs Free Energy'/
      data Cv/'                     E (Thermal)             CV'/
      data temp/' Temperature '/
c     r=1.9872d0
      r=0.00198720425864083
c     r=0.001987165d0
      nt=0
      nfreq=0
      nmass=0
      nfrc=0
    1 read(5,100,end=99)a
  100 format(a)
      n=len(a)
      if(a(1:15).eq.freqs) then
      read(a(17:27),"(f11.4)") freq(nfreq+1)
      read(a(40:50),"(f11.4)") freq(nfreq+2)
      read(a(63:73),"(f11.4)") freq(nfreq+3)
      n0=0
      if(freq(nfreq+1).ne.0.0d0) n0=n0+1
      if(freq(nfreq+2).ne.0.0d0) n0=n0+1
      if(freq(nfreq+3).ne.0.0d0) n0=n0+1
      nfreq=nfreq+n0
      else if(a(1:15).eq.redmass) then
      read(a(17:27),"(f11.4)") redmas(nmass+1)
      read(a(40:50),"(f11.4)") redmas(nmass+2)
      read(a(63:73),"(f11.4)") redmas(nmass+3)
      n0=0
      if(redmas(nmass+1).ne.0.0d0) n0=n0+1
      if(redmas(nmass+2).ne.0.0d0) n0=n0+1
      if(redmas(nmass+3).ne.0.0d0) n0=n0+1
      nmass=nmass+n0

