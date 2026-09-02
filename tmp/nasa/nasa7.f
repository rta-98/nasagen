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
      else if(a(1:15).eq.frcconst) then
      read(a(17:27),"(f11.4)") force(nfrc+1)
      read(a(40:50),"(f11.4)") force(nfrc+2)
      read(a(63:73),"(f11.4)") force(nfrc+3)
      n0=0
      if(force(nfrc+1).ne.0.0d0) n0=n0+1
      if(force(nfrc+2).ne.0.0d0) n0=n0+1
      if(force(nfrc+3).ne.0.0d0) n0=n0+1
      nfrc=nfrc+n0
      else if(a(1:12).eq.vib) then
      read(a(62:69),"(f8.3)") s
      ssum(nt)=ssum(nt)+s
      else if(a(1:42).eq.zeroP) then
      read(a(51:65),"(f15.6)") z(nt)
      else if(a(1:41).eq.enthalpy) then
      read(a(51:65),"(f15.6)") h(nt)
      else if(a(1:44).eq.freeE) then
      read(a(51:65),"(f15.6)") g(nt)
      else if(a(1:33).eq.zeroPC) then
      read(a(43:58),"(f15.6)") zC(nt)
      else if(a(1:32).eq.enthC) then
      read(a(43:58),"(f15.6)") hC(nt)
      else if(a(1:40).eq.freeC) then
      read(a(43:58),"(f15.6)") gC(nt)
c     heat capacity section
      else if(a(1:47).eq.Cv) then
      read(5,100) a
      read(5,100) a
      read(a(22:31),"(f10.3)") ecal(nt)
      read(a(41:50),"(f10.3)") cvcal
      cpcal(nt)=cvcal+1000.0d0*r
      read(a(60:69),"(f10.3)") scal(nt)
      else if(a(1:16).eq.vibal) then
      read(a(61:69),"(f9.3)") vibs(nt)
      else if(a(1:13).eq.temp) then
c     new temperature
      nt=nt+1
      read(a(14:22),"(f9.3)") t(nt)
      nfreqs=nfreq
      nfreq=0
      nmass=0
      nfrc=0
      ssum(nt)=0.0d0
      z(nt)=0.0d0
      h(nt)=0.0d0
      g(nt)=0.0d0
      endif
      goto 1
   99 continue
      Total=z(1)-zC(1)
c     write(6,*) Total
c     do 18 n=1,nt
c     write(6,124) zC(n),hC(n),gC(n)
c 124 format(3f15.6)
c  18 continue
c     mass in amu, force in mdyne/Angstrum
c     1 mdyne/angstrum=143.836 kcal/mol angstrum^2
c     freq=108.5913586*sqrt(k/mu)
c     do 122 n=1,nfreqs
c     ak=143.836d0*force(n)
c     compfreq=108.5913586d0*sqrt(ak/redmas(n))
c     write(6,123) n,freq(n),redmas(n),force(n),force(n)/15.569141,ak
c    $ ,compfreq,
c    $ freq(n)/sqrt(ak/redmas(n))
  123 format(i5,6f12.5)
  122 continue
      nfreq=nfreqs
      do 5 n=1,nt
      st(n)=(h(n)-g(n))/t(n)
    5 continue
      h300=627.51*(h(4)-z(1))
      s300=627.51*1000.0d0*(st(4)-st(3))
  102 format(10f15.6)
      write(6,103)
  103 format('    T         Zero_Point      Enthalpy     Free_Energy',
     $ '       Full_S         S             Cp              H',
     $ '               G')
      do 120 n=1,nt
      write(6,104) t(n),z(n),h(n),g(n),1000.0d0*627.51d0*st(n),
     $ scal(n),cpcal(n),627.51d0*(h(n)-z(n)),627.51d0*(g(n)-z(n))
  120 continue
  104 format(f9.3,10f15.6)
      stop
      end
