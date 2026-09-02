c     reads gaussian output and corrects vibrational entropy
      implicit none
      real*8 s,ssum(100),r,h(100),g(100),z(100),t(100),sg(100),st(100)
      real*8 sgh(100),sqh(100)
      real*8 sn(100),sqt(100),vibs(100),sqr(100),sgrm(100),hz(100),sh
      real*8 zC(100),hC(100),gC(100),lnQH(100),lnQG(100)
      real*8 freq(3000),q,sq,sqdiff,sC,sqC,sqdifG,SGrim,ezero
      real*8 redmas(3000),force(3000),compfreq,ak,Total
      real*8 dlnqdtH,dlnqdtG,stmp
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
      data temp/' Temperature '/
c     r=1.9872d0
      r=0.001987165d0
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
      hz(n)=0.0d0
      sgrm(n)=0.0d0
      do 10 i=1,nfreq
      if(freq(i).gt.0.0d0) then
      call free_energy(freq(i),t(n),ezero,sh,sq,SGrim)
c     write(6,105) i,t(n),freq(i),1000*(s-sq),1000*sq
  105 format(i5,4f10.3)
      else
      ezero=0.0d0
      sh=0.0d0
      sq=0.0d0
      SGrim=0.0d0
      dlnqdtH=0.0d0
      dlnqdtG=0.0d0
      endif
c     write(6,*) i,s,sq
      hz(n)=hz(n)+ezero
      sgh(n)=sgh(n)+sh
      sqh(n)=sqh(n)+sq
      sgrm(n)=sgrm(n)+SGrim
   10 continue
    5 continue
      write(6,103)
  103 format('    T         Zero_Point      Enthalpy     Free_Energy',
     $ '        Full_S         GS         Free_Grimme     H0+Full_S',
     $ '         H-H0            SqH')
      do 120 n=1,nt
c     remove vibrational entropy
      stmp=st(n)-sgh(n)/627.51d0
c     add back in the grimme entropy
      stmp=stmp+sgrm(n)/627.51d0
      write(6,104) t(n),z(n),h(n),g(n),1000.0d0*st(n),1000.0d0*stmp,
     $ z(n)-t(n)*stmp,z(n)-t(n)*st(n),h(n)-z(n),1000.0d0*sqh(n)/627.51d0
  120 continue
  104 format(f9.3,10f15.6)
      stop
      end
c     subroutine free_energy(e,t,ezero,sh,sq,SGrim)
      subroutine free_energy(e,t,ezero,shs,sqs,SGrim)
      implicit none
      real*8 e,t,exps(3000),ezero,ek,probs(3000)
      integer*4 i,j,n,nsum,ie,msum
      real*8 r,q,dqdt,dlnqdt,u,sh,sq,plnp,dlnqdtH,dlnqdtG
      real*8 hbar,eh,epersec,pi,moi,redmoi,boltz
      real*8 rotQG,SqG,Erot,dqdtG,weight,SGrim
      real*8 eus,qhos,dlnqs,shs,sqs,se
      real*8 wf
      nsum=50
      r=0.001987165d0
c     write(6,*)n
c     ek in kcal/mol, e in cm-1
      ek=e/349.755091d0
      ezero=0.5d0*ek
      do 5 i=0,nsum
      exps(i+1)=dexp(-((ek*(dfloat(i)+0.5d0)-ezero))/(r*t))
    5 continue
      q=0.0d0
      do 10 i=0,nsum
      q=q+exps(i+1)
   10 continue
      do 20 i=0,nsum
      probs(i+1)=exps(i+1)/q
   20 continue
c     plnp
      plnp=0.0d0
      do 25 i=0,nsum
      plnp=plnp+probs(i+1)*dlog(probs(i+1))
   25 continue
      dqdt=0.0d0
      do 30 i=0,nsum
      dqdt=dqdt+(ek*(dfloat(i)+0.5d0)-ezero)*exps(i+1)
   30 continue
      dqdt=dqdt/(r*t*t)
      dlnqdtH=dqdt/q
c     write(6,*) dqdt,q,dlnqdtH
      sq=r*dlog(q)
      se=r*t*dlnqdtH
      sh=se+sq
c     write(6,321) sq,se,sh
  321 format('sq,se,sh=',3f15.6)
c     harmonic vibration
      eus=dexp(-ek/(r*t))
      qhos=1.0d0/(1.0d0-eus)
      dlnqs=(ek/(r*t))*eus*qhos
      sqs=r*dlog(qhos)
      shs=r*dlnqs+sqs
c     Grimme entropy correction
c     if vib_en in hartree
c     Planck's constant (h)  = 6.62606957 * 10-34  Joule-secs
      hbar = 1.0545718d-34
c     convert energy to hartree
      eh=e*4.55633525275523d-06
c     convert from hartrees to 1/s
      epersec=eh*6.57966d+15
      pi=3.14159265359d0
      moi=hbar/(4*pi*epersec)
      redmoi=moi*1.0d-44/(moi+1.0d-44)
      boltz= 1.3806488d-23
      rotQG = sqrt(8.0d0*(pi**3)*redmoi*boltz*T/hbar**2)
      SqG = r*dlog(rotQG)
      Erot=0.5d0*r*t
      dqdtG=Erot
      dqdtG = dqdtG/(r*t*t)
c     dlnQ/dT
      dlnqdtG = dqdtG/rotQG
c     SqG=SqG+r*t*dlnqdtG
c     write(6,*) dqdtG,rotQG,dlnqdtG
c     Combine harmonic with Grimme
      wf=t*100.0d0/298.0d0
      weight=1.0/(1.0+(wf/e)**4)
c     weight=1.0/(1.0+(100.0/e)**4)
c     weight=1.0/(1.0+(300.0/e)**4)
      SGrim=weight*sqs+(1.0-weight)*SqG
c     write(6,*) 1000.0d0*s,1000.0d0*r*t*dlnqdt,1000.0d0*sq,
c    $ -r*plnp*1000.0d0
c     write(6,123) sh,sq,shs,sqs,SGrim,SqG
  123 format('sss=',6f12.8)
      return
      end
