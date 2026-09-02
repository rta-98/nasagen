#!/bin/csh
#script to calculate a set of spin natural orbitals 
# $1 = checkpoint file name
set c=$#argv
if($c<1) then
echo need to specify a .chk file name 
exit(1)
else

echo $g16root
cp $1.chk thermo2000.chk
$g16root/g16/g16 <<EOF >{$1}_2000.log
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=1 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=100 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=200 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=300 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=400 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=500 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=600 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=700 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=800 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=900 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=1000 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=1100 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=1200 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=1300 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=1400 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=1500 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=1600 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=1700 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=1800 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=1900 geom=allcheck

--Link1--
%mem=24GB
%nprocshared=8
%chk=thermo2000.chk
# freq=(readfc) temperature=2000 geom=allcheck

EOF

endif
