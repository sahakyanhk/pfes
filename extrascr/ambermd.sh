#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --mem=50g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=7-00:00:00



set -e

display_usage() {
    echo -e "\n this is a test MD script"
    echo "-i input pdb file"
    echo "-o output directory"
}

if [  $# -le 1 ]
then
    display_usage
    exit 1
fi

#set defaults
PDB=ranked_0.pdb
OUT_DIR=MDOUT

while getopts i:o: flag
do
    case "${flag}" in
	  i) PDB=${OPTARG};;
	  o) OUT_DIR=${OPTARG};;
    esac
done




mkdir -p $OUT_DIR  
cp $PDB $OUT_DIR/input.pdb
cd $OUT_DIR 

mkdir -p inp parm min equil md 



module add amber/20-gpu

pdb4amber -i input.pdb -o parm/prot.pdb -y


cat > ./inp/tleap.in <<EOF
source leaprc.protein.ff19SB
source leaprc.gaff
source leaprc.water.tip3p
loadamberparams frcmod.ionsjc_tip3p
set default PBradii mbondi3
system = loadpdb parm/prot.pdb
savepdb system parm/system.pdb
saveamberparm system parm/prot.prmtop parm/prot.inpcrd
solvateOct system TIP3PBOX 10.0 
addions system Cl- 0
addions system K+  0
savepdb system parm/prot_solv.pdb 
saveamberparm system parm/prot_solv.prmtop parm/prot_solv.inpcrd
quit
EOF

cat > ./inp/pmd_hmr <<EOF
hmassrepartition
outparm parm/complex_hmr.parm7
EOF


cat > ./inp/min.in <<EOF
Minimize all atoms
  System minimization:
&cntrl
   imin=1, ntmin=1, nmropt=0, drms=0.1
   maxcyc=8000, ncyc=4000,
   ntx=1, irest=0,
   ntpr=1000, ntwr=1000, iwrap=0,
   ntc=2, ntf=2, ntb=1, cut=10.0, nsnb=25,
   igb=0,
   ibelly=0, ntr=0, restraintmask=':1-363',
  restraint_wt=10.0,
&end
/
EOF



cat > ./inp/heat.in <<EOF
heat NPT 0.5ps
  Heating System
&cntrl
   imin=0, nmropt=1,
   ntx=1, irest=0,
   ntpr=10000, ntwr=10000, ntwx=10000, iwrap=1,
   ntf=2, ntb=1, cut=10.0, nsnb=25,
   igb=0, ibelly=0, ntr=1,
   nstlim=100000, nscm=1000, dt=0.001,
   ntt=3, temp0=310.15, tempi=0.0, tautp=1.0
   ntc=2, restraintmask="@CA,N,C,O",
   restraint_wt=10.0,
&end

&wt type='REST', istep1=0, istep2=0, value1=1.0, value2=1.0, &end
&wt type='TEMP0', istep1=0, istep2=250000, value1=0.0, value2=310.15, &end
&wt type='END' &end
/
EOF

cat > ./inp/equil1.in <<EOF
equil 1
 &cntrl
  imin=0,irest=1,ntx=5,
  nstlim=100000,dt=0.002,
  ntc=2,ntf=2,
  cut=10.0, ntb=2, ntp=1, taup=1.0,
  ntpr=10000, ntwx=10000,
  ntt=3, gamma_ln=2.0,
  temp0=310.15, iwrap=1,
  ntr=1, restraintmask="@CA,N,C,O",
  restraint_wt=10.0,
/

EOF

cat > ./inp/equil2.in <<EOF
equil 2
&cntrl
	imin=0,irest=1, ntx=5,
	ioutfm=1,
	nstlim=500000, dt=0.002,
	cut =10.0, ntb=2, ntp=1,
	ntc=2, ntf= 2, ig=-1,
	ntwr = 10000,
	ntpr = 10000, ntwx = 10000,
	ntwe = 10000,
	iwrap = 1,
	ntr = 0,
	ntwprt = 0,
	ntt = 3,gamma_ln = 2.0,restraintmask="@CA,N,C,O",
	restraint_wt=10.0,
	tempi = 310.15,
	temp0 = 310.15,
/
EOF

cat > ./inp/md.in <<EOF
&cntrl
  imin=0,
  irest=1,
  ntx=5,
  nstlim=1250000000,
  dt=0.004,
  ntc=2,
  ntf=2,
  tol=0.00001,
  cut=10.0,
  ntb=2,
  ntp=1,
  taup=2.0,
  ntpr=10000,
  ntwx=10000,
  ntwr = 10000, 
  ntwe = 10000,
  iwrap=1,
  ntt=3,
  barostat=2,
  gamma_ln=2.0,
  temp0=310.15,
  tempi=310.15,
  ig=-1,
/

EOF



tleap -f inp/tleap.in
parmed -n parm/prot_solv.prmtop inp/pmd_hmr

echo "input files are created"

echo "MINIMIZATION..."
pmemd.cuda -O -i inp/min.in  -p parm/prot_solv.prmtop  -c parm/prot_solv.inpcrd -r min/minim.rst -ref parm/prot_solv.inpcrd -inf min/mdinfo.min -o min/minim.out

echo "HEATING..."
pmemd.cuda -O -i inp/heat.in  -p parm/prot_solv.prmtop  -c min/minim.rst -r equil/heat.rst -ref min/minim.rst -inf equil/mdinfo.heat -o equil/heat.out  -x equil/heat.nc

echo "EQUILIBRATION 1..."
pmemd.cuda -O -i inp/equil1.in  -p parm/prot_solv.prmtop -c equil/heat.rst -r equil/equil1.rst -ref equil/heat.rst -o equil/equil1.out -inf equil/mdinfo.eq1 -x equil/eq1.nc

echo "EQUILIBRATION 2..."
pmemd.cuda -O -i inp/equil2.in  -p parm/prot_solv.prmtop  -c equil/equil1.rst -r equil/equil2.rst -ref equil/equil1.rst -o equil/equil2.out -inf equil/mdinfo.eq2 -x equil/eq2.nc

echo "RUNNING MD!"
pmemd.cuda -O -i inp/md.in  -p parm/complex_hmr.parm7 -c equil/equil2.rst -r md/md.rst -ref equil/equil2.rst  -o md/md.out -inf mdinfo.md -e md/md.en -x md/md.nc 


cpptraj << EOF
parm parm/prot_solv.prmtop
trajin md/md.nc
autoimage
strip :Cl-,WAT,K+
rmsd first out backbone_rmsd.dat @C,CA,N,O time 0.004 mass
trajout md/prot_traj.xtc xtc offset 25
go 
exit
EOF
