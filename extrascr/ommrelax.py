from openmm.app import *
from openmm import *
from openmm.unit import *
from openmm.app import element as E
from pdbfixer import PDBFixer
import gzip
import sys, os



def minimize(pdbin, pdbout):
    pdbname = os.path.basename(pdbin).split('.')[0]

    restraint_fc = 50.0 # kJ/mol

    if pdbin.split('.')[-1] == 'gz':
        os.system(f'zcat {pdbin} > tmp.pdb')
        fixer = PDBFixer('tmp.pdb')
    else: 
        fixer = PDBFixer(pdbin)

    #read pdb add missing atom and make the file readable for modeller
    fixer.findMissingResidues()

    # only add missing residues in the middle of the chain, do not add terminal ones
    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    missingResidues = dict()
    for key in keys:
        chain = chains[key[0]]
        if not (key[1] == 0 or key[1] == len(list(chain.residues()))):
            missingResidues[key] = fixer.missingResidues[key]
    fixer.missingResidues = missingResidues


#    print('preparing pdb...')
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    forcefield = ForceField('/data/saakyanh2/WD/PFES/pfes/GB99dms.xml')
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(forcefield)

#    print('preparing topology and FF...')
    system = forcefield.createSystem(modeller.topology, 
                                     nonbondedMethod=CutoffNonPeriodic,
                                     nonbondedCutoff=2 * nanometer,
                                     constraints=HBonds,
                                     hydrogenMass=2*amu,
                                     implicitSolventKappa=0.7 / nanometer,
                                    )


    # prepare simulation
    integrator = LangevinMiddleIntegrator(1*kelvin, 1/picosecond, 0.002*picoseconds)
    integrator.setConstraintTolerance(0.00001)

    equilibration = Simulation(modeller.topology, system, integrator)
    equilibration.context.setPositions(modeller.positions)
    
    # pos_res = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2;")
    # pos_res.addPerParticleParameter("k")
    # pos_res.addPerParticleParameter("x0")
    # pos_res.addPerParticleParameter("y0")
    # pos_res.addPerParticleParameter("z0")

    # for ai, atom in enumerate(modeller.topology.atoms()):
    #     if atom.element is not E.hydrogen:
    #         x = modeller.positions[ai][0].value_in_unit(nanometers)
    #         y = modeller.positions[ai][1].value_in_unit(nanometers)
    #         z = modeller.positions[ai][2].value_in_unit(nanometers)
    #         pos_res.addParticle(ai, [restraint_fc, x, y, z])

    # equilibration.addForce(pos_res)
    
    # minimize
    print(f'Minimizing {pdbname}...')
    equilibration.minimizeEnergy(tolerance=Quantity(value=0.01, unit=kilojoule/mole), maxIterations=2000)
    crd = equilibration.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(modeller.topology, crd, open(pdbout,'w'))



pdbdir = str(sys.argv[1])
outdir = str(sys.argv[2])
os.makedirs(outdir, exist_ok=True)
for pdb in os.listdir(pdbdir):
    pdbname = os.path.basename(pdb).split('.')[0]
    pdbin_path = os.path.join(pdbdir,pdb)
    pdbout_path = os.path.join(outdir,pdbname+'.pdb')
    minimize(pdbin_path, pdbout_path)
