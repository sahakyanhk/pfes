import argparse
import os, re
import pandas as pd
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Analyse PFES")
parser.add_argument('-wd', '--workdir', type=str, help='working directory')
parser.add_argument('-pdb', '--pdbdir', type=str, help='directory with pdb files', default='pdb')
parser.add_argument('-t', '--traj', type=str, help='make backbone trajectory', default='pfestraj.pdb')
parser.add_argument('-l', '--log', type=str, help='log file name', default='progress.log')
parser.add_argument('-o', '--outdir', type=str, help='output directory name', default='visual_pfes_results')
args = parser.parse_args()

wd = os.path.abspath(args.log)
outdir = args.outdir 
pdbdir = args.pdbdir
plotdir = os.path.join(outdir, 'plots/')
bestpdb = os.path.join(outdir, 'bestpdb/')
log = pd.read_csv(args.log, sep='\t')

print(pdbdir)

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


def make_plots(log):
    os.makedirs(plotdir, exist_ok=True)

    for colname, coldata in log.iteritems(): 
        if not colname in ['seq', 'sequence', 'dssp', 'index', 'id', 'genindex']:
            plt.plot(coldata,'.')
            plt.legend([colname], loc ="upper left")
            plt.savefig(plotdir + colname + '.png')
            plt.close()



def backbone_traj(log, pdbdir, trajout=args.traj):
    """
    make trajectory from C-alpha atoms
    TO DO: 
    1. find the biggest pdb in folder, or from progress log. \
    Make it initial topology that should be uploaded first or append in in the begining of the trajectory file
    2. save trajectory in xyz format to save space. 
    """
    
    bestlog = log.drop_duplicates(subset='genindex')

    pfeslen = len(bestlog)

    os.makedirs(bestpdb, exist_ok=True)
    if os.path.isdir(bestpdb) and len(os.listdir(bestpdb)) != pfeslen:
        shutil.rmtree(bestpdb)
        os.makedirs(bestpdb, exist_ok=True)
        
        for genndx, pdbid in tqdm(zip(bestlog.genindex, bestlog.id), total=len(bestlog)):
            try:
                shutil.copy(pdbdir + pdbid +'.pdb', bestpdb +'/'+ genndx + '.pdb')
            except FileNotFoundError:
                print(pdbdir + pdbid +'.pdb is missing' )
        print(f'The best folds from {pfeslen} generations are selected')
    else: 
        print(f'The best folds from {pfeslen} generations are selected')


    print("extracting backbone coordinates...")
    i=0
    PDB_A, PDB_B, lastBB = [], [], [] 
    for pdb in tqdm(sorted_alphanumeric(os.listdir(bestpdb))):
        with open(os.path.join(bestpdb, pdb), 'r') as file:
            pdb_txt = file.read()
        bb_chain_A, bb_chain_B = [], []
        for line in pdb_txt.splitlines():
            col = line.split()
            if (col[0] == 'ATOM' and (
                col[2] == 'N' or 
                col[2] == 'CA' or 
                col[2] == 'C' or 
                col[2] == 'O') and 
                col[4] == 'A'):
                bb_chain_A.append(line + '\n') 
            if (col[0] == 'ATOM' and (
                col[2] == 'N' or 
                col[2] == 'CA' or 
                col[2] == 'C' or 
                col[2] == 'O') and 
                col[4] == 'B'):
                bb_chain_B.append(line + '\n')

        lastresidue=''.join([str(elem) for elem in bb_chain_A[-4:]]) # keep the last four lines to repeate them and make numer of atom in all models equal
        PDB_A.append(bb_chain_A) #save chain A
        PDB_B.append(bb_chain_B) #save chain A
        lastBB.append(lastresidue) #save bb of the last residue of chain A 

    topmax = max([len(i) for i in PDB_A])
    for pdbA, pdbB in zip(PDB_A, PDB_B): 
        if len(pdbA) == topmax:
            toppdb = ''.join(pdbA + pdbB)
            break
    
    print('writing backbone trajectory...')
    with open(os.path.join(outdir, trajout), 'w') as f:
        i=1
        f.write(f'MODEL        {i}\n' + toppdb + 'TER\nENDMDL\n')
        for chA, chB, lastresBB in tqdm(zip(PDB_A, PDB_B, lastBB), total=len(PDB_A)): 
            i+=1
            dumlines = lastresBB  * int((topmax - len(chA)) / 4)
            f.write(f'MODEL        {i}\n' + ''.join(chA) + dumlines + 'TER\n' + ''.join(chB) + 'TER\nENDMDL\n')



make_plots(log)
backbone_traj(log, pdbdir)


