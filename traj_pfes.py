import argparse
import os, re
import pandas as pd
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Analyse PFES")
parser.add_argument('-l', '--log', type=str, help='log file name', default='progress.log', required=True) #rename log to pfes traj
parser.add_argument('-pdb', '--pdbdir', type=str, help='directory with pdb files', default='pdb')
parser.add_argument('-t', '--traj', type=str, help='make backbone trajectory', default='pfestraj.pdb')
parser.add_argument('-o', '--outdir', type=str, help='output directory name', default='visual_pfes_results')

args = parser.parse_args()


log = pd.read_csv(args.log, sep='\t', comment='#')
outdir = args.outdir 
pdbdir = os.path.join(args.pdbdir)
plotdir = os.path.join(outdir, 'plots/')



print(pdbdir)

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


def make_plots(log):
    print('processing evolution trajectory do make plots')
    os.makedirs(plotdir, exist_ok=True)

    for colname, coldata in log.iteritems(): 
        if not colname in ['seq', 'sequence', 'ss', 'genindex' ,'dssp', 'index', 'id', 'genndx']:
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
    bestpdb = os.path.join(outdir, 'bestpdb/')
    bestlog = log.drop_duplicates(subset='genndx')
    pfeslen = len(bestlog)

    os.makedirs(bestpdb, exist_ok=True)
    if os.path.isdir(bestpdb) and len(os.listdir(bestpdb)) != pfeslen:
        shutil.rmtree(bestpdb)
        os.makedirs(bestpdb, exist_ok=True)
        print(f'Selecting best folds from {pfeslen} generations') # do not copy files, just make a list and extract BB coords from pdb dir
        for genndx, pdbid in tqdm(zip(bestlog.genndx, bestlog.id), total=len(bestlog)):
            try:
                shutil.copy(pdbdir +'/' + pdbid +'.pdb', bestpdb +'/'+ genndx + '.pdb')
            except FileNotFoundError:
                #print(pdbid +'.pdb is missing' )
                pass
    else: 
        print(f'The best folds from {pfeslen} generations are selected')


    print("extracting backbone coordinates...")
    i=0
    PDB_A, PDB_B, lastBB_A, lastBB_B = [], [], [], []
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

        lastresidueA=''.join([str(elem) for elem in bb_chain_A[-4:]]) # keep the last four lines to repeate them and make numer of atom in all models equal
        lastresidueB=''.join([str(elem) for elem in bb_chain_B[-4:]]) # keep the last four lines to repeate them and make numer of atom in all models equal

        PDB_A.append(bb_chain_A) #save chain A
        PDB_B.append(bb_chain_B) #save chain A
        lastBB_A.append(lastresidueA) #save bb of the last residue of chain A 
        lastBB_B.append(lastresidueB) #save bb of the last residue of chain B

    topmax_A = max([len(i) for i in PDB_A])
    topmax_B = max([len(i) for i in PDB_A])
    for pdbA, pdbB in zip(PDB_A, PDB_B): 
        if len(pdbA) == topmax_A:
            toppdb = ''.join(pdbA + pdbB)
            break
    
    print('writing backbone trajectory...')
    with open(os.path.join(outdir, trajout), 'w') as f:
        i=1
        f.write(f'MODEL        {i}\n' + toppdb + 'TER\nENDMDL\n')
        for chA, chB, lastresBB_A, lastresBB_B in tqdm(zip(PDB_A, PDB_B, lastBB_A, lastBB_B), total=len(PDB_A)): 
            i+=1
            dumlinesA = lastresBB_A  * int((topmax_A - len(chA)) / 4)
            dumlinesB = lastresBB_B  * int((topmax_B - len(chB)) / 4)

            f.write(f'MODEL        {i}\n' + ''.join(chA) + dumlinesA + 'TER\n' + ''.join(chB) + dumlinesB + 'TER\nENDMDL\n')



make_plots(log)
backbone_traj(log, pdbdir)


