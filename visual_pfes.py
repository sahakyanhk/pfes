import argparse
import os, re
import pandas as pd
import numpy as np
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors

import MDAnalysis as mda
from MDAnalysis.analysis import align

import warnings


parser = argparse.ArgumentParser(description="Analyse PFES")
parser.add_argument('-l', '--log', type=str, help='log file name', default='progress.log', required=True) #rename log to pfes traj
parser.add_argument('-s', '--pdbdir', type=str, help='directory with pdb files', default='structures')
parser.add_argument('-t', '--traj', type=str, help='make backbone trajectory', default='pfestraj.pdb')
parser.add_argument('-o', '--outdir', type=str, help='output directory name', default='visual_pfes_results')

args = parser.parse_args()

log = pd.read_csv(args.log, sep='\t', comment='#')
#log['ndx']  = log.index

bestlog = log.groupby('gndx').head(1)

outdir = args.outdir 
pdbdir = os.path.join(args.pdbdir)
plotdir = os.path.join(outdir, 'plots/')
trajpath = os.path.join(outdir, args.traj)

os.makedirs(outdir, exist_ok=True)
bestlog.to_csv(os.path.join(outdir, 'bestlog.tsv'), sep='\t', index=False, header=True)


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


def make_plots(log, bestlog):
    print('processing evolution trajectory to make plots')
    
    ms=0.5
    lw=1.0
    dpi=500

    os.makedirs(plotdir, exist_ok=True)
    for colname in log.keys(): 
       if not colname in ['seq', 'sequence', 'ss', 'genindex' ,'dssp', 'mutation', 'index', 'id', 'prev_id', 'gndx']:
            plt.plot(log[colname],'.', markersize=ms)
            plt.plot(bestlog[colname],'-', linewidth=lw)
            plt.legend([colname], loc ="upper left")
            plt.savefig(plotdir + colname + '.png', dpi=dpi)
            plt.clf()

    #======================= seconday structure plot =======================#
    max_seq_len = int(max(bestlog.seq_len))
    bestlog_len = len(bestlog)

    sse = np.empty((bestlog_len, max_seq_len), dtype='U1')
    i=0
    for ss in bestlog.ss:
        sse[i] = list(ss + "X"*(max_seq_len-len(ss)))
        i+=1

    def sse_to_num(sse):
        num = np.empty(sse.shape, dtype=int)
        num[sse == 'C'] = 0
        num[sse == 'E'] = 1
        num[sse == 'B'] = 2
        num[sse == 'S'] = 3
        num[sse == 'T'] = 4
        num[sse == 'H'] = 5
        num[sse == 'G'] = 6
        num[sse == 'I'] = 7
        num[sse == 'X'] = 8
        num[sse == 'F'] = 9

        return num

    sse_digit = sse_to_num(sse)


    color_assign = {
        r"coil": "grey",
        r"$\beta$-sheet": "yellow",
        r"$\beta$-bridge": "orange",
        r"bend": "cyan",
        r"turn": "brown",
        r"$3_{10}$-helix": "purple",
        r"$\alpha$-helix": "pink",
        r"$\pi$-helix": "blue",
        r"dum": "white",
        r"dum1": "red"
        }


    cmap = colors.ListedColormap(color_assign.values())
    ticks = np.arange(0, len(bestlog)+1, 1000)

    plt.figure(figsize=(12, 8), dpi=dpi)
    plt.imshow(sse_digit.T, origin='lower',   interpolation='nearest', aspect='auto')
    plt.xticks(ticks, ticks.astype(int))
    plt.xlabel("# mutations x Pop size")
    plt.ylabel("Residue")

    custom_lines = [
        Line2D([0], [0], color=cmap(i), lw=4) for i in range(len(color_assign)-1)]

    plt.legend(
        custom_lines, color_assign.keys(), loc="upper center",
        bbox_to_anchor=(0.5, 1.1), ncol=len(color_assign), fontsize=8)

    plt.savefig(os.path.join(outdir,'Secondary_structures.png'), dpi=dpi) 


    #======================= Summary plot =======================#
    fig, axs = plt.subplots(3,2, figsize=(14, 10))

    fig.suptitle(None)
    
    if 'num_inter_conts' in log.keys():
        axs[0,0].plot(log.num_inter_conts, '.', markersize=ms)
        axs[0,0].plot(bestlog.num_inter_conts, '-', linewidth=lw)
#        axs[0,0].plot(averlog.num_inter_conts, '-', linewidth=lw)
        axs[0,0].set(xlabel=None, ylabel='num_inter_conts')
    else: 
        axs[0,0].plot(log.mean_plddt, '.', markersize=ms)
        axs[0,0].plot(bestlog.mean_plddt, '-', linewidth=lw)
        axs[0,0].set(xlabel=None, ylabel='mean_plddt')
    
    axs[1,0].plot(log.ptm, '.', markersize=ms)
    axs[1,0].plot(bestlog.ptm, '-', linewidth=lw)
    axs[1,0].set(xlabel=None, ylabel='ptm')
    
    axs[2,0].plot(log.score,  '.', markersize=ms)
    axs[2,0].plot(bestlog.score,  '-', linewidth=lw)
    axs[2,0].set(xlabel='# mutation', ylabel='score')
        
    axs[0,1].plot(log.seq_len, '.', markersize=ms)
    axs[0,1].plot(bestlog.seq_len, '-', linewidth=lw)
    axs[0,1].set(xlabel=None, ylabel='seq_len')
    
    axs[1,1].plot(log.prot_len_penalty, '.', markersize=ms)
    axs[1,1].plot(bestlog.prot_len_penalty, '-', linewidth=lw)
    axs[1,1].set(xlabel=None, ylabel='max_helix_penalty')

    axs[2,1].plot(log.num_conts, '.', markersize=ms)
    axs[2,1].plot(bestlog.num_conts, '-', linewidth=lw)
    axs[2,1].set(xlabel='# mutation', ylabel='num_conts')
    
    #for ax in axs.flat:
    #   ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    #for ax in axs.flat:
    #   ax.label_outer()

    fig.savefig(os.path.join(outdir,'Summary_plot.png'), dpi=dpi)


def backbone_traj(bestlog, pdbdir):
    """
    make trajectory from C-alpha atoms
    TO DO: 
    1. find the biggest pdb in folder, or from progress log. \
    Make it initial topology that should be uploaded first or append in in the begining of the trajectory file
    2. save trajectory in xyz format to save space. 
    """
    bestpdb = os.path.join(outdir, 'bestpdb/')
    bestlog = bestlog.drop_duplicates(subset = 'sequence')
    pfeslen = len(bestlog)

    os.makedirs(bestpdb, exist_ok=True)
    if os.path.isdir(bestpdb) and len(os.listdir(bestpdb)) != pfeslen:
        shutil.rmtree(bestpdb)
        os.makedirs(bestpdb, exist_ok=True)
        print(f'{pfeslen} uniqs sequences with the best folds are selected') # do not copy files, just make a list and extract BB coords from pdb dir
        for gndx, pdbid in tqdm(zip(bestlog.gndx, bestlog.id), total=len(bestlog)):
            try:
                shutil.copy(pdbdir +'/' + pdbid +'.pdb', bestpdb +'/'+ gndx + '.pdb')
            except FileNotFoundError:
                print(pdbid +'.pdb is missing' )
                pass
    else: 
        print(f'The best folds from {pfeslen} generations are selected')

    #if single_chian:
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
    topmax_B = max([len(i) for i in PDB_B])
    for pdbA, pdbB in zip(PDB_A, PDB_B): 
        if len(pdbA) == topmax_A:
            toppdb = ''.join(pdbA + pdbB)
            break
    
    print('preparing backbone trajectory...')
    with open(outdir+'/.tmp.pdb', 'w') as f:
        i=1
        f.write(f'MODEL        {i}\n' + toppdb + 'TER\nENDMDL\n')
        for chA, chB, lastresBB_A, lastresBB_B in tqdm(zip(PDB_A, PDB_B, lastBB_A, lastBB_B), total=len(PDB_A)): 
            i+=1
            dumlinesA = lastresBB_A  * int((topmax_A - len(chA)) / 4)
            dumlinesB = lastresBB_B  * int((topmax_B - len(chB)) / 4)

            f.write(f'MODEL        {i}\n' + ''.join(chA) + dumlinesA + 'TER\n' + ''.join(chB) + dumlinesB + 'ENDMDL\n')

    print('writing aligned backbone trajectory...')
    traj = mda.Universe(outdir+'/.tmp.pdb')
    top = traj.select_atoms('protein')

    warnings.filterwarnings("ignore")
    align.AlignTraj(traj,  # trajectory to align
                    top,  # reference
                    select='chainID A',  # selection of atoms to align
                    filename=trajpath,  # file to write the trajectory to
                ).run()

    os.remove(outdir+'/.tmp.pdb')


make_plots(log, bestlog)
#backbone_traj(bestlog, pdbdir)

"""

#ete3
from ete3 import Tree
import pandas as pd
import numpy as np



def extract_tree(head_size = 1e7, log_path = '/home/saakyanh2/WD/PFES/OUTPUTS/sft04292024/progress.log'):
	log = pd.read_csv(log_path, comment='#', sep='\t')
	treelog=log.head(head_size)
	treelog = treelog.drop_duplicates(['prev_id', 'id'], keep="first")
	parent_child_list = treelog[['prev_id', 'id']].apply(tuple, axis=1).tolist()
	tree = Tree.from_parent_child_table(parent_child_list)
	print(tree.write(format=1))
	return tree 

t = extract_tree(1000)

print(t)


names = [l.name for l in t]
dists  = [t.get_distance(l, 'init') for l in t]



for leaf in t:
	if t.get_distance(leaf, 'init') < 4.1: 
		print(leaf.name, t.get_distance(leaf.name, 'init'))
		t.search_nodes(name=leaf.name)[0].delete() 


print(t)







	treelog.prev_id = treelog.gndx +"_"+ treelog.prev_id
	treelog.id = treelog.gndx +"_"+ treelog.id
    """