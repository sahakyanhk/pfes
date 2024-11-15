import argparse
import os, re
import pandas as pd
import numpy as np
import shutil
import gzip
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors


import warnings


parser = argparse.ArgumentParser(description="Analyse PFES")
parser.add_argument('-l', '--log', type=str, help='log file name', default='progress.log') 
parser.add_argument('-s', '--pdbdir', type=str, help='directory with pdb files', default='structures')
parser.add_argument('-t', '--traj', type=str, help='make backbone trajectory', default='pfestraj.pdb')
parser.add_argument('-o', '--outdir', type=str, help='output directory name', default='visual_pfes_results')
parser.add_argument('-b', '--start', type=int, help='first point to read from trajectory', default=0)
parser.add_argument('-e', '--end', type=int, help='last point to read from trajectory', default=99999999)
parser.add_argument('--notraj', action='store_false', )
parser.add_argument('--noplots', action='store_false', )


args = parser.parse_args()


#class VisualPFES():


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def extract_lineage(log):
    traj_len = len(log)
    #pop_size = len(log[log.gndx == 'gndx0'])

    print(f'Processing a trajectory with {traj_len} mutations')
    lineage = log.drop_duplicates('gndx').tail(1)
    df = lineage
    ndx = df.id.to_string(index=False)
        
    def return_ancestor(log, node):
        parent = log[log.id == node]
        parent = parent.drop_duplicates('sequence')
        return parent

    
    pbar = tqdm(desc='Extracting lineage')
    i=0
    while not df.empty:
        ndx = return_ancestor(log, ndx)
        df = ndx
        lineage = pd.concat([lineage, df], axis=0)
        ndx = ndx.prev_id.to_string(index=False)
        i=+1
        pbar.update(i)
    pbar.close()
    lineage = lineage.sort_index()
    ltail = lineage.tail(1)
    print(f"""
{ltail[["gndx","seq_len","ptm","mean_plddt","num_conts","iplddt", "num_inter_conts","score"]].iloc[-1]}
{ltail.sequence.iloc[-1]}
{ltail.ss.iloc[-1]}
""")
    return lineage

def extract_sequences(log):
    fasta  = "fasta"
    return fasta

#======================= make separate plots =======================#
def make_plots(log, bestlog, lineage):

    ms=0.1
    lw=1.4
    dpi=500

    os.makedirs(plotdir, exist_ok=True)
    for colname in log.keys(): 
        if not colname in ['seq', 'sequence', 'ss', 'genindex','dssp', 'mutation', 'index', 'id', 'prev_id', 'gndx', 'sel_mode']:
                fig, ax1 = plt.subplots(figsize=(9, 3))
                ax1.plot(log[colname],'.', markersize=ms,    color='silver', label='all mutations')
                ax1.plot(bestlog[colname],'-', linewidth=lw, label='best of the generation')
                ax1.plot(lineage[colname],'-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={len(lineage[colname])})')
                ax1.legend(loc ="lower right")
                ax1.grid(True, which="both",linestyle='--', linewidth=0.3)
                ax1.set(xlabel="Total number of mutations", ylabel=colname.capitalize())
                #ax2 = ax1.twiny()
                #ax2.plot(lineage[colname].tolist(),'-', linewidth=lw, color='mediumslateblue')
                #ax2.set(xlabel="Lineage length")
                fig.tight_layout()
                fig.savefig(plotdir + colname + '.png', dpi=dpi)
                fig.clf()

#======================= Summary plot =======================#
def make_summary_plot(log, bestlog, lineage):
    
    ms=0.1
    lw=1.0
    dpi=500

    fig, axs = plt.subplots(3,2, figsize=(10, 8))

    fig.suptitle(None)

    L = len(lineage)
    axs[0,0].plot(log.mean_plddt, '.', markersize=ms,    color='silver', label='all mutations')
    axs[0,0].plot(bestlog.mean_plddt, '-', linewidth=lw, label='best of the generation')
    axs[0,0].plot(lineage.mean_plddt, '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={L})')
    axs[0,0].set(xlabel=None, ylabel='mean pLDDT')
    axs[0,0].grid(True, which="both",linestyle='--', linewidth=0.5)
    axs[0,0].set_xticklabels([])

    axs[1,0].plot(log.ptm, '.', markersize=ms,    color='silver', label='all mutations')
    axs[1,0].plot(bestlog.ptm, '-', linewidth=lw, label='best of the generation')
    axs[1,0].plot(lineage.ptm, '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={L})')
    axs[1,0].set(xlabel=None, ylabel='pTM')
    axs[1,0].grid(True, which="both",linestyle='--', linewidth=0.5)
    axs[1,0].set_xticklabels([])

    axs[2,0].plot(log.score,  '.', markersize=ms,    color='silver', label='all mutations')
    axs[2,0].plot(bestlog.score, '-', linewidth=lw,  label='best of the generation')
    axs[2,0].plot(lineage.score,  '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={L})')
    axs[2,0].set(xlabel='Total number of mutations', ylabel='Score')
    axs[2,0].grid(True, which="both",linestyle='--', linewidth=0.5)
    axs[2,0].legend(loc ="lower right")

    axs[0,1].plot(log.seq_len, '.', markersize=ms,    color='silver', label='all mutations')
    axs[0,1].plot(bestlog.seq_len, '-', linewidth=lw, label='best of the generation')
    axs[0,1].plot(lineage.seq_len, '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={L})')
    axs[0,1].set(xlabel=None, ylabel='Sequence length')
    axs[0,1].grid(True, which="both",linestyle='--', linewidth=0.5)
    axs[0,1].set_xticklabels([])

    if 'num_inter_conts' in bestlog.columns and bestlog.num_inter_conts.max() != 1:
        axs[1,1].plot(log.iplddt, '.', markersize=ms,    color='silver', label='all mutations')
        axs[1,1].plot(bestlog.iplddt, '-', linewidth=lw, label='best of the generation')
        axs[1,1].plot(lineage.iplddt, '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={L})')
        axs[1,1].set(xlabel=None, ylabel='iPLDDT')
        axs[1,1].grid(True, which="both",linestyle='--', linewidth=0.5)
        axs[1,1].set_xticklabels([])

    else:     
        axs[1,1].plot(log.max_beta_penalty, '.', markersize=ms,    color='silver', label='all mutations')
        axs[1,1].plot(bestlog.max_beta_penalty, '-', linewidth=lw, label='best of the generation')
        axs[1,1].plot(lineage.max_beta_penalty, '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={L})')
        axs[1,1].set(xlabel=None, ylabel='SS penalty')
        axs[1,1].grid(True, which="both",linestyle='--', linewidth=0.5)
        axs[1,1].set_xticklabels([])

    axs[2,1].plot(log.num_conts, '.', markersize=ms,  color='silver', label='all mutations')
    axs[2,1].plot(bestlog.num_conts, '-', linewidth=lw, label='best of the generation')
    axs[2,1].plot(lineage.num_conts, '-', linewidth=lw, color='mediumslateblue', label=f'lineage (L={L})')
    axs[2,1].set(xlabel='Total number of mutations', ylabel='Number of contacts')
    axs[2,1].grid(True, which="both",linestyle='--', linewidth=0.5)

    #plt.xticks(rotation=45)

    #for ax in axs.flat:
    #   ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    #for ax in axs.flat:
    #   ax.label_outer()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir,'Summary.png'), dpi=dpi)


#======================= seconday structure plot =======================#
def make_ss_plot(lineage):

    dpi=500

    max_seq_len = int(max(lineage.seq_len))
    lineage_len = len(lineage)

    sse = np.empty((lineage_len, max_seq_len), dtype='U1')
    i=0
    for ss in lineage.ss:
        sse[i] = list(ss + "X"*(max_seq_len-len(ss)))
        i+=1

    def sse_to_num(sse):
        num = np.empty(sse.shape, dtype=int)
        num[sse == 'F'] = 0
        num[sse == 'f'] = 0
        num[sse == 'g'] = 0
        num[sse == 's'] = 0
        num[sse == 'P'] = 0 
        num[sse == 'C'] = 0 
        num[sse == 'E'] = 1 
        num[sse == 'B'] = 2 
        num[sse == 'S'] = 3 
        num[sse == 'T'] = 4 
        num[sse == 'H'] = 5 
        num[sse == 'G'] = 6 
        num[sse == 'I'] = 7 
        num[sse == 'X'] = 8 
        return num


    sse_digit = sse_to_num(sse)

    color_assign = {
        r"coil": "darkgrey",
        r"$\beta$-sheet": "yellow",
        r"$\beta$-bridge": "y",
        r"bend": "orange",
        r"turn": "brown",
        r"$\alpha$-helix": "purple",
        r"$3_{10}$-helix": "blue",
        r"$\pi$-helix": "mediumpurple",
        r"": "white"
        }

    cmap = colors.ListedColormap(color_assign.values())
    if len(lineage) > 2000:
        ticks = np.arange(0, len(lineage)+1, 1000)
    else: 
        ticks = np.arange(0, len(lineage)+1, 100)

    plt.figure(figsize=(9, 3), dpi=dpi)
    plt.imshow(sse_digit.T, origin='lower', cmap=cmap,  interpolation='nearest', aspect='auto')
    plt.xticks(ticks, ticks.astype(int))
    plt.xlabel("Lineage length")
    plt.ylabel("Residues")

    custom_lines = [
        Line2D([0], [0], color=cmap(i), lw=4) for i in range(len(color_assign)-1)]

    plt.legend(
        custom_lines, color_assign.keys(), loc="upper center",
        bbox_to_anchor=(0.5, 1.15), ncol=len(color_assign), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,'Secondary_structures.png'), dpi=dpi) 

def backbone_traj(trajlog, pdbdir):
    """
    make trajectory from C-alpha atoms
    TO DO: 
    1. find the biggest pdb in folder, or from progress log. \
    Make it initial topology that should be uploaded first or append in in the begining of the trajectory file
    2. save trajectory in xyz format to save space. 
    """
    trajpdb = os.path.join(outdir, 'trajpdb/')
    trajlog = trajlog.drop_duplicates(subset = 'sequence')
    pfeslen = len(trajlog)

    os.makedirs(trajpdb, exist_ok=True)
    if os.path.isdir(trajpdb) and len(os.listdir(trajpdb)) != pfeslen:
        shutil.rmtree(trajpdb)
        os.makedirs(trajpdb, exist_ok=True)
        print(f'{pfeslen} unique sequences with the best folds are selected') 
        for gndx, pdbid in tqdm(zip(trajlog.gndx, trajlog.id), total=len(trajlog)):
            try:
                shutil.copy(pdbdir +'/' + pdbid +'.pdb.gz', trajpdb +'/'+ gndx + '.pdb.gz')
            except FileNotFoundError:
                print(pdbid +'.pdb.gz is missing' )
                pass
    else: 
        print(f'The best folds from {pfeslen} generations are selected')

    #if single_chian:
    print("extracting backbone coordinates...")
    i=0
    PDB_A, PDB_B, lastBB_A, lastBB_B = [], [], [], []
    for pdb in tqdm(sorted_alphanumeric(os.listdir(trajpdb))):
        with gzip.open(os.path.join(trajpdb, pdb), 'rb') as file:
            pdb_txt = file.read().decode()
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

        lastresidueA=''.join([str(elem) for elem in bb_chain_A[-4:]]) # keep the last four lines to repeat them and make the number of atoms in all models equal
        lastresidueB=''.join([str(elem) for elem in bb_chain_B[-4:]]) # keep the last four lines to repeat them and make number of atoms in all models equal

        PDB_A.append(bb_chain_A) #save chain A
        PDB_B.append(bb_chain_B) #save chain B
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
    try: 
        import MDAnalysis as mda
        from MDAnalysis.analysis import align

        print('writing aligned backbone trajectory...')
        traj = mda.Universe(outdir+'/.tmp.pdb')
        top = traj.select_atoms('protein')

        warnings.filterwarnings("ignore")
        if 'num_inter_conts' in trajlog.columns and bestlog.num_inter_conts.max() != 1:
            chianID = 'chainID B'
        else:
            chianID = 'chainID A'

        align.AlignTraj(traj,  # trajectory to align
                        top,  # reference
                        select=chianID,  # selection of atoms to align
                        filename=trajpath,  # file to write the trajectory to
                        ).run()

        os.remove(outdir+'/.tmp.pdb')
    except:
        print('Warning: MDAnalysis package is not available, writing unaligned trajectory')
        shutil.copy(outdir+'/.tmp.pdb', trajpath)



outdir = args.outdir 
os.makedirs(outdir, exist_ok=True)

pdbdir = args.pdbdir
plotdir = os.path.join(outdir, 'plots/')
trajpath = os.path.join(outdir, args.traj)

log = pd.read_csv(args.log, sep='\t', comment='#')

log = log.iloc[args.start:args.end]

bestlog = log.groupby('gndx').head(1)
bestlog.to_csv(os.path.join(outdir, 'bestlog.tsv'), sep='\t', index=False, header=True)


print('==================================')
lineage = extract_lineage(log)
lineage.to_csv(os.path.join(outdir, 'lineage.tsv'), sep='\t', index=False, header=True)



if args.noplots:
    print('making plots')
    make_plots(log, bestlog, lineage)

    print('making summary plot')
    make_summary_plot(log, bestlog, lineage)

    print('making secondary structure plot')
    make_ss_plot(lineage)

if args.notraj:
    print('making backbone trajectory')
    backbone_traj(lineage, pdbdir)

print('=================================='"\n")
