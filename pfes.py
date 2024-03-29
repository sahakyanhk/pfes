import argparse
import os, sys, shutil
import pandas as pd
import numpy as np
import typing as T

from evolver import sequence_mutator, selector, randomseq
from score import get_nconts, get_inter_nconts
from psique import pypsique


import torch
import esm

def backup_output(outpath):
    print(f'\nSaving output files to {args.outpath}')
    if os.path.isdir(outpath): 
        backup_list = []
        last_backup = int()
        for dir_name in os.listdir():
            if dir_name.startswith(outpath + '.'):
                backup=(dir_name.split('.')[-1])
                if backup.isdigit(): 
                    backup_list.append(backup)
                    last_backup = int(max(backup_list))
        print(f'\n{outpath} already exists, renameing it to {outpath}.{str(last_backup +  1)}') 
        os.replace(outpath, outpath + '.' + str(last_backup +  1))


def esm_to_data(esm_out, model):
    output = {key: value.cpu() for key, value in esm_out.items()}
    pdb_txt = model.output_to_pdb(esm_out)
    
    ptm = esm_out["ptm"][0].tolist()
    mean_plddt = esm_out["mean_plddt"][0].tolist()
    return(round(ptm, 3), round(mean_plddt*0.01, 3), pdb_txt)

def create_batched_sequence_datasest(sequences: T.List[T.Tuple[str, str]], max_tokens_per_batch: int = 1024
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:

    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)

    yield batch_headers, batch_sequences




#needed for score
pL0 = 120 # protein lenght penalty (0.5 at 120)
hL0 = 30 # helix lenght penalty (0.5 at 20)
def sigmoid(x,L0=0,c=0.1):
    return 1 / (1+2.71828182**(c * (L0-x)))

#================================single_fold_evolver================================# 
def single_fold_evolver(args): 
    print('not ready yet')
#def dimer_evolver(model, args):  
#    print("evolution of interacting dimers")
#================================single_fold_evolver================================# 


#evolution of an interacting chain
PDB_6WXQ=":MKSYFVTMGFNETFLLRLLNETSAQKEDSLVIVVPSPIVSGTRAAIESLRAQISRLNYPPPRIYEIEITDFNLALSKILDIILTLPEPIISDLTMGMRMINLILLGIIVSRKRFTVYVRDE" # 6WXQ (12 to 134) 
NZ_CP011286=":LNIIKLFHGHKYCLIFYVLP" #intergenic region from Yersinia
PDB_1RFA=":ASNTIRVFLPNKQRTVVNVRNGMSLHDCLMKALKVRGLQPECCAVFRLLHEHKGKKARLDWNTDAASLIGEELQVDFLD" #1RFA (55 to 132)
PDB_1RFP=":QCRRLCYKQRCVTYCRGR" # 1RFP contains S-S bond
PDB_4REX=":DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLSQ" #4REX (170 to 207)
PDB_6SVE=":WEKRMSRNSGRVYYFNHITNASQF" #WW domain
PDB_4QR0=":MMVLVTYDVNTETPAGRKRLRHVAKLCVDYGQRVQNSVFECSVTPAEFVDIKHRLTQIIDEKTDSIRFYLLGKNWQRRVETLGRSDSYDPDKGVLLL" #Cas2 from Streptococcus pyogenes serotype M1 (301447)



def inter_evolver(args):  

    #load models
    print('\nloading esm.pretrained.esmfold_v1... \n')
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()

    os.makedirs(pdb_path, exist_ok=True)
    with open(os.path.join(args.outpath, args.log), 'w') as f:
        f.write(' '.join(sys.argv[1:]) + '\n')

    seq2 =  PDB_4QR0 

    #creare an initial pool of sequences with pop_size
    columns=['genndx',
             'id', 
             'seq_len', 
             'prot_len_penalty', 
             'max_helix_penalty',
             'ptm', 
             'mean_plddt', 
             'num_conts', 
             'num_inter_conts', 
             'score', 
             'sequence', 
             'dssp']
    

    init_gen = pd.DataFrame({'sequence': [sequence_mutator(args.initial_seq) for i in range(args.pop_size)]})
    ancestral_memory = pd.DataFrame(columns=columns)
    ancestral_memory.to_csv(os.path.join(args.outpath, args.log), mode='a', index=False, header=True, sep='\t') #write header of the progress log
    
    #mutate seqs from init_gen and select the best n seqs for the next generation    
    for gen_i in range(args.num_generations):
        n = 0
        new_gen = pd.DataFrame(columns=columns)
        sequences = []
        for sequence in init_gen.sequence:

            id = "gen{0}_seq{1}".format(gen_i, n); n+=1 # give an uniq id even if the same sequence already exists            
            seq = sequence_mutator(sequence)
            
            #chek if the mutated seqeuece was already predicted
            seqmask = ancestral_memory.sequence == seq 
            if args.norepeat and seqmask.any(): #if seq is in the ancestral_memory mutate it again 
                while seqmask.any():
                    seq = sequence_mutator(seq)
                    seqmask = ancestral_memory.sequence == seq 

        
            sequences.append((id, seq+seq2))

            if seqmask.any():
                repit = ancestral_memory[seqmask].drop_duplicates(subset=['sequence'])
                repit.id = id #assing a new id to the already exiting sequence
                new_gen = new_gen.append(repit)

            batched_sequences = create_batched_sequence_datasest(sequences, args.max_tokens_per_batch)

            #predict data for the new sequence
            for headers, sequences in batched_sequences:
                try:
                    with torch.no_grad(): 
                        output = model.infer(sequences) #, num_recycles=args.num_recycles)
                except RuntimeError as e:
                    if e.args[0].startswith("CUDA out of memory"):
                        if len(sequences) > 1:
                            print(
                                f"Failed (CUDA out of memory) to predict batch of size {len(sequences)}. "
                                "Try lowering `--max-tokens-per-batch`."
                            )
                        else:
                            print(
                                f"Failed (CUDA out of memory) on sequence {headers[0]} of length {len(sequences[0])}."
                            )

                        continue
                    raise 


############################################################################################################
            ptm, _, pdb_txt = esm_to_data(model.infer(seq + seq2), model)
            seq_len = len(seq)
            with open(pdb_path + id + '.pdb', 'w') as f: # TODO conver this into a function
                f.write(pdb_txt)   

            #================================SCORING================================# 
            num_conts, mean_plddt = get_nconts(pdb_txt, 'A', 6, 70)
            num_inter_conts, _ = get_inter_nconts(pdb_txt, 'A', 'B', 6, 70) #TODO dinamicaly change the cutoff plddt
            dssp, max_helix = pypsique(pdb_path + id + '.pdb', 'A')

            #Rg, aspher = get_aspher(pdb_txt)
            prot_len_penalty =  1 - sigmoid(seq_len, pL0) * np.tanh(seq_len*0.05)
            max_helix_penalty = 1 - sigmoid(max_helix, hL0, 0.5)

            score  = np.prod([mean_plddt,           #[0, 1]
            ptm,                  #[0, 1]
            prot_len_penalty,     #[0, 1]
            max_helix_penalty,    #[0, 1]
            num_conts,            #[1, inf]
            num_inter_conts])     #[1, inf]
            #================================SCORING================================#
                
            new_gen = new_gen.append({'genndx': gen_i,
                                    'id': id, 
                                    'seq_len': seq_len,
                                    'prot_len_penalty': round(prot_len_penalty, 3), 
                                    'max_helix_penalty': round(max_helix_penalty, 3),
                                    'ptm': ptm, 
                                    'mean_plddt': mean_plddt, 
                                    'num_conts': num_conts, 
                                    'num_inter_conts': num_inter_conts, 
                                    'score': round(score, 3), 
                                    'sequence': seq, 
                                    'dssp': dssp
                                    }, ignore_index=True)
               
                #write a log file NOW same as new gen 
                #log = (f'{id}\t{seq_len}\t{round(prot_len_penalty,2)}\t{round(max_helix_penalty,2)}\t{ptm}\t{mean_plddt}\t{num_conts}\t{num_inter_conts}\t{round(score,2)}\t{seq}\t{dssp}')
                #print(f'{log}')
                
            print(new_gen.drop('genndx', axis=1).tail(1).to_string(index=False, header=False).replace(' ', '\t'))
        ancestral_memory =  ancestral_memory.append(init_gen)
        
        #select the next generation 
        init_gen = selector(new_gen, init_gen, args.pop_size, args.selection_mode, args.norepeat)
        init_gen.genndx = f'genndx{gen_i}' #assign a new gen index
        init_gen.to_csv(os.path.join(args.outpath, args.log), mode='a', index=False, header=False, sep='\t')
        print(batch)
        #with open(os.path.join(args.outpath, args.log), 'a') as f:
        #    f.write(f'{init_gen}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Sample sequences based on a given structure.'
    )
    parser.add_argument(
            '-em', '--evolution_mode', type=str,
            help='evolution mode',
            default='inter_chain',
    )
    parser.add_argument(
            '-sm', '--selection_mode', type=str,
            help='selection mode\n options: strong, weak ',
            default="weak"
    )
    parser.add_argument(
            '-seq', '--initial_seq', type=str,
            help='a sequence to initiate with',
            default='random'
    )    
    parser.add_argument(
            '-o' ,'--outpath', type=str,
            help='output filepath for saving sampled sequences',
            default='output',
    )
    parser.add_argument(
            '-ng', '--num_generations', type=int,
            help='number of generations',
            default=100,
    )
    parser.add_argument(
            '-ps', '--pop_size', type=int,
            help='population size',
            default=10
    )
    parser.add_argument(
            '-l', '--log', type=str,
            help='log output',
            default='progress.log',
    )
    parser.add_argument(                        #!
            '-nrep', '--norepeat', action='store_true',        #!
            help='do not allow to generate the same sequences',     #1
    )
    parser.add_argument(
            '-nbk', '--nobackup', action='store_true', 
            help='owerride files if exists',
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=None,
        help="Number of recycles to run. Defaults to number used in training (4).",
    )
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=1024,
        help="Maximum number of tokens per gpu forward-pass. This will group shorter sequences together "
        "for batched prediction. Lowering this can help with out of memory issues, if these occur on "
        "short sequences.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunks axial attention computation to reduce memory usage from O(L^2) to O(L). "
        "Equivalent to running a for loop over chunks of of each dimension. Lower values will "
        "result in lower memory usage at the cost of speed. Recommended values: 128, 64, 32. "
        "Default: None.",
    )

    args = parser.parse_args()
    
    print(' '.join(sys.argv[1:]))

    #backup if output directory exists
    if args.nobackup:
        if os.path.isdir(args.outpath):
            print(f'\nWARNING! Directory {args.outpath} exists, it will be replaced!' )
            shutil.rmtree(args.outpath)
        os.makedirs(args.outpath)
    else:
        backup_output(args.outpath)
    
    
    # TODO check arguments and input paths before loading models 


    pdb_path = args.outpath + '/pdb/' #set paths 
    basename = "pfes" #(os.path.basename(args.pdbfile).split('.')[0])

    if args.evolution_mode == "inter_chain":
        inter_evolver(args)
    else: 
        single_fold_evolver(args)

