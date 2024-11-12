import argparse
import os
import sys
import shutil
import pandas as pd
import numpy as np
import typing as T
import threading
import gzip
import time
from datetime import datetime

import torch
import esm

from evolution import Evolver
from score import get_nconts, cbiplddt
from psique import pypsique
from openfold.utils.loss import compute_tm


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


def create_batched_sequence_dataset(sequences: T.List[T.Tuple[str, str]], max_tokens_per_batch: int = 1524
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:
    batch_headers, batch_sequences, num_tokens, num_sequences= [], [], 0, 0 
    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens, num_sequences= [], [], 0, 0 
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)
        num_sequences += 1
        if num_sequences > args.pop_size / 2: #TODO test this with args.pop_size / 4 and lartge pop size
           yield batch_headers, batch_sequences
           batch_headers, batch_sequences, num_tokens, num_sequences= [], [], 0, 0
    yield batch_headers, batch_sequences

def pdbtxt2bbcoord(pdb_txt, chain='A'):
    # can extract this directly from esm output
    # positions contains coordinates, and aatype contains the sequence
    coords3 = np.array([line[30:54].split()  for line in pdb_txt.splitlines() if line[:4] == "ATOM" and 
                        line[20:22].strip() == chain and 
                        ((line[11:16].strip() == "N") | 
                         (line[11:16].strip()== "CA") | 
                         (line[11:16].strip() == "C"))], dtype='float32')
    coords33 = coords3.reshape(int(coords3.shape[0]/3),3,3)
    return(coords33)

def esm2data(esm_out):
    output = {key: value.cpu() for key, value in esm_out.items()} 
    pdbs = model.output_to_pdb(output) 
    mask = output["atom37_atom_exists"][:,:,1] == 1 
    chainA_mask = torch.logical_and(mask, output["chain_index"] == 0)
    sl = np.sum(chainA_mask.numpy(), 1) # chainA_len
    sl_len = len(sl)
    ptm = [compute_tm(output["ptm_logits"][i][None, :sl[i],:sl[i]]).item() for i in range(sl_len)] #ptm only for chain A
    ptm_full = esm_out["ptm"].tolist() # will clculate pTM for entire complex if more than one chain
    plddt =  [output["plddt"][:,:,1][i][chainA_mask[i]]/100 for i in range(sl_len)] 
    mean_plddt = [plddt[i].mean().item() for i in range(len(sl))]
    return(pdbs, ptm, mean_plddt) #return score instead

    #calculate the number of contacts
    # bins = np.append(0,np.linspace(2.3125,21.6875,63))
    # #you do not need softmax to keep the actual values 
    # sm_contacts = softmax(output["distogram_logits"],-1)
    # sm_contacts = sm_contacts[...,bins<8].sum(-1)
    # mask = output["atom37_atom_exists"][0,:,1] == 1
    # contact_map = sm_contacts[0][mask,:][:,mask]
    # num_conts = []
    """
    Return the number of contacts and individual plddts (write it in the log). 
    In the case of dimers, the number of interchain interactions with indexes is also returned. 
    Use indexes to calculate iPLDDT

    """

def sigmoid(x,L0=0,c=0.1):
    return 1 / (1+2.71828182**(c * (L0-x)))


#==============================================================================================#
#================================== EXTRACT AND SCORE =========================================#
#==============================================================================================#

def extract_results(gen_i, headers, sequences, pdbs, ptms, mean_plddts) -> None:
    global new_gen #this will be modified in the fold_evolver()

    for meta_id, seq, pdb_txt, ptm, mean_plddt, in zip(headers, sequences, pdbs, ptms, mean_plddts): #which plddt is better? this is plddt for both A and B chains in case of inter_chain
        
        all_seqs = seq.split(':')
        seq = all_seqs[0]
        seq_len = len(seq)
        
        id_data = meta_id.split('_')

        id = id_data[0]
        prev_id = id_data[1]
        mutation = id_data[2]

        with open(pdb_path + id + '.pdb', 'wb') as f: 
            f.write(pdb_txt.encode())   

        #=======================================================================# 
        #================================SCORING================================# 
        num_conts, _mean_plddt_ = get_nconts(pdb_txt, 'A', 6.0, 50) #plddt is better only for chain A and for residues > 50

        if args.evolution_mode == "single_chain": #if there are two or more chains, then calculate the number of interacting contacts
            num_inter_conts, iplddt = 1, 1
        else:
            num_inter_conts, iplddt = cbiplddt(pdb_txt, 'A', 'B', 6.0, 40) 

        ss, max_helix, max_beta = pypsique(pdb_txt, 'A')
        #Rg, aspher = get_aspher(pdb_txt)
        #dG = dGscore(pdbtxt2bbcoord(pdb_txt), seq) 
        prot_len_penalty =  1 - sigmoid(seq_len, args.prot_len_penalty, 0.2)
        max_alpha_penalty = 1 - sigmoid(max_helix, args.helix_len_penalty, 0.5)
        max_beta_penalty = 1 - sigmoid(max_beta, args.beta_len_penalty, 0.6)
        
        score  = np.prod([mean_plddt,           #[0, 1]
                          ptm,                  #[0, 1]
                          iplddt,               #[0, 1]
                          prot_len_penalty,     #[0, 1]
                          max_beta_penalty,     #[0, 1]
                          max_alpha_penalty,    #[0, 1]
                          #dG, #~[0, inf]
                          (num_conts + seq_len) / seq_len,
                          (num_inter_conts + seq_len) / (seq_len + 1) # change this to sigmod so the number of inter contacts > X would not increase the score 
                          ]) 
        #================================SCORING================================#
        #=======================================================================# 

        iterlog = pd.DataFrame({'gndx': gen_i,
                                'id': id, 
                                'seq_len': seq_len,
                                'prot_len_penalty': round(prot_len_penalty, 2), 
                                'max_alpha_penalty': round(max_alpha_penalty, 2),
                                'max_beta_penalty': round(max_beta_penalty, 2),
                                'ptm': round(ptm, 2), 
                                'mean_plddt': round(mean_plddt, 2), 
                                'num_conts': num_conts, 
                                'iplddt': iplddt,
                                'num_inter_conts': num_inter_conts, 
                                'sel_mode': args.selection_mode,
                                #'dG': round(dG, 3),
                                #'ptm_full': ptm_full,
                                #'cd' contact_density
                                'score': round(score, 3), 
                                'sequence': seq, 
                                'mutation': mutation,
                                'prev_id': prev_id,
                                'ss': ss}, index=[0])
        
        new_gen = pd.concat([new_gen, iterlog], axis=0, ignore_index=True) 
        os.system(f"gzip {pdb_path}{id}'.pdb' &")

    print(new_gen.tail(args.pop_size).drop('gndx', axis=1).to_string(index=False, header=False))


def multimer_evolver(model, args):  
    print("evolution of interacting dimers")

global new_gen #this will be modified in the extract_results() 

#============================================================================# 
#================================FOLD_EVOLVER================================# 
 

def fold_evolver(args, model, evolver, logheader, init_gen) -> None: 

    os.makedirs(pdb_path, exist_ok=True)
    with open(os.path.join(args.outpath, args.log), 'w') as f:
        f.write(logheader)

    condition = True
    
    #creare an initial pool of sequences with pop_size
    columns=['gndx',
             'id', 
             'seq_len', 
             'prot_len_penalty', 
             'max_alpha_penalty',
             'max_beta_penalty',
             'ptm', 
             'mean_plddt', 
             'num_conts', 
             'iplddt',
             'num_inter_conts',
             'sel_mode',
             #'dG',
             'score', 
             'sequence', 
             'mutation',
             'prev_id',
             'ss']
    

    ancestral_memory = pd.DataFrame(columns=columns)
    ancestral_memory.to_csv(os.path.join(args.outpath, args.log), mode='a', index=False, header=True, sep='\t') #write header of the progress log
    
    #mutate seqs from init_gen and select the best N seqs for the next generation    
    for gen_i in range(args.num_generations):
        n = 0
        global new_gen #this will be modified in the extract_results() 
        new_gen = pd.DataFrame(columns=columns)
        #now = datetime.now()
        generated_sequences = []
        mutation_collection = []

        for prev_id, sequence in zip(init_gen.id, init_gen.sequence):
            seq, mutation_data= evolver.mutate(sequence)
            
            #check if the mutated seqeuece was already predicted
            seqmask = ancestral_memory.sequence == seq 
            
            #if --norepeat and seq is in the ancestral_memory mutate it again
            if args.norepeat and seqmask.any():  
                while seqmask.any():
                    seq, mutation_data = evolver.mutate(seq)
                    seqmask = ancestral_memory.sequence == seq 

            id = "g{0}seq{1}_{2}_{3}".format(gen_i, n, prev_id, mutation_data); n+=1 # gives an unique id even if the same sequence already exists            

            if seqmask.any(): #if sequence already exits do not predict a structure again 
                repeat = ancestral_memory[seqmask].drop_duplicates(subset=['sequence'], keep='last') 
                #try:
                #    shutil.copyfile(pdb_path + repeat.id.values[0] + '.pdb', pdb_path + id.split('_')[0] + '.pdb')  
                #except  FileNotFoundError: 
                #    pass
                #repeat.id = id.split('_')[0] #assing a new id to the already exiting sequence
                new_gen = pd.concat([new_gen, repeat])
            else:
                generated_sequences.append((id, seq)) 
                mutation_collection.append(mutation_data)    
        
        batched_sequences = create_batched_sequence_dataset(generated_sequences, args.max_tokens_per_batch)

        #predict data for the new batch
        for headers, sequences in batched_sequences:
            pdbs, ptms, mean_plddts = [], [], []
            with torch.no_grad(): 
                pdbs, ptms, mean_plddts  = esm2data(model.infer(sequences, 
                                                               num_recycles = args.num_recycles,
                                                               residue_index_offset = 1,
                                                               chain_linker = "G" * 25))
            
            #run extract_results() in becground and imediately start next the round of model.infer()
            trd = threading.Thread(target=extract_results, args=(gen_i, headers, sequences, pdbs, ptms, mean_plddts))
            trd.start()

        while trd.is_alive(): 
            time.sleep(0.2)
        
        #print(f"#GENtime {datetime.now() - now}")
        ancestral_memory =  pd.concat([ancestral_memory, init_gen])

        #select the next generation 
        init_gen = evolver.select(new_gen, init_gen, args.pop_size, args.selection_mode, args.norepeat, args.beta)
        init_gen.gndx = f'gndx{gen_i}' #assign a new gen index
        init_gen.to_csv(os.path.join(args.outpath, args.log), mode='a', index=False, header=False, sep='\t')

        #TODO write init_gen as a checkpoit file to continue the simulation

        #Change the selection with a condition (plddt, ptm)
        if args.strong_selection_by_condition:
            if (init_gen['mean_plddt'] > 0.6) & (init_gen['ptm'] > 0.5).any() & condition:
                args.selection_mode = 'strong'
                condition = False #do not change args.selection_mode anymore
                with open(os.path.join(args.outpath, args.log), mode='a') as f:
                    f.write("#changing the selection mode to strong")

        #Change the selection mode after n generations
        if args.strong_selection_after_n_gen > 0:
            if (gen_i > args.strong_selection_after_n_gen) & condition:
                args.selection_mode = 'strong'
                evolver = Evolver('flatoptim')
                condition = False #do not change args.selection_mode anymore
                print("#changing the selection mode to strong")
                with open(os.path.join(args.outpath, args.log), mode='a') as f:
                    f.write("#changing the selection mode to strong")

        #stop simulation by a condition
        if args.stop_by_condition:
            if (init_gen['mean_plddt'] > 0.85).any() & (init_gen['ptm'] > 0.75).any():
                print(f'gndx={gen_i}; the condition reached, breaking!')
                break

 
#================================FOLD_EVOLVER================================# 
#============================================================================# 





#==================================================================================#
#================================INTER_FOLD_EVOLVER================================# 

def inter_fold_evolver(args, model, evolver, logheader, init_gen) -> None: 

    #evolution of an interacting chain
    NZ_CP011286=":LNIIKLFHGHKYCLIFYVLP" #intergenic region from Yersinia
    PDB_1RFP=":QCRRLCYKQRCVTYCRGR" # 1RFP contains S-S bond
    PDB_6SVE=":WEKRMSRNSGRVYYFNHITNASQF" #WW domain
    PDB_5YIW=":GAMDMSWTDERVSTLKKLWLDGLSASQIAKQLGGVTRNAVIGKVHRLGL" #HTH
    PDB_4REX=":DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLSQ" #4REX (170 to 207) 
    PDB_6M6W=":MNDIIINKIATIKRCIKRIQQVYGDGSQFKQDFTLQDSVILNLQRCCEACIDIANHINRQQQLGIPQSSRDSFTLLAQNNLITQPLSDNLKKMVGLRNIAVHDAQELNLDIVVHVVQHHLEDFEQFIDVIKAE" #HEPN toxin
    PDB_4OO8=":GQKNSRERMKRIEEGIKELGSQILKEHPVENTQLQNEKLYLYYLQNGRDMYVDQELDINRLSDYDVDHIVPQSFLKDDSIDNKVLTRSDKNRGKSDNVPSEEVVKKMKNYWRQLLNAKLITQRKFDNLTKAERGGL" #CAS9 HNH
    PDB_5VGB=":GAASEIEKRQEENRKDREKAAAKFREYFPNFVGEPKSKDILKLRLYEQQHGKCLYSGKEINLGRLNEKGYVEIDHALPFSRTWDDSFNNKVLVLGSENQNKGNQTPYEYFNGKDNSREWQEFKARVETSRFPRSKKQRILLQ" #CAS9 HNH
    PDB_5O56=":SKNSRERMKRIEEGIKELGSQILKEHPVENTQLQNEKLYLYYLQNGRDMYVDQELDINRLSDYDVDHIVPQSFLKDDSIDNKVLTRSDKNRGKSDNVPSEEVVKKMKNYWRQLLNAKLITQRKFDNLTKAERG"
    seq2 = ':' + args.initial_seq2

    os.makedirs(pdb_path, exist_ok=True)
    with open(os.path.join(args.outpath, args.log), 'w') as f:
        f.write(logheader)


    #creare an initial pool of sequences with pop_size
    columns = ['gndx',
               'id', 
               'seq_len', 
               'prot_len_penalty',
               'max_alpha_penalty',
               'max_beta_penalty',
               'ptm', 
               'mean_plddt', 
               'num_conts', 
               'iplddt',
               'num_inter_conts',
               'sel_mode',
               #'dG',
               'score', 
               'sequence', 
               'mutation',
               'prev_id',
               'ss'] 
      
    ancestral_memory = pd.DataFrame(columns=columns)
    ancestral_memory.to_csv(os.path.join(args.outpath, args.log), mode='a', index=False, header=True, sep='\t') #write header of the progress log
    
    #mutate seqs from init_gen and select the best n seqs for the next generation    
    for gen_i in range(args.num_generations):
        n = 0
        global new_gen #this will be modified in the extract_results() 
        new_gen = pd.DataFrame(columns=columns)
        #now = datetime.now()
        generated_sequences = []
        mutation_collection = []

        for prev_id, sequence in zip(init_gen.id, init_gen.sequence):
            seq, mutation_data= evolver.mutate(sequence)
            
            #chek if the mutated sequence was already predicted
            seqmask = ancestral_memory.sequence == seq 
            
            #if --norepeat and seq is in the ancestral_memory mutate it again
            if args.norepeat and seqmask.any():  
                while seqmask.any():
                    seq, mutation_data = evolver.mutate(seq)
                    seqmask = ancestral_memory.sequence == seq 

            id = "g{0}seq{1}_{2}_{3}".format(gen_i, n, prev_id, mutation_data); n+=1 # give an uniq id even if the same sequence already exists            

            if seqmask.any(): #if sequence already exits do not predict a structure again 
                repeat = ancestral_memory[seqmask].drop_duplicates(subset=['sequence'], keep='last') 
                #try:
                #    shutil.copyfile(pdb_path + repeat.id.values[0] + '.pdb', pdb_path + id.split('_')[0] + '.pdb')  
                #except  FileNotFoundError: 
                #    pass
                #repeat.id = id.split('_')[0] #assing a new id to the already exiting sequence
                new_gen = pd.concat([new_gen, repeat])
            else:
                generated_sequences.append((id, seq + seq2)) #(seq+seq2)) add a function to select the sma
                mutation_collection.append(mutation_data)    


        batched_sequences = create_batched_sequence_dataset(generated_sequences, args.max_tokens_per_batch)

        #predict data for the new batch
        for headers, sequences in batched_sequences:
            pdbs, ptms, mean_plddts = [], [], [] #TODO calculate pTM only of chain A
            with torch.no_grad(): 
                pdbs, ptms, mean_plddts = esm2data(model.infer(sequences, 
                                                               num_recycles = args.num_recycles,
                                                               residue_index_offset = 1,
                                                               chain_linker = "GP" + "G"*30 + "PG"))
            
            #run extract_results() in background and immediately start next round of model.infer()
            trd = threading.Thread(target=extract_results, args=(gen_i, headers, sequences, pdbs, ptms, mean_plddts))
            trd.start()

        while trd.is_alive(): 
            time.sleep(0.2)
        
        #print(f"#GENtime {datetime.now() - now}")
        ancestral_memory =  pd.concat([ancestral_memory, init_gen])

        #select the next generation 
        init_gen = evolver.select(new_gen, init_gen, args.pop_size, args.selection_mode, args.norepeat)
        init_gen.gndx = f'gndx{gen_i}' #assign a new gen index
        init_gen.to_csv(os.path.join(args.outpath, args.log), mode='a', index=False, header=False, sep='\t')

#================================INTER_FOLD_EVOLVER================================# 
#==================================================================================#


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Sample sequences based on a given structure.'
    )
    parser.add_argument(
            '-em', '--evolution_mode', type=str,
            help='evolution mode: single_chain, inter_chain, multimer',
            default='single_chain, ',
    )
    parser.add_argument(
            '-sm', '--selection_mode', type=str,
            help='selection mode\n options: strong, weak ',
            default="weak"
    )
    parser.add_argument(
            '-b', '--beta', type=float,
            help='selection strength',
            default=1,
    )
    parser.add_argument(
            '-iseq', '--initial_seq', type=str,
            help='a sequence to initiate with, if "random" pop_size random sequences will be generated, the length of the random sequences can be assigned with "--random_seq_len"',
            default='random'
    )
    parser.add_argument(
            '-iseq2', '--initial_seq2', type=str,
            help='second sequence'
    )
    parser.add_argument(
            '-l', '--log', type=str,
            help='log output',
            default='progress.log',
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
            default=10,
    )
    parser.add_argument(
            '-ed', '--evoldict', type=str,
            help='population size',
            default='flatrates',
    )
    parser.add_argument(
            '-pl0', '--prot_len_penalty', type=int,
            help='population size',
            default=250,
    )
    parser.add_argument(
            '-hl0', '--helix_len_penalty', type=int,
            help='population size',
            default=20,
    )
    parser.add_argument(
            '-bl0', '--beta_len_penalty', type=int,
            help='population size',
            default=12,
    )
    parser.add_argument(
            '--random_seq_len', type=int,
            help='a sequence to initiate with',
            default=24,
    )
    parser.add_argument(                      
            '--norepeat', action='store_true', 
            help='do not generate and/or select the same sequences more than once', 
    )
    parser.add_argument(
            '--nobackup', action='store_true', 
            help='overwrite files if exists',
    )
    parser.add_argument(
            '--stop_by_condition', action='store_true', 
            help='experimental',
    )
    parser.add_argument(
            '--strong_selection_by_condition', action='store_true', 
            help='experimental',
    )
    parser.add_argument(
            '--strong_selection_after_n_gen', type=int,
            help='',
            default=1000,
    )
    # parser.add_argument(
    #         '--continue', action='store_true', 
    #         help='',
    # )
    parser.add_argument(
            '--num-recycles',
            type=int,
            default=1,
            help="Number of recycles to run. Defaults to number used in training (4).",
    )
    parser.add_argument(
            '--max-tokens-per-batch',
            type=int,
            default=2048, # 5120 works fine with A100
            help="Maximum number of tokens per gpu forward-pass. This will group shorter sequences together "
            "for batched prediction. Lowering this can help with out of memory issues, if these occur on "
            "short sequences."
    )

    args = parser.parse_args()
    evolver = Evolver(args.evoldict)

    now = datetime.now() # current date and time
    date_now = now.strftime("%d-%b-%Y")
    time_now = now.strftime("%H:%M:%S")
    


    logheader = f'''#======================== PFESv0.1 ========================#
#====================== {date_now} =======================#
#======================== {time_now} ========================#
#WD: {os.getcwd()}
#$pfes.py {' '.join(sys.argv[1:])}
#
#====================  pfes input params ==================#
#
#--evolution_mode, -em \t\t = {args.evolution_mode}
#--selection_mode, -sm\t\t = {args.selection_mode}
#--initial_seq, -iseq\t\t = {args.initial_seq}
#--pop_size, -ps\t\t = {args.pop_size}
#--evoldict, -ed\t\t = {args.evoldict}
#--log, -l\t\t\t = {args.log}
#--outpath, -o\t\t\t = {args.outpath}
#--random_seq_len\t\t = {args.random_seq_len}
#--beta, -b\t\t\t = {args.beta}
#--helix_len_penalty, -hl0\t = {args.helix_len_penalty}
#--prot_len_penalty, -pl0\t = {args.prot_len_penalty}
#--num_generations, -ng\t\t = {args.num_generations}
#--strong_selection_after_n_gen\t\t = {args.strong_selection_after_n_gen}
#--norepeat\t\t\t = {args.norepeat}
#--nobackup\t\t\t = {args.nobackup}
#--num-recycles\t\t\t = {args.num_recycles}
#--max-tokens-per-batch\t\t = {args.max_tokens_per_batch}
# evolution dictionary = {evolver.evoldict}
# evolution dictionary normalized = {evolver.evoldict_normal}
#==========================================================#
'''
    
    print(logheader)

    #backup if output directory exists
    if args.nobackup:
        if os.path.isdir(args.outpath):
            print(f'\nWARNING! Directory {args.outpath} exists, it will be replaced!')
            shutil.rmtree(args.outpath)
        os.makedirs(args.outpath)
    else:
        backup_output(args.outpath)

    pdb_path = args.outpath + '/structures/' 

    #create the initial generation
    if args.initial_seq == 'random':
        randomsequence = evolver.randomseq(args.random_seq_len)
        init_gen = pd.DataFrame({'id': ['init_seq'] * args.pop_size, 
                                 'sequence': [randomsequence] * args.pop_size,
                                 'score': [0.001] * args.pop_size})
    elif args.initial_seq == 'randoms':
        init_gen = pd.DataFrame({'id': [f'init_seq{i}' for i in range(args.pop_size)], 
                                 'sequence': [evolver.randomseq(args.random_seq_len) for i in range(args.pop_size)]})
    #elif args.initial_seq == 'c':
    #    init_gen = pd.read_csv('test.chk', sep='\t')
    else: 
        init_gen = pd.DataFrame({'id': ['init_seq'] * args.pop_size, 
                                 'sequence': [args.initial_seq] * args.pop_size})
    

    #load models
    print('\nloading esm.pretrained.esmfold_v1... \n')
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()

    print('running PFES... \n')
    if args.evolution_mode == "single_chain":
        fold_evolver(args, model, evolver, logheader, init_gen)
    elif args.evolution_mode == "inter_chain":
        inter_fold_evolver(args, model, evolver, logheader, init_gen)
    elif args.evolution_mode == "multimer":
        print("sorry, I am not ready yet")
    elif not args.evolution_mode in ['single_chain', 'inter_chain', 'multimer']:
        print("Unknown PFES mode: aveilable options are: single_chain, inter_chain or multimer")






