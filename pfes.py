import argparse
import os, sys, shutil
import pandas as pd
import numpy as np
import typing as T
import threading
import multiprocessing
import time
import torch
import esm

from evolver import sequence_mutator, selector, randomseq
from score import get_nconts, get_inter_nconts
from psique import pypsique
from datetime import datetime



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


def create_batched_sequence_datasets(sequences: T.List[T.Tuple[str, str]], max_tokens_per_batch: int = 1524
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
        if num_sequences > args.pop_size / 2: #TODO CHECK THIS WITH args.pop_size / 4 and lartge pop size
           yield batch_headers, batch_sequences
           batch_headers, batch_sequences, num_tokens, num_sequences= [], [], 0, 0
    yield batch_headers, batch_sequences


def sigmoid(x,L0=0,c=0.1):
    return 1 / (1+2.71828182**(c * (L0-x)))


def esm2data(esm_out):
    output = {key: value.cpu() for key, value in esm_out.items()}
    pdbs = model.output_to_pdb(output)
    ptm = esm_out["ptm"].tolist()
    mean_plddt = esm_out["mean_plddt"].tolist()
    return(pdbs, ptm, mean_plddt)


def extract_results(gen_i, headers, sequences, pdbs, ptms, mean_plddts):
    global new_gen #this will be modified in the fold_evolver()
    
    for full_id, seq, pdb_txt, ptm, _mean_plddt_, in zip(headers, sequences, pdbs, ptms, mean_plddts):
        
        all_seqs = seq.split(':')
        seq = all_seqs[0]
        seq_len = len(seq)
        
        id_data = full_id.split('_')

        id = id_data[0]
        prev_id = id_data[1]
        mutation = id_data[2]

        with open(pdb_path + id + '.pdb', 'w') as f: # TODO conver this into a function
            f.write(pdb_txt)   

        #================================SCORING================================# 
        num_conts, mean_plddt = get_nconts(pdb_txt, 'A', 6.0, 50)
        
        if args.evolution_mode == "single_chian": #if there are two or more chains, then calculate the number of interacting contacts
            num_inter_conts = 1
        else:
            num_inter_conts, _ = get_inter_nconts(pdb_txt, 'A', 'B', 6.0, 50) 

        ss, max_helix = pypsique(pdb_path + id + '.pdb', 'A')
        #Rg, aspher = get_aspher(pdb_txt)
        prot_len_penalty =  (1 - sigmoid(seq_len, args.prot_len_penalty, 0.2)) * np.tanh(seq_len*0.1)
        max_helix_penalty = 1 - sigmoid(max_helix, args.helix_len_penalty, 0.5)
        score  = np.prod([mean_plddt,           #[0, 1]
                          ptm,                  #[0, 1]
                          prot_len_penalty,     #[0, 1]
                          max_helix_penalty,    #[0, 1]
                          num_conts**(1/3),     #[~0, inf]
                          num_inter_conts**(1/4)])   #[~0, inf]
        #================================SCORING================================#
        iterlog = pd.DataFrame({'gndx': gen_i,
                                'prev_id': prev_id,
                                'id': id, 
                                'seq_len': seq_len,
                                'prot_len_penalty': round(prot_len_penalty, 3), 
                                'max_helix_penalty': round(max_helix_penalty, 3),
                                'ptm': round(ptm, 3), 
                                'mean_plddt': mean_plddt, 
                                'num_conts': num_conts, 
                                'num_inter_conts': num_inter_conts, 
                                'score': round(score, 3), 
                                'sequence': seq, 
                                'mutation': mutation,
                                'ss': ss}, index=[0])
    
        new_gen = pd.concat([new_gen, iterlog], axis=0, ignore_index=True) 
    print(new_gen.tail(args.pop_size).drop('gndx', axis=1).to_string(index=False, header=False))



#========================================CONCEPTS========================================# 
def fold_evolver(args): 
    print('not ready yet')
def multimer_evolver(model, args):  
    print("evolution of interacting dimers")
#========================================CONCEPTS========================================# 

global new_gen #this will be modified in the extract_results() 

#============================================================================#
#================================FOLD_EVOLVER================================# 
def fold_evolver(args, model, logheader, init_gen): 

    os.makedirs(pdb_path, exist_ok=True)
    with open(os.path.join(args.outpath, args.log), 'w') as f:
        f.write(logheader)


    #creare an initial pool of sequences with pop_size
    columns=['gndx',
             'prev_id',
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
             'mutation',
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
            seq, mutation_data= sequence_mutator(sequence)
            
            #chek if the mutated seqeuece was already predicted
            seqmask = ancestral_memory.sequence == seq 
            
            #if --norepeat and seq is in the ancestral_memory mutate it again
            if args.norepeat and seqmask.any():  
                while seqmask.any():
                    seq, mutation_data = sequence_mutator(seq)
                    seqmask = ancestral_memory.sequence == seq 

            id = "g{0}seq{1}_{2}_{3}".format(gen_i, n, prev_id, mutation_data); n+=1 # give an uniq id even if the same sequence already exists            

            if seqmask.any(): #if sequence already exits do not predict a structure again 
                repeat = ancestral_memory[seqmask].drop_duplicates(subset=['sequence'], keep='last') 
                try:
                    shutil.copyfile(pdb_path + repeat.id.values[0] + '.pdb', pdb_path + id.split('_')[0] + '.pdb')  
                except  FileNotFoundError: 
                    pass
                repeat.id = id.split('_')[0] #assing a new id to the already exiting sequence
                new_gen = new_gen.append(repeat)
            else:
                generated_sequences.append((id, seq)) 
                mutation_collection.append(mutation_data)    


        batched_sequences = create_batched_sequence_datasets(generated_sequences, args.max_tokens_per_batch)
        

        #predict data for the new batch
        for headers, sequences in batched_sequences:
            pdbs, ptms, mean_plddts = [], [], []
            with torch.no_grad(): 
                pdbs, ptms, mean_plddts = esm2data(model.infer(sequences, 
                                                               num_recycles = args.num_recycles,
                                                               residue_index_offset = 1,
                                                               chain_linker = "G" * 25))
            
            #run extract_results() in becground and imediately start next round of model.infer()
            trd = threading.Thread(target=extract_results, args=(gen_i, headers, sequences, pdbs, ptms, mean_plddts))
            trd.start()


            #extract_results(gen_i, id, headers, sequences, pdbs, ptms, mean_plddts)
            # p1 = multiprocessing.Process(target=extract_results, args=(gen_i, id, headers, sequences, pdbs, ptms, mean_plddts))
            # p1.start()
            # p1.join()

        while trd.is_alive(): 
            time.sleep(0.2)
        
        #print(f"#GENtime {datetime.now() - now}")
        ancestral_memory =  ancestral_memory.append(init_gen)

        #select the next generation 
        init_gen = selector(new_gen, init_gen, args.pop_size, args.selection_mode, args.norepeat)
        init_gen.gndx = f'gndx{gen_i}' #assign a new gen index
        init_gen.to_csv(os.path.join(args.outpath, args.log), mode='a', index=False, header=False, sep='\t')

#================================FOLD_EVOLVER================================# 
#============================================================================# 





#==================================================================================#
#================================INTER_FOLD_EVOLVER================================# 

def inter_fold_evolver(args, model, logheader, init_gen): 

    #evolution of an interacting chain
    PDB_6WXQ=":MKSYFVTMGFNETFLLRLLNETSAQKEDSLVIVVPSPIVSGTRAAIESLRAQISRLNYPPPRIYEIEITDFNLALSKILDIILTLPEPIISDLTMGMRMINLILLGIIVSRKRFTVYVRDE" # 6WXQ (12 to 134) 
    NZ_CP011286=":LNIIKLFHGHKYCLIFYVLP" #intergenic region from Yersinia
    PDB_1RFA=":ASNTIRVFLPNKQRTVVNVRNGMSLHDCLMKALKVRGLQPECCAVFRLLHEHKGKKARLDWNTDAASLIGEELQVDFLD" #1RFA (55 to 132)
    PDB_1RFP=":QCRRLCYKQRCVTYCRGR" # 1RFP contains S-S bond
    PDB_4REX=":DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLSQ" #4REX (170 to 207)
    PDB_6SVE=":WEKRMSRNSGRVYYFNHITNASQF" #WW domain
    PDB_4QR0=":MMVLVTYDVNTETPAGRKRLRHVAKLCVDYGQRVQNSVFECSVTPAEFVDIKHRLTQIIDEKTDSIRFYLLGKNWQRRVETLGRSDSYDPDKGVLLL" #Cas2 from Streptococcus pyogenes serotype M1 (301447)
    PDB_4QR02=":MMVLVTYDVNTETPAGRKRLRHVAKLCVDYGQRVQNSVFECSVTPAEFVDIKHRLTQIIDEKTDSIRFYLLGKNWQRRVET" #Cas2 from Streptococcus pyogenes serotype M1 (301447)
    PDB_6M6W=":MNDIIINKIATIKRCIKRIQQVYGDGSQFKQDFTLQDSVILNLQRCCEACIDIANHINRQQQLGIPQSSRDSFTLLAQNNLITQPLSDNLKKMVGLRNIAVHDAQELNLDIVVHVVQHHLEDFEQFIDVIKAE" #HEPN toxin
    PDB_5YIW=":GAMDMSWTDERVSTLKKLWLDGLSASQIAKQLGGVTRNAVIGKVHRLGL" #HTH
    PDB_4OO8=":GQKNSRERMKRIEEGIKELGSQILKEHPVENTQLQNEKLYLYYLQNGRDMYVDQELDINRLSDYDVDHIVPQSFLKDDSIDNKVLTRSDKNRGKSDNVPSEEVVKKMKNYWRQLLNAKLITQRKFDNLTKAERGGL" #CAS9 HNH
    
    seq2 = PDB_5YIW

    os.makedirs(pdb_path, exist_ok=True)
    with open(os.path.join(args.outpath, args.log), 'w') as f:
        f.write(logheader)


    #creare an initial pool of sequences with pop_size
    columns = ['gndx',
               'prev_id',
               'id', 
               'seq_len', 
               'prot_len_penalty', s
               'max_helix_penalty',
               'ptm', 
               'mean_plddt', 
               'num_conts', 
               'num_inter_conts',
               'score', 
               'sequence', 
               'mutation',
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
            seq, mutation_data= sequence_mutator(sequence)
            
            #chek if the mutated seqeuece was already predicted
            seqmask = ancestral_memory.sequence == seq 
            
            #if --norepeat and seq is in the ancestral_memory mutate it again
            if args.norepeat and seqmask.any():  
                while seqmask.any():
                    seq, mutation_data = sequence_mutator(seq)
                    seqmask = ancestral_memory.sequence == seq 

            id = "g{0}seq{1}_{2}_{3}".format(gen_i, n, prev_id, mutation_data); n+=1 # give an uniq id even if the same sequence already exists            

            if seqmask.any(): #if sequence already exits do not predict a structure again 
                repeat = ancestral_memory[seqmask].drop_duplicates(subset=['sequence'], keep='last') 
                try:
                    shutil.copyfile(pdb_path + repeat.id.values[0] + '.pdb', pdb_path + id.split('_')[0] + '.pdb')  
                except  FileNotFoundError: 
                    pass
                repeat.id = id.split('_')[0] #assing a new id to the already exiting sequence
                new_gen = new_gen.append(repeat)
            else:
                generated_sequences.append((id, seq +":"+ seq)) #(seq+seq2)) add a function to select the sma
                mutation_collection.append(mutation_data)    


        batched_sequences = create_batched_sequence_datasets(generated_sequences, args.max_tokens_per_batch)
        

        #predict data for the new batch
        for headers, sequences in batched_sequences:
            pdbs, ptms, mean_plddts = [], [], []
            with torch.no_grad(): 
                pdbs, ptms, mean_plddts = esm2data(model.infer(sequences, 
                                                               num_recycles = args.num_recycles,
                                                               residue_index_offset = 1,
                                                               chain_linker = "G" * 25))
            
            #run extract_results() in becground and imediately start next round of model.infer()
            trd = threading.Thread(target=extract_results, args=(gen_i, headers, sequences, pdbs, ptms, mean_plddts))
            trd.start()

            # p1 = multiprocessing.Process(target=extract_results, args=(gen_i, id, headers, sequences, pdbs, ptms, mean_plddts))
            # p1.start()
            # p1.join()

        while trd.is_alive(): 
            time.sleep(0.2)
        
        #print(f"#GENtime {datetime.now() - now}")
        ancestral_memory =  ancestral_memory.append(init_gen)

        #select the next generation 
        init_gen = selector(new_gen, init_gen, args.pop_size, args.selection_mode, args.norepeat)
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
            '-iseq', '--initial_seq', type=str,
            help='a sequence to initiate with, if "random" pop_size random sequneces will be generated, the lenght of the random sequences can be assigned with "--random_seq_len"',
            default='random'
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
            '-pl0', '--prot_len_penalty', type=int,
            help='population size',
            default=100,
    )
    parser.add_argument(
            '-hl0', '--helix_len_penalty', type=int,
            help='population size',
            default=25,
    )
    parser.add_argument(
            '--random_seq_len', type=int,
            help='a sequence to initiate with',
            default=18,
    )
    parser.add_argument(                      
            '--norepeat', action='store_true', 
            help='do not allow to generate the same sequences', 
    )
    parser.add_argument(
            '--nobackup', action='store_true', 
            help='owerride files if exists',
    )
    parser.add_argument(
        '--num-recycles',
        type=int,
        default=4,
        help="Number of recycles to run. Defaults to number used in training (4).",
    )
    parser.add_argument(
        '--max-tokens-per-batch',
        type=int,
        default=1024,
        help="Maximum number of tokens per gpu forward-pass. This will group shorter sequences together "
        "for batched prediction. Lowering this can help with out of memory issues, if these occur on "
        "short sequences.",
    )

    args = parser.parse_args()
    

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
#--log, -l\t\t\t = {args.log}
#--outpath, -o\t\t\t = {args.outpath}
#--helix_len_penalty, -hl0\t = {args.helix_len_penalty}
#--prot_len_penalty, -pl0\t = {args.prot_len_penalty}
#--num_generations, -ng\t\t = {args.num_generations}
#--pop_size, -ps\t\t = {args.pop_size}
#--random_seq_len\t\t = {args.random_seq_len}
#--norepeat\t\t\t = {args.norepeat}
#--nobackup\t\t\t = {args.nobackup}
#--num-recycles\t\t\t = {args.num_recycles}
#--max-tokens-per-batch\t\t = {args.max_tokens_per_batch}
#
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
        randomsequence = randomseq(args.random_seq_len)
        init_gen = pd.DataFrame({'id': ['init_seq'] * args.pop_size, 
                                 'sequence': [randomsequence] * args.pop_size})

    elif args.initial_seq == 'randoms':
        init_gen = pd.DataFrame({'id': [f'init_seq{i}' for i in range(args.pop_size)], 
                                 'sequence': [randomseq(args.random_seq_len) for i in range(args.pop_size)]})

    else: 
        init_gen = pd.DataFrame({'id': ['init_seq'] * args.pop_size, 
                                 'sequence': [args.initial_seq] * args.pop_size})
    


    # TODO check arguments and input paths before loading models 
    #load models
    print('\nloading esm.pretrained.esmfold_v1... \n')
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()


    if args.evolution_mode == "single_chain":
        fold_evolver(args, model, logheader, init_gen)
    elif args.evolution_mode == "inter_chain":
        inter_fold_evolver(args, model, logheader, init_gen)
    elif args.evolution_mode == "multimer":
        print("sorry, I am not ready yet")
    elif not args.evolution_mode in ['single_chain', 'inter_chain', 'multimer']:
        print("Unknown PFES mode: aveilable options are: single_chain, inter_chain or multimer")






