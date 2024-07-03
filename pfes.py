import argparse
import os, sys, shutil
import pandas as pd
import numpy as np
import typing as T
import threading
import gzip
import time
import torch
import esm
from scipy.special import softmax


from evolution import Evolver
from score import get_nconts, cbiplddt
from psique import pypsique
from datetime import datetime


#class PFEStools():

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

def pdbtxt2bbcoord(pdb_txt, chain):
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
    ptm = esm_out["ptm"].tolist()
    mean_plddt = esm_out["mean_plddt"].tolist()
    plddt = np.array(output["plddt"][0,:,1].tolist())/100
    
    #calculate the numbe of contacts
    # bins = np.append(0,np.linspace(2.3125,21.6875,63))
    # #you do not need softmax to keep the actual values! 
    # sm_contacts = softmax(output["distogram_logits"],-1)
    # sm_contacts = sm_contacts[...,bins<8].sum(-1)
    # mask = output["atom37_atom_exists"][0,:,1] == 1
    # contact_map = sm_contacts[0][mask,:][:,mask]
    # num_conts = []
    """
    Return the number of contact and individual plddts (write it in the log). 
    In the case of dimers, return also the number of interchain interaction with indexes. 
    Use indexes to calculate iPLDDT

    """
    return(pdbs, ptm, mean_plddt) #return score instead



def selection_mode_changer(plddt:float, ptm:float) -> str:
    if ((init_gen['mean_plddt'] > plddt) & (init_gen['ptm'] > ptm)).any():
        return 'strong'

#to score.py

def sigmoid(x,L0=0,c=0.1):
    return 1 / (1+2.71828182**(c * (L0-x)))



#--------------------------------------------

def extract_results(gen_i, headers, sequences, pdbs, ptms, mean_plddts) -> None:
    global new_gen #this will be modified in the fold_evolver()

    for full_id, seq, pdb_txt, ptm, _mean_plddt_, in zip(headers, sequences, pdbs, ptms, mean_plddts):
        
        all_seqs = seq.split(':')
        seq = all_seqs[0]
        seq_len = len(seq)
        
        id_data = full_id.split('_')

        id = id_data[0]
        prev_id = id_data[1]
        mutation = id_data[2]

        with open(pdb_path + id + '.pdb', 'wb') as f: # TODO conver this into a function
            f.write(pdb_txt.encode())   

        #================================SCORING================================# 
        num_conts, mean_plddt = get_nconts(pdb_txt, 'A', 6.0, 50)
        
        if args.evolution_mode == "single_chian": #if there are two or more chains, then calculate the number of interacting contacts
            num_inter_conts, iplddt = 1,1
        else:
            num_inter_conts, iplddt = cbiplddt(pdb_txt, 'A', 'B', 6.0, 50) 

        ss, max_helix = pypsique(pdb_path + id + '.pdb', 'A')
        #Rg, aspher = get_aspher(pdb_txt)
        prot_len_penalty =  (1 - sigmoid(seq_len, args.prot_len_penalty, 0.2)) * np.tanh(seq_len*0.1)
        max_helix_penalty = 1 - sigmoid(max_helix, args.helix_len_penalty, 0.5)
        score  = np.prod([mean_plddt,           #[0, 1]
                          ptm,                  #[0, 1]
                          prot_len_penalty,     #[0, 1]
                          max_helix_penalty,    #[0, 1]
                          iplddt,               #[0, 1]
                          (num_conts + 2*seq_len) / seq_len])     #[~0, inf]
        
        #score  = np.prod([mean_plddt, ptm])   #[~0, inf]
        #================================SCORING================================#
        iterlog = pd.DataFrame({'gndx': gen_i,
                                'id': id, 
                                'seq_len': seq_len,
                                'prot_len_penalty': round(prot_len_penalty, 3), 
                                'max_helix_penalty': round(max_helix_penalty, 3),
                                'ptm': round(ptm, 3), 
                                'mean_plddt': mean_plddt, 
                                'num_conts': num_conts, 
                                'iplddt': iplddt,
                                'num_inter_conts': num_inter_conts, 
                                'score': round(score, 3), 
                                'sequence': seq, 
                                'mutation': mutation,
                                'prev_id': prev_id,
                                'ss': ss}, index=[0])
        
        new_gen = pd.concat([new_gen, iterlog], axis=0, ignore_index=True) 
        os.system(f"gzip {pdb_path}{id}'.pdb' &")

        # with open(pdb_path + id + '.pdb', 'rb') as f_pdb:
        #     with gzip.open(pdb_path + id + '.pdb.gz', 'wb') as f_pdb_gz:
        #         shutil.copyfileobj(f_pdb, f_pdb_gz)
        #         shutil

    print(new_gen.tail(args.pop_size).drop('gndx', axis=1).to_string(index=False, header=False))

    

#========================================CONCEPTS========================================# 
def multimer_evolver(model, args):  
    print("evolution of interacting dimers")
#========================================CONCEPTS========================================# 

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
             'max_helix_penalty',
             'ptm', 
             'mean_plddt', 
             'num_conts', 
             'iplddt',
             'num_inter_conts',
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
            
            #check if the mutated seqeuece was already predicted
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
                new_gen = new_gen.append(repeat)
            else:
                generated_sequences.append((id, seq)) 
                mutation_collection.append(mutation_data)    

        
        batched_sequences = create_batched_sequence_datasets(generated_sequences, args.max_tokens_per_batch)
        

        #predict data for the new batch
        for headers, sequences in batched_sequences:
            pdbs, ptms, mean_plddts = [], [], []
            with torch.no_grad(): 
                pdbs, ptms, mean_plddts  = esm2data(model.infer(sequences, 
                                                               num_recycles = args.num_recycles,
                                                               residue_index_offset = 1,
                                                               chain_linker = "G" * 25))
            
            #run extract_results() in becground and imediately start next round of model.infer()
            trd = threading.Thread(target=extract_results, args=(gen_i, headers, sequences, pdbs, ptms, mean_plddts))
            trd.start()

        while trd.is_alive(): 
            time.sleep(0.2)
        
        #print(f"#GENtime {datetime.now() - now}")
        ancestral_memory =  ancestral_memory.append(init_gen)
        #select the next generation 
        init_gen = evolver.select(new_gen, init_gen, args.pop_size, args.selection_mode, args.norepeat)
        init_gen.gndx = f'gndx{gen_i}' #assign a new gen index
        init_gen.to_csv(os.path.join(args.outpath, args.log), mode='a', index=False, header=False, sep='\t')

        #write init_gen as a checkpoit file to continue the simulation


        #if condition:
            #then the rest here
        #Change the selection with a condition (plddt, ptm)
        if args.strong_sm_by_condition:
            if (init_gen['mean_plddt'] > 0.6) & (init_gen['ptm'] > 0.5).any() & condition:
                args.selection_mode = 'strong'
                condition = False #do not change args.selection_mode anymore
                with open(os.path.join(args.outpath, args.log), mode='a') as f:
                    f.write("#changing the selection mode to strong")

        #Change the selection mode after n generations
        if args.strong_sm_after_n_steps > 0:
            if (gen_i > args.strong_sm_after_n_steps) & condition:
                args.selection_mode = 'strong'
                condition = False #do not change args.selection_mode anymore
                print("#changing the selection mode to strong")
                with open(os.path.join(args.outpath, args.log), mode='a') as f:
                    f.write("#changing the selection mode to strong")

        #STOPPER
        if args.stop_by_condition:
            if (init_gen['mean_plddt'] > 0.9) & (init_gen['ptm'] > 0.8).any():
                print(f'gndx={gen_i}; the condition reached, breaking!')
                break


#================================FOLD_EVOLVER================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================================================================================================# 
#==================================================================================#
#================================INTER_FOLD_EVOLVER================================# 

def inter_fold_evolver(args, model, evolver, logheader, init_gen) -> None: 

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
               'id', 
               'seq_len', 
               'prot_len_penalty',
               'max_helix_penalty',
               'ptm', 
               'mean_plddt', 
               'num_conts', 
               'iplddt',
               'num_inter_conts',
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
            
            #chek if the mutated seqeuece was already predicted
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
                new_gen = new_gen.append(repeat)
            else:
                generated_sequences.append((id, seq +":"+ seq2)) #(seq+seq2)) add a function to select the sma
                mutation_collection.append(mutation_data)    


        batched_sequences = create_batched_sequence_datasets(generated_sequences, args.max_tokens_per_batch)
        

        #predict data for the new batch
        for headers, sequences in batched_sequences:
            pdbs, ptms, mean_plddts = [], [], []
            with torch.no_grad(): 
                pdbs, ptms, mean_plddts, num_contacts = esm2data(model.infer(sequences, 
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
            '-ed', '--evoldict', type=str,
            help='population size',
            default='codonrates',
    )
    parser.add_argument(
            '-pl0', '--prot_len_penalty', type=int,
            help='population size',
            default=200,
    )
    parser.add_argument(
            '-hl0', '--helix_len_penalty', type=int,
            help='population size',
            default=20,
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
            help='',
    )
    parser.add_argument(
            '--strong_sm_by_condition', action='store_true', 
            help='',
    )
    parser.add_argument(
            '--strong_sm_after_n_steps', type=int,
            help='',
            default=0,
    )
    # parser.add_argument(
    #         '--continue', action='store_true', 
    #         help='owerride files if exists',
    # )
    parser.add_argument(
            '--num-recycles',
            type=int,
            default=16,
            help="Number of recycles to run. Defaults to number used in training (4).",
    )
    parser.add_argument(
            '--max-tokens-per-batch',
            type=int,
            default=5120,
            help="Maximum number of tokens per gpu forward-pass. This will group shorter sequences together "
            "for batched prediction. Lowering this can help with out of memory issues, if these occur on "
            "short sequences.",
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
#--log, -l\t\t\t = {args.log}
#--outpath, -o\t\t\t = {args.outpath}
#--helix_len_penalty, -hl0\t = {args.helix_len_penalty}
#--prot_len_penalty, -pl0\t = {args.prot_len_penalty}
#--num_generations, -ng\t\t = {args.num_generations}
#--pop_size, -ps\t\t = {args.pop_size}
#--evoldict, -ed\t\t = {args.evoldict}
#--random_seq_len\t\t = {args.random_seq_len}
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
                                 'sequence': [randomsequence] * args.pop_size})

    elif args.initial_seq == 'randoms':
        init_gen = pd.DataFrame({'id': [f'init_seq{i}' for i in range(args.pop_size)], 
                                 'sequence': [evolver.randomseq(args.random_seq_len) for i in range(args.pop_size)]})

    else: 
        init_gen = pd.DataFrame({'id': ['init_seq'] * args.pop_size, 
                                 'sequence': [args.initial_seq] * args.pop_size})
    


    # TODO check arguments and input paths before loading models 
    #load models
    print('\nloading esm.pretrained.esmfold_v1... \n')
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()


    if args.evolution_mode == "single_chain":
        fold_evolver(args, model, evolver, logheader, init_gen)
    elif args.evolution_mode == "inter_chain":
        inter_fold_evolver(args, model, evolver, logheader, init_gen)
    elif args.evolution_mode == "multimer":
        print("sorry, I am not ready yet")
    elif not args.evolution_mode in ['single_chain', 'inter_chain', 'multimer']:
        print("Unknown PFES mode: aveilable options are: single_chain, inter_chain or multimer")






