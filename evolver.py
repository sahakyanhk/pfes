import random
import numpy as np
import pandas as pd


evoldict ={'A' : 1,  'C' : 1,  'D' : 1,  'E' : 1,  
           'F' : 1,  'G' : 1,  'H' : 1,  'I' : 1,  
           'K' : 1,  'L' : 1,  'M' : 1,  'N' : 1,  
           'P' : 1,  'Q' : 1,  'R' : 1,  'S' : 1,  
           'T' : 1,  'V' : 1,  'W' : 1,  'Y' : 1,  
           '+' : 1,   #insertion
           '-' : 1,   #single deletion
           '*' : 1,   #partial duplication
           '/' : 1,   #partial deletion
           'd' : 0.1} #full duplication    

mutation_types = list(evoldict.keys())  #mutation type
p = list(evoldict.values()) #probability for each mutation
aa_alphabet = mutation_types[:20] #allowed substitutions for point mutations 

w = p[:20] #probabilities for random sequence generation

    
#random sequence generator
def randomseq(nres=18, weights=w):
    return ''.join(random.choices(aa_alphabet, weights=weights, k=nres))


def sequence_mutator(sequence):  
    mutation_posiotion = random.choice(range(len(sequence)))
    mutation =  random.choices(mutation_types, weights=p)[0]
    
    if mutation in aa_alphabet:
        sequence_mutated = sequence[:mutation_posiotion] + mutation + sequence[mutation_posiotion + 1:]
    
    elif mutation =='+':
        mutation = random.choices(aa_alphabet)[0]
        sequence_mutated = sequence[:mutation_posiotion + 1] + mutation + sequence[mutation_posiotion + 1:]
    
    elif mutation == '-':
        sequence_mutated = sequence[:mutation_posiotion] + sequence[mutation_posiotion + 1:]
    
    elif mutation =='*' and len(sequence) > 5:
        insertion_len = random.choice(range(2, int(len(sequence)/2))) #what is the probable insertion lenght?
        sequence_mutated = sequence[:mutation_posiotion] + sequence[mutation_posiotion:][:insertion_len] + sequence[mutation_posiotion:]
    
    elif mutation =='/' and len(sequence) > 5:
        deletion_len = random.choice(range(2, int(len(sequence)/2))) #what is the probable deletion lenght?
        sequence_mutated = sequence[:mutation_posiotion] + sequence[mutation_posiotion + deletion_len:]
    
    elif mutation =='d':
        sequence_mutated = sequence + 'GGGG' + sequence     
    
    elif mutation =='r' and len(sequence) > 5: #TODO recombination 
        sequence_mutated = sequence
    
    return sequence_mutated



def selector(new_gen, init_gen, pop_size, selection_mode, norepeat): # TODO add differen selection modes here
    mixed_pop = pd.concat([new_gen, init_gen], axis=0, ignore_index=True) 
    
    if norepeat:
        mixed_pop = mixed_pop.drop_duplicates(subset=['sequence'])

    if selection_mode == "strong":
        new_init_gen = mixed_pop.sort_values('score', ascending=False).head(pop_size)

    if selection_mode == "weak":
        weights = np.array(mixed_pop.score / mixed_pop.score.sum())
        weights[np.isnan(weights)] = 0
        new_init_gen = mixed_pop.sample(n=pop_size, weights=weights).sort_values('score', ascending=False)

    return new_init_gen

