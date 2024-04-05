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


#by number of codons
evoldict2 ={'A' : 4,  'C' : 2,  'D' : 2,  'E' : 2,  
            'F' : 2,  'G' : 4,  'H' : 2,  'I' : 3,  
            'K' : 2,  'L' : 4,  'M' : 1,  'N' : 2,  
            'P' : 4,  'Q' : 2,  'R' : 2,  'S' : 2,  
            'T' : 4,  'V' : 4,  'W' : 1,  'Y' : 2,  
            '+' : 1,   #insertion
            '-' : 1,   #single deletion
            '*' : 1,   #partial duplication
            '/' : 1,   #partial deletion
            'd' : 0.1} #full duplication    

#aafreq in sr_filter           this is normalized by codons
#"A" => 0.07422,                'A'	:	4	0.078
#"C" => 0.02469,                'C'	:	2	0.039
#"D" => 0.05363,                'D'	:	2	0.039
#"E" => 0.05431,                'E'	:	2	0.039
#"F" => 0.04742,                'F'	:	2	0.039
#"G" => 0.07415,                'G'	:	4	0.078
#"H" => 0.02621,                'H'	:	2	0.039
#"I" => 0.06792,                'I'	:	3	0.059
#"K" => 0.05816,                'K'	:	2	0.039
#"L" => 0.09891,                'L'	:	4	0.078
#"M" => 0.02499,                'M'	:	1	0.020
#"N" => 0.04465,                'N'	:	2	0.039
#"P" => 0.03854,                'P'	:	4	0.078
#"Q" => 0.03426,                'Q'	:	2	0.039
#"R" => 0.05161,                'R'	:	2	0.039
#"S" => 0.05723,                'S'	:	2	0.039
#"T" => 0.05089,                'T'	:	4	0.078
#"V" => 0.07292,                'V'	:	4	0.078
#"W" => 0.01303,                'W'	:	1	0.020
#"Y" => 0.03228,                'Y'	:	2	0.039


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

