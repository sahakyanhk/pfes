import random
import numpy as np
import pandas as pd



flatrates ={'A' : 1,  'C' : 1,  'D' : 1,  'E' : 1,  
            'F' : 1,  'G' : 1,  'H' : 1,  'I' : 1,  
            'K' : 1,  'L' : 1,  'M' : 1,  'N' : 1,  
            'P' : 1,  'Q' : 1,  'R' : 1,  'S' : 1,  
            'T' : 1,  'V' : 1,  'W' : 1,  'Y' : 1,  
            '+' : 0.8,    #insertion
            '-' : 1,    #single deletion
            '*' : 0,#0.3,    #partial duplication
            '/' : 0,#0.3,    #random insertion
            '%' : 0,#1,    #partial deletion
            'd' : 0.01} #full duplication    


#by number of codons
codontrates ={'A' : 4,  'C' : 2,  'D' : 2,  'E' : 2,  
              'F' : 2,  'G' : 4,  'H' : 2,  'I' : 3,  
              'K' : 2,  'L' : 6,  'M' : 1,  'N' : 2,  
              'P' : 4,  'Q' : 2,  'R' : 6,  'S' : 2,  
              'T' : 4,  'V' : 4,  'W' : 1,  'Y' : 2,  
              '+' : 3,   #insertion
              '-' : 3,   #single deletion
              '*' : 2,   #partial duplication
              '#' : 2,   #random insertion
              '%' : 2,   #partial deletion
              'd' : 0.1} #full duplication    

#https://www.uniprot.org/uniprotkb/statistics#amino-acid-composition
uniprotrates = {'A' : 0.0826, 'C' : 0.0139, 'D' : 0.0546, 'E' : 0.0672, 
                'F' : 0.0387, 'G' : 0.0707, 'H' : 0.0228, 'I' : 0.0591, 
                'K' : 0.0580, 'L' : 0.0965, 'M' : 0.0241, 'N' : 0.0406, 
                'P' : 0.0475, 'Q' : 0.0393, 'R' : 0.0553, 'S' : 0.0665, 
                'T' : 0.0536, 'V' : 0.0686, 'W' : 0.0110, 'Y' : 0.0292}


evoldict = flatrates

mutation_types = list(evoldict.keys())  #mutation type
p = list(evoldict.values()) #probability for each mutation
aa_alphabet = mutation_types[:20] #allowed substitutions for point mutations 

w = p[:20] #probabilities for random sequence generation
upw = list(uniprotrates.values())
    
#random sequence generator
def randomseq(nres=20, weights=w):
    return ''.join(random.choices(aa_alphabet, weights=weights, k=nres))


def sequence_mutator(sequence):  
    mutation_position = random.choice(range(len(sequence)))
    mutation =  random.choices(mutation_types, weights=p)[0]
    
    if mutation in aa_alphabet:
        sequence_mutated = sequence[:mutation_position] + mutation + sequence[mutation_position + 1:]
        mutation_info = f'{sequence[mutation_position]}{mutation_position+1}.{mutation}'

    elif mutation =='+':
        mutation = random.choices(aa_alphabet)[0]
        sequence_mutated = sequence[:mutation_position + 1] + mutation + sequence[mutation_position + 1:]
        mutation_info = f'{sequence[mutation_position]}{mutation_position+1}+{mutation}'

    elif mutation == '-':
        sequence_mutated = sequence[:mutation_position] + sequence[mutation_position + 1:]
        mutation_info = f'{sequence[mutation_position]}{mutation_position+1}-'

    elif mutation =='*' and len(sequence) > 5: #partial duplication
        insertion_len = random.choice(range(2, int(len(sequence)/2))) #what is the probable insertion lenght?
        sequence_mutated = sequence[:mutation_position] + sequence[mutation_position:][:insertion_len] + sequence[mutation_position:]
        mutation_info = f'{sequence[mutation_position]}{mutation_position+1}*{sequence[mutation_position:][:insertion_len]}'

    elif mutation =='/': #random insertion
        mutation = randomseq(random.choice(range(2, int(len(sequence)/2))), weights=upw) #using UP rates for random insertions
        sequence_mutated = sequence[:mutation_position + 1] + mutation + sequence[mutation_position + 1:]
        mutation_info = f'{sequence[mutation_position]}{mutation_position+1}/{mutation}'

    elif mutation =='%' and len(sequence) > 5: #partial deletion
        deletion_len = random.choice(range(2, int(len(sequence)/2))) #what is the probable deletion lenght?
        sequence_mutated = sequence[:mutation_position] + sequence[mutation_position + deletion_len:]
        mutation_info = f'{sequence[mutation_position]}{mutation_position+1}%{deletion_len}'

    elif mutation =='d':
        linker = randomseq(4)
        sequence_mutated = sequence + linker + sequence     
        mutation_info = f'd{linker}'
        
    elif mutation =='r' and len(sequence) > 5: #TODO recombination 
        sequence_mutated = sequence
        mutation_info = f'{mutation_position+1}'

    return sequence_mutated, mutation_info



def selector(input_new_gen, input_init_gen, pop_size, selection_mode, norepeat): # TODO add differen selection modes here
    mixed_pop = pd.concat([input_new_gen, input_init_gen], axis=0, ignore_index=True) 
    
    if norepeat:
        mixed_pop = mixed_pop.drop_duplicates(subset=['sequence'])

    if selection_mode == "strong":
        new_init_gen = mixed_pop.sort_values('score', ascending=False).head(pop_size)

    if selection_mode == "weak":
        weights = np.array(mixed_pop.score / mixed_pop.score.sum())
        weights[np.isnan(weights)] = 1e-100
        new_init_gen = mixed_pop.sample(n=pop_size, weights=weights).sort_values('score', ascending=False)

    return new_init_gen

