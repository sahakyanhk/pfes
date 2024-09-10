import random
import numpy as np
import pandas as pd
import typing as T



class Evolver():

    flatrates = {'A' : 1,  'C' : 1,  'D' : 1,  'E' : 1,  
                 'F' : 1,  'G' : 1,  'H' : 1,  'I' : 1,  
                 'K' : 1,  'L' : 1,  'M' : 1,  'N' : 1,  
                 'P' : 1,  'Q' : 1,  'R' : 1,  'S' : 1,  
                 'T' : 1,  'V' : 1,  'W' : 1,  'Y' : 1
                 }
    
    flatoptim = {'A' : 1,  'C' : 1,  'D' : 1,  'E' : 1,  
                 'F' : 1,  'G' : 1,  'H' : 1,  'I' : 1,  
                 'K' : 1,  'L' : 1,  'M' : 1,  'N' : 1,  
                 'P' : 1,  'Q' : 1,  'R' : 1,  'S' : 1,  
                 'T' : 1,  'V' : 1,  'W' : 1,  'Y' : 1,
                 '+' : 1,
                 '-' : 1
                 }


    #by number of codons. 
    # Calculated as (Codon_i/sum(all codons)) * 20
    # so the mean probability is 1. 
    # This makes it easier to optimize probability of non point mutations

    codonrates = {'A' : 1.311475, #4 
                  'C' : 0.655738, #2
                  'D' : 0.655738, #2
                  'E' : 0.655738, #2
                  'F' : 0.655738, #2
                  'G' : 1.311475, #4
                  'H' : 0.655738, #2
                  'I' : 0.983607, #3
                  'K' : 0.655738, #2
                  'L' : 1.967213, #6
                  'M' : 0.327869, #1
                  'N' : 0.655738, #2
                  'P' : 1.311475, #4
                  'Q' : 0.655738, #2
                  'R' : 1.967213, #6
                  'S' : 1.967213, #6
                  'T' : 1.311475, #4
                  'V' : 1.311475, #4
                  'W' : 0.327869, #1
                  'Y' : 0.655738 #2
                  }



    #https://www.uniprot.org/uniprotkb/statistics#amino-acid-composition
    uniprotrates = {'A' : 0.0826, 'C' : 0.0139, 'D' : 0.0546, 'E' : 0.0672, 
                    'F' : 0.0387, 'G' : 0.0707, 'H' : 0.0228, 'I' : 0.0591, 
                    'K' : 0.0580, 'L' : 0.0965, 'M' : 0.0241, 'N' : 0.0406, 
                    'P' : 0.0475, 'Q' : 0.0393, 'R' : 0.0553, 'S' : 0.0665, 
                    'T' : 0.0536, 'V' : 0.0686, 'W' : 0.0110, 'Y' : 0.0292}


    codonrates_pmo = {'A' : 1.311475, #4
                      'C' : 0.655738, #2 
                      'D' : 0.655738, #2 
                      'E' : 0.655738, #2 
                      'F' : 0.655738, #2 
                      'G' : 1.311475, #4 
                      'H' : 0.655738, #2 
                      'I' : 0.983607, #3 
                      'K' : 0.655738, #2 
                      'L' : 1.967213, #6 
                      'M' : 0.327869, #1 
                      'N' : 0.655738, #2 
                      'P' : 1.311475, #4 
                      'Q' : 0.655738, #2 
                      'R' : 1.967213, #6 
                      'S' : 1.967213, #6 
                      'T' : 1.311475, #4 
                      'V' : 1.311475, #4 
                      'W' : 0.327869, #1 
                      'Y' : 0.655738, #2        
                      '+' : 1,
                      '-' : 1,
                      }


    non_point_mutations = {'+' : 1.0,   #single residue insertion
                        '-' : 1.0,   #single residue deletion
                        '*' : 0.4,   #partial duplication
                        '/' : 0.4,   #random insertion 
                        '%' : 0.9,   #partial deletion
                        'p' : 0.1,   #Circular permutation
                        'd' : 0.05   #full duplication    
                        } 


    one2three = {'C': 'CYS', 'D': 'ASP', 'S': 'SER', 'Q': 'GLN', 'K': 'LYS',
                 'I': 'ILE', 'P': 'PRO', 'T': 'THR', 'F': 'PHE', 'N': 'ASN', 
                 'G': 'GLY', 'H': 'HIS', 'L': 'LEU', 'R': 'ARG', 'W': 'TRP', 
                 'A': 'ALA', 'V': 'VAL', 'E': 'GLU', 'Y': 'TYR', 'M': 'MET'}


    three2one = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
                 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
                 'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


    evoldicts = {'flatrates': flatrates, 'codonrates': codonrates, 'flatoptim': flatoptim, 'uniprotrates': uniprotrates} 

    def __init__(self, evoldict: str):
        
        try:
            if evoldict == 'flatoptim':
                self.evoldict = Evolver.evoldicts[evoldict]
            else:
                self.evoldict = Evolver.evoldicts[evoldict]
                self.evoldict.update(self.non_point_mutations) # add non point mutations 
        except: 
            print(f'WARNING! Unknown evoldict "{evoldict}", "codonrates" will be used. \nAvailable evoldicts are {list(Evolver.evoldicts.keys())}. \nSee github for details' )
            self.evoldict = Evolver.evoldicts["codonrates"]
            pass
        self.evoldict_normal = {j:round(i/sum(self.evoldict.values()),4) for i, j in zip(self.evoldict.values(), self.evoldict.keys())}
        self.mutation_types = list(self.evoldict.keys())  #mutation type
        self.p = list(self.evoldict.values()) #probability for each mutation
        self.aa_alphabet = self.mutation_types[:20] #allowed substitutions for point mutations 
        self.w = self.p[:20] #probabilities for random sequence generation


    #random sequence generator
    def randomseq(self, nres=24, weights = None ) -> str:
        return ''.join(random.choices(self.aa_alphabet, weights=self.w, k=nres))
    

    def mutate(self, sequence: str) -> T.Tuple[str, str]:  
            seq_len = len(sequence)
            if seq_len < 6:
                mutation = 'd'
            else:
                mutation_position = random.choice(range(seq_len))
                mutation =  random.choices(self.mutation_types, weights=self.p)[0]
            
            if mutation in self.aa_alphabet:
                sequence_mutated = sequence[:mutation_position] + mutation + sequence[mutation_position + 1:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}.{mutation}'

            elif mutation =='+':
                mutation = random.choices(self.aa_alphabet)[0]
                sequence_mutated = sequence[:mutation_position + 1] + mutation + sequence[mutation_position + 1:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}+{mutation}'

            elif mutation == '-':
                sequence_mutated = sequence[:mutation_position] + sequence[mutation_position + 1:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}-'

            elif mutation =='*' and seq_len > 5: #partial duplication
                insertion_len = random.choice(range(2, int(seq_len/2))) #TODO what is the probable insertion lenght?
            #   insertion_len = round(np.random.normal(loc=round(seq_len/2), scale=1.0, size=None))  to use normal distribution. TODO try also exp decline          
                sequence_mutated = sequence[:mutation_position] + sequence[mutation_position:][:insertion_len] + sequence[mutation_position:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}*{sequence[mutation_position:][:insertion_len]}'

            elif mutation =='/': #random insertion
                mutation = self.randomseq(random.choice(range(2, int(seq_len/2)))) 
                sequence_mutated = sequence[:mutation_position + 1] + mutation + sequence[mutation_position + 1:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}/{mutation}'

            elif mutation =='%' and seq_len > 5: #partial deletion
                deletion_len = random.choice(range(2, int(seq_len/2))) #what is the probable deletion lenght?
                sequence_mutated = sequence[:mutation_position] + sequence[mutation_position + deletion_len:]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}%{deletion_len}'

            elif mutation =='p' and seq_len > 5: #permutation 
                sequence_mutated =  sequence[mutation_position:] + sequence[:mutation_position]
                mutation_info = f'{sequence[mutation_position]}{mutation_position+1}p{mutation}'

            elif mutation =='d': #full duplication #TODO reduce the duplication probability with sequence growth
                linker = self.randomseq(2)
                sequence_mutated = sequence + linker + sequence     
                mutation_info = f'd{linker}'
                
            elif mutation =='r' and seq_len > 5: #TODO recombination 
                sequence_mutated = sequence
                mutation_info = f'{mutation_position+1}'

            #random change for a chanck of the sequence. (imitation of a frameshift)
            return sequence_mutated, mutation_info



    def select(self, input_new_gen, input_init_gen, pop_size:int, selection_mode:str, norepeat:bool, beta = 1): 
        mixed_pop = pd.concat([input_new_gen, input_init_gen], axis=0, ignore_index=True) 
        
        e=2.71828182846

        if norepeat:
            mixed_pop = mixed_pop.drop_duplicates(subset=['sequence'])

        if selection_mode == "strong":
            new_init_gen = mixed_pop.sort_values('score', ascending=False).head(pop_size)

        if selection_mode == "weak":
            weights = np.array((1- beta + beta * mixed_pop.score) / ((1 - beta + beta * mixed_pop.score).sum()))

#            weights = np.array(e**(beta * mixed_pop.score) / (e**(beta * mixed_pop.score).sum()))

            weights[np.isnan(weights)] = 1e-100
            new_init_gen = mixed_pop.sample(n=pop_size, weights=weights, replace=(not norepeat)).sort_values('score', ascending=False)

        return new_init_gen

