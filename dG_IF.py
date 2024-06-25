#https://github.com/KULL-Centre/_2024_cagiada_stability/blob/main/stab_ESM_IF.ipynb



import os,time,subprocess,re,sys,shutil
import torch
import numpy as np
import pandas as pd


import esm

from esm.inverse_folding.util import load_structure, extract_coords_from_structure,CoordBatchConverter
from esm.inverse_folding.multichain_util import extract_coords_from_complex,_concatenate_coords,load_complex_coords

IF_model_name = "esm_if1_gvp4_t16_142M_UR50.pt"

print("importing the model")

model, alphabet = esm.pretrained.load_model_and_alphabet(IF_model_name)
model.eval().cuda().requires_grad_(False)


#@title PRELIMINARY OPERATIONS: Load EXTRA functions

def run_model(coords,sequence,model,cmplx=False,chain_target='A'):

    device = next(model.parameters()).device

    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, sequence)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(
        batch, device=device)

    prev_output_tokens = tokens[:, :-1].to(device)
    target = tokens[:, 1:]
    target_padding_mask = (target == alphabet.padding_idx)

    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)

    logits_swapped=torch.swapaxes(logits,1,2)
    token_probs = torch.softmax(logits_swapped, dim=-1)

    return token_probs

def score_variants(sequence,token_probs,alphabet):

    aa_list=[]
    wt_scores=[]
    skip_pos=0

    alphabetAA_L_D={'-':0,'_' :0,'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}
    alphabetAA_D_L={v: k for k, v in alphabetAA_L_D.items()}

    for i,n in enumerate(sequence):
      aa_list.append(n+str(i+1))
      score_pos=[]
      for j in range(1,21):
          score_pos.append(masked_absolute(alphabetAA_D_L[j],i, token_probs, alphabet))
          if n == alphabetAA_D_L[j]:
            WT_score_pos=score_pos[-1]

      wt_scores.append(WT_score_pos)

    return aa_list, wt_scores

def masked_absolute(mut, idx, token_probs, alphabet):

    mt_encoded = alphabet.get_idx(mut)

    score = token_probs[0,idx, mt_encoded]
    return score.item()


#@title  MODEL RUN
#@markdown Run this cell to evaluate the ΔG for the selected structure and sequence

#@markdown **N.B:** the ΔG value will be output in the scale of the chosen metric and also in kcal/mol (see the manuscript for how we converted the scale)
a=0.10413378327743603 ## fitting param from the manuscript to convert IF score scale to kcal/mol
b=0.6162549378400894 ## fitting param from the manuscript to convert IF score scale to kcal/mol

prob_tokens = run_model(coords_structure,sequence_structure,model,chain_target=chain_id)
aa_list, wt_scores = score_variants(sequence_structure,prob_tokens,alphabet)

dg_IF= np.nansum(wt_scores)
print('ΔG predicted (likelihoods sum): ',dg_IF)

dg_kcalmol= a * dg_IF + b

print('ΔG predicted (kcal/mol): ', dg_kcalmol)

aa_list_export=aa_list+['dG_IF','dG_kcalmol']
wt_scores_export=wt_scores+[dg_IF,dg_kcalmol]

df_export=pd.DataFrame({'Residue':aa_list_export,'score':wt_scores_export})

df_export.to_csv(f"outputs/"+f"{jobname}_dG_pos_scores_and_total.csv",sep=',')