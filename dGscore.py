#https://github.com/KULL-Centre/_2024_cagiada_stability/blob/main/stab_ESM_IF.ipynb
import numpy as np
import pandas as pd


import torch
import esm

from esm.inverse_folding.util import CoordBatchConverter

model_if, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model_if.eval().cuda().requires_grad_(False)

def dGscore(coord:np.array, sequence:str):

    def run_model_if(coords, sequence, model_if, cmplx=False, chain_target='A'):

        device = next(model_if.parameters()).device

        batch_converter = CoordBatchConverter(alphabet)
        batch = [(coords, None, sequence)]
        coords, confidence, strs, tokens, padding_mask = batch_converter(
            batch, device=device)

        prev_output_tokens = tokens[:, :-1].to(device)
        target = tokens[:, 1:]
        target_padding_mask = (target == alphabet.padding_idx)

        logits, _ = model_if.forward(coords, padding_mask, confidence, prev_output_tokens)

        logits_swapped=torch.swapaxes(logits,1,2)
        token_probs = torch.softmax(logits_swapped, dim=-1)

        return token_probs

    def score_variants(sequence, token_probs, alphabet):

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


    prob_tokens = run_model_if(coord, sequence, model_if)
    aa_list, wt_scores = score_variants(sequence, prob_tokens, alphabet)

    a=0.10413378327743603 ## fitting param from the manuscript to convert IF score scale to kcal/mol
    b=0.6162549378400894 ## fitting param from the manuscript to convert IF score scale to kcal/mol

    dg_IF= np.nansum(wt_scores)
    dg_kcalmol= a * dg_IF + b

    return(dg_kcalmol)

    # aa_list_export=aa_list+['dG_IF','dG_kcalmol']
    # wt_scores_export=wt_scores+[dg_IF,dg_kcalmol]

    # df_export=pd.DataFrame({'Residue':aa_list_export,'score':wt_scores_export})

    # df_export.to_csv(f"{jobname}_dG_pos_scores_and_total.csv",sep=',')