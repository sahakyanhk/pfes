# PFES: protein fold evolution simulation



Code for [In silico evolution of globular protein folds from random sequences
Harutyun Sahakyan, Sanasar Babajanyan, Yuri I. Wolf, Eugene V. Koonin
bioRxiv 2024.11.10.622830; doi: https://doi.org/10.1101/2024.11.10.622830](https://www.biorxiv.org/content/10.1101/2024.11.10.622830v1)


This code requires [ESMfold](https://github.com/facebookresearch/esm) to run. 

### Instalation and usage examples 
```
wget https://github.com/sahakyanhk/PFES/archive/refs/heads/alpha.zip -O pfes-alpha.zip; unzip pfes-alpha.zip

python pfes-alpha/pfes.py -h

#run a simulation starting from random peptides and analyse results 
python pfes-alpha/pfes.py  -ng 100 -ps 50 -sm weak -em single_chain -iseq random --random_seq_len 24 -o pfes_test_random
python pfes-alpha/visual_pfes.py -l pfes_test_random/progress.log -s pfes_test_random/structures/ -o pfes_test_random/

#run a simulation starting from polyalanine
python pfes-alpha/pfes.py  -ng 100 -ps 50 -sm weak -em single_chain -iseq AAAAAAAAAAAAAAAAAAAAAAAA -o pfes_test_polyA

```
### Extended data
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14061036.svg)](https://doi.org/10.5281/zenodo.14061036)


### Hardware requirements 
PFES was tested on Rocky Linux 8.7 (Green Obsidian) with NVIDIA Tesla V100 and A100 GPUs. 



