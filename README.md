# PFES: protein fold evolution simulation

Code for "[In silico evolution of globular protein folds from random sequences](https://www.biorxiv.org/content/10.1101/2024.11.10.622830v1)"

This code requires [ESMfold](https://github.com/facebookresearch/esm) to run. 

### Instalation and usage
wget https://github.com/sahakyanhk/PFES/archive/refs/heads/alpha.zip -O pfes-alpha.zip; unzip pfes-alpha.zip\
python pfes-alpha/pfes.py -h

python pfes-alpha/pfes.py  -ng 500 -ps 50 -sm weak -em single_chain -iseq random --random_seq_len 24 -o pfes_test 

