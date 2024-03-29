import matplotlib.pyplot as plt
import pandas as pd 
import sys, os, re

file = str(sys.argv[1])
plots_dir = os.path.dirname(file) + '/plots/'



def make_plots(file):
    os.makedirs(plots_dir, exist_ok=True)
    data=['id', 'len', 'helix_penalty', 'len_penalty', 'ptm', 'plddt', 'num_cont', 'num_icont', 'score', 'seq','dssp'] #
    df = pd.read_csv(file, sep='\t', names=data)

    for dat in data[1:-2]:
        plt.plot(df[dat],'.')
        plt.legend([dat], loc ="upper left")
        plt.savefig(plots_dir + dat + '.png')
        plt.close()

make_plots(file)
