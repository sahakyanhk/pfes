import os 
import re
import sys
import shutil
import pandas as pd
from pymol import cmd
from moviepy.editor import ImageClip, concatenate_videoclips

#usage: python visual_pfes.py outdir

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)



wd = os.path.abspath(str(sys.argv[1])) # get path from the first argument
pdb_dir = wd + '/pdb/'
selected_pdb = wd + '/pdb2/'
frames = wd + '/frames/'
wdname = os.path.basename(wd)

os.chdir(wd)

if os.path.exists(frames):
    shutil.rmtree(frames)
os.makedirs(frames, exist_ok=True)

if os.path.exists(selected_pdb):
    shutil.rmtree(selected_pdb)
os.makedirs(selected_pdb, exist_ok=True)


names=['id','len', 'len_p', 'helix_p', 'ptm', 'plddt', 'nconts', 'niconts', 'score', 'seq','dssp'] #
log = pd.read_csv('output.log', sep='\t', header=None, names=names)


print('preparing strucutres with best score')

pdbs=[]
for i in range(len(log[log.id.str.contains(f'seq0')])):
    gen1 = log[log.id.str.contains(f'gen{i}_')]
    pdbid = gen1[gen1.score == gen1.score.max()].head(1).id.item()
    pdbs.append(pdbid +'.pdb')
pdbs = pdbs[0:len(pdbs):1] #selec every one change 1 to 2 to select every 2th

n = len(pdbs)

for pdb in pdbs: 
    shutil.copy(pdb_dir + pdb, selected_pdb + pdb)

print(f'{n} structures will be rendered')

cmd.load(f'{pdb_dir}/gen0_seq0.pdb', 'pdb_0')
cmd.orient()
# cmd.color("grey", 'pdb_0' and "chain B")
# cmd.spectrum("b", "blue_white_red",  'pdb_0' and "chain A")
#view = cmd.get_view(0) # or set view from pymol

view = (\
     0.126038805,   -0.159885079,    0.979056120,\
    -0.870360196,    0.455739141,    0.186471283,\
    -0.476010621,   -0.875634670,   -0.081716686,\
     0.000000000,    0.000000000, -163.258407593,\
    -0.677261353,   -1.067066193,    0.053632736,\
   133.446701050,  193.070144653,  -20.000000000 )

# cmd.set_view(view)
# cmd.png('frames/frame_0.png', width=800, height=600, dpi=30)


print(pdbs)

i = 0
for pdb in pdbs: 
    i+=1
    cmd.load(pdb_dir + pdb, f'pdb_{i}')
    q=f'pdb_{i}'
    t=f'pdb_{i-1}'
    cmd.align(q, t)
    cmd.delete(t)
    cmd.set_view(view)
    cmd.bg_color(color="white")
    cmd.color("white", q and "chain B")
    cmd.spectrum("b", "red_yellow_blue",  q and "chain A", minimum=0, maximum=100)
    cmd.show("sticks",  q and "chain A")
    cmd.set("ray_shadows", 1)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_trace_gain", 25)
    #cmd.ray(800, 800, -1, 0, 0, 0)
    cmd.png(f'{frames}/frame_{i}.png', width=1800, height=1800, dpi=300, ray=1, quiet=1)
    print(f'{i} of {n}')


clips = [ImageClip(frames + m).set_duration(0.1)
           for m in sorted_alphanumeric(os.listdir(frames))]

concat_clip = concatenate_videoclips(clips, 
                                     method="compose", 
                                     bg_color=(255, 255, 255))
concat_clip.write_videofile(f'{wdname}.mp4', 24)
