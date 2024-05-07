from subprocess import Popen, PIPE
import sys, os

def pypsique(input_pdb_path, chainid='A'):    
    
    cmd = "/data/saakyanh2/WD/PFES/pfes/bin/psique --format stride " +input_pdb_path+ " | awk 'BEGIN { ORS = \"\" } $1==\"ASG\" && $3==\"" +chainid+ "\" {print $6}'" #udate bin path here
    print(cmd)
    process = Popen(cmd, #["./sh/psique --format stride ", input_pdb_path, " | awk 'BEGIN { ORS = \"\" } $1==\"ASG\" && $3==\"" , chainid,  "\" {print $6}'"], 
                    stdout=PIPE, 
                    stderr=PIPE, shell=True)
    stdout, stderr = process.communicate()
    SSstring = stdout.decode('ascii')
    maxhelix = len(max(SSstring.replace('G', 'C').replace('F', 'C').replace('T', 'C').replace('P', 'C').split('C')))
    
    return  SSstring, int(maxhelix)

#input_pdb_path = str(sys.argv[1])
#chainid = str(sys.argv[2])
#
#if os.path.isfile(input_pdb_path): 
#    print(pypsique(input_pdb_path, chainid))
#
#

