from subprocess import Popen, PIPE
import tempfile
import sys, os

pfesdir = os.path.dirname(os.path.realpath(__file__))
psiquepath = pfesdir + '/bin/psique'
os.chmod(psiquepath, 0o755)

def pypsique(pdb_txt, chainid='A'):    
    cmd = psiquepath+' --format stride /dev/stdin' \
        + "| awk 'BEGIN { ORS = \"\" } $1==\"ASG\" && $3==\"" +chainid+ "\" {print $6}'" #udate bin path 
 

#    process = Popen(cmd, #["./sh/psique --format stride ", input_pdb_path, " | awk 'BEGIN { ORS = \"\" } $1==\"ASG\" && $3==\"" , chainid,  "\" {print $6}'"], 
    process = Popen(cmd,
                    stdin=PIPE,
                    stdout=PIPE, 
                    stderr=PIPE, shell=True)
    
    stdout, stderr = process.communicate(input=str.encode(pdb_txt))
    
    SSstring = stdout.decode('ascii')
    simplestring = SSstring.replace('G', 'C').replace('F', 'C').replace('T', 'C').replace('P', 'C')
    maxhelix = len(max(simplestring.replace('E', 'C').split('C')))
    maxbeta = len(max(simplestring.replace('H', 'C').split('C')))
    
    return SSstring, int(maxhelix), int(maxbeta)


# input_pdb_path = str(sys.argv[1])
# chainid = str(sys.argv[2])

# if  os.path.isfile(input_pdb_path) and chainid.isascii(): 
#     file = open(input_pdb_path,'r')
#     pdb_txt = file.read()
#     print(pypsique(pdb_txt, "A"))
