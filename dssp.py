from subprocess import Popen, PIPE
import sys, os

def pydssp(input_pdb_path, chainid='A'):    
    process = Popen(['./bin/dssp.sh', input_pdb_path, chainid], 
                     stdout=PIPE, 
                     stderr=PIPE, universal_newlines=True)
    stdout, stderr = process.communicate()
    dssp, max_helix = stdout.split(',')
    return  dssp, int(max_helix)

input_pdb_path = str(sys.argv[1])
chainid = str(sys.argv[2])

if  os.path.isfile(input_pdb_path): 
    print(pydssp(input_pdb_path, chainid))





"""
import subprocess
import re


def parse_dssp_output(input_pdb_path, chain):
    
    
    # Parses DSSP program text output, extracts secondary structure for a specific chain,
    # and returns the length of the longest H-helix segment.

    # Args:
    #     input_pdb_path (str): Path to the input PDB file.
    #     chain (str): Chain ID to process.

    # Returns:
    #     int: Length of the longest H-helix segment in the specified chain.


    process = subprocess.Popen(["dssp", input_pdb_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"DSSP execution failed with error: {stderr.decode()}")

    # Filter lines for the specified chain (based on 11th character)
    chain_data = [line for line in stdout.splitlines() if line[11] == chain]

    # Extract secondary structure assignments using a regular expression
    ss_pattern = r"\s{16}([A-Z])"
    ss = "".join(re.findall(ss_pattern, "".join(chain_data)))

    # Find the longest H-helix segment
    longest_helix = max(len(h) for h in re.findall(r"(H+)", ss))

    return longest_helix


if __name__ == "__main__":
    input_pdb_path = input("Enter PDB file path: ")
    chain = input("Enter chain ID: ")

    try:
        longest_helix = parse_dssp_output(input_pdb_path, chain)
        print(f"Length of longest H-helix segment in chain {chain}: {longest_helix}")
    except RuntimeError as e:
        print(f"Error: {e}")

"""

