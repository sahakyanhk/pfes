import numpy as np
import sys, os
import math



def get_aspher(pdb_txt):

    #from DOI: 10.1016/j.bpj.2018.01.002 (HullRad)

    def distance(ax,ay,az, bx,by,bz):
        #Euclidean distance between two atoms
        return math.sqrt((ax - bx)**2.0 + (ay - by)**2.0 + (az - bz)**2.0)

    def model_from_pdb(pdb_txt):
        
        all_atm_rec = []
        # Get all relevant atoms even if in wrong order
        for line in pdb_txt.splitlines():
            if (line[:4] == 'ATOM'):
                all_atm_rec.append(line)
        # Convert all_atm_rec to multi-item list
        all_atm_array = [['X' for j in range(8)] for i in range(len(all_atm_rec))]
        for row in range(len(all_atm_rec)):
            all_atm_array[row][0] = row					#Atom Index
            all_atm_array[row][1] = (all_atm_rec[row][11:16]).strip()	#Atom Name
            all_atm_array[row][2] = (all_atm_rec[row][17:20]).strip()	#Residue Name
            all_atm_array[row][3] = (all_atm_rec[row][20:22]).strip()	#ChainID
            all_atm_array[row][4] = (all_atm_rec[row][22:26]).strip()	#Residue Number
            all_atm_array[row][5] = (all_atm_rec[row][30:38]).strip()	#x
            all_atm_array[row][6] = (all_atm_rec[row][38:46]).strip()	#y
            all_atm_array[row][7] = (all_atm_rec[row][46:54]).strip()	#z
        return all_atm_array

    all_atm_rec = model_from_pdb(pdb_txt)

    # Radius of Gyration
    # Calc center of mass
    X = 0.0
    Y = 0.0
    Z = 0.0
    tot = 0.0
    for row in range(len(all_atm_rec)):
        X = X + (float(all_atm_rec[row][5]))
        Y = Y + (float(all_atm_rec[row][6]))
        Z = Z + (float(all_atm_rec[row][7]))
        tot += 1
    com_x = (X/tot)
    com_y = (Y/tot)
    com_z = (Z/tot)
    Rg2  = 0.0
    for row in range(len(all_atm_rec)):
        Rg2 += ((distance(com_x, com_y, com_z, float(all_atm_rec[row][5]),\
            float(all_atm_rec[row][6]), float(all_atm_rec[row][7])))**2)
    Rg = math.sqrt(Rg2/tot)

    asphr = 0.0	
    Ixx,Ixy,Ixz,Iyy,Iyz,Izz = 0,0,0,0,0,0
    for row in range(len(all_atm_rec)):
        Ixx += ((float(all_atm_rec[row][5])) - com_x) * ((float(all_atm_rec[row][5])) - com_x)
        Ixy += ((float(all_atm_rec[row][5])) - com_x) * ((float(all_atm_rec[row][6])) - com_y)
        Ixz += ((float(all_atm_rec[row][5])) - com_x) * ((float(all_atm_rec[row][7])) - com_z)
        Iyy += ((float(all_atm_rec[row][6])) - com_y) * ((float(all_atm_rec[row][6])) - com_y)
        Iyz += ((float(all_atm_rec[row][6])) - com_y) * ((float(all_atm_rec[row][7])) - com_z)
        Izz += ((float(all_atm_rec[row][7])) - com_z) * ((float(all_atm_rec[row][7])) - com_z)
    Ixx= Ixx/row
    Iyy= Iyy/row
    Izz= Izz/row
    Ixy= Ixy/row
    Ixz= Ixz/row
    Iyz= Iyz/row
    gyration_tensor = [[Ixx,Ixy,Ixz],[Ixy,Iyy,Iyz],[Ixz,Iyz,Izz]]

    #print(gyration_tensor)
    evals, evecs = np.linalg.eig(gyration_tensor)
    L1 = evals[0]
    L2 = evals[1]
    L3 = evals[2]
    asphr = ((L1 - L2)**2 + (L2 - L3)**2 + (L1 - L3)**2)/(2.0*((L1 + L2 + L3)**2))

    return(Rg, asphr)
 


def get_nconts(pdb_txt, chain="A", distance_cutoff=6.5, plddt_cutoff=0): 
    """
    Calculates number of contaict in a protein.

    """

    # Get all C-alpha atoms with specific pLDDT cutoff
    ca_data, plddt = [],[]
    for line in pdb_txt.splitlines():
        col = line.split()
        if col[0] == 'ATOM'and col[4] == chain:
            plddt.append(float(col[10]))
        if (col[0] == 'ATOM' and col[2] == 'CB' and float(col[10]) > plddt_cutoff and col[4] == chain) :
            ca_data.append([
            int(col[5]), # residue index 
            np.array(list(map(float, col[6:9]))), #xyz
            float(col[10]) #pLDDT   
            ])
    
    if len(ca_data) == 0:
        mean_plddt = np.mean(np.array(plddt))
        return(1, round(mean_plddt * 0.01, 2))
    else:    
        coords = np.array([item[1] for item in ca_data])  # Extract coordinates
        CA_pLDDT = np.mean(np.array([item[2] for item in ca_data]))
        n_atoms = len(coords)
        pairs_data = np.zeros((0, 4))

        distances_matrix = np.linalg.norm(coords[:, None] - coords, axis=2)
        row = 0
        for i in range(n_atoms):
            for j in range(i + 4, n_atoms): # do not calc dist between atom i, ... i+4
                if distances_matrix[i, j] < distance_cutoff:
                    pairs_data = np.append(pairs_data, [[row, ca_data[i][0], ca_data[j][0], distances_matrix[i, j]]], axis=0)
                    row += 1
        return(len(pairs_data)+1, round(CA_pLDDT * 0.01, 2))






def get_inter_nconts(pdb_txt, chainA='A', chainB='B', distance_cutoff=6.5, plddt_cutoff=0): 
    """
    Calculates number of contaict between two protein chains
    returns a tuple (number of contacts, average plddt of residues with plddt > plddt_cutoff)
    """

    # Get all C-beta atoms with specific pLDDT cutoff
    ca_data_A, ca_data_B, = [], []
    for line in pdb_txt.splitlines():
        col = line.split()
        if (col[0] == 'ATOM' and col[2] == 'CB' and float(col[10]) > plddt_cutoff and col[4] == chainA) :
            ca_data_A.append([
            int(col[5]), # residue index 
            np.array(list(map(float, col[6:9]))), #xyz
            float(col[10]) #pLDDT   
            ])
        if (col[0] == 'ATOM' and col[2] == 'CB' and float(col[10]) > plddt_cutoff and col[4] == chainB) :
            ca_data_B.append([
            int(col[5]), # residue index 
            np.array(list(map(float, col[6:9]))), #xyz
            float(col[10]) #pLDDT   
            ])


    if len(ca_data_A) == 0 or len(ca_data_B) == 0: 
        return(1, 1)
    else:    
        coords_A = np.array([item[1] for item in ca_data_A])
        coords_B = np.array([item[1] for item in ca_data_B])
        CA_pLDDT_A = np.mean(np.array([item[2] for item in ca_data_A]))

        #make pairs of coordinates and calculate distace between them
        n_atoms_A = len(ca_data_A)
        n_atoms_B = len(ca_data_B)
        pairs_data = np.zeros((0, 4))

        distances_matrix = np.linalg.norm(coords_A[:, None] - coords_B, axis=2)
        row = 0
        for i in range(n_atoms_A):
            for j in range(i + 1, n_atoms_B):
                if distances_matrix[i, j] < distance_cutoff:
                    pairs_data = np.append(pairs_data, [[row, ca_data_A[i][0], ca_data_B[j][0], distances_matrix[i, j]]], axis=0)
                    row += 1 
        return(len(pairs_data)+1, round(CA_pLDDT_A * 0.01, 2))


# for rosetta 
# def get_best_score(score_file_path):
#     score_dict = {}    
#     f=open(score_file_path, 'r').readlines()
#     for i in range(1,len(f)):
#         score_id = f[i].split()[-1][:-5] 
#         score = float(f[i].split()[1]) 
#         score_dict[score_id] = score
#     best_score_seq_id = min(score_dict, key=score_dict.get)
#     best_score = score_dict[best_score_seq_id]
#     return(best_score_seq_id, best_score)

#        cmd = "score -in:file:s {0} -out:file:scorefile {1} -score_app:linmin > /dev/null".format\
#                            (pdb_path + id + '.pdb', data_path + f'/gen_{num_gen}.dat') 
#        os.system(cmd)            
#        best_seq_id, best_score = get_best_score(data_path + f'/gen_{num_gen}.dat')
#        print(best_seq_id, tmp_d[best_seq_id], best_score)
#        if best_score < prev_best_score:
#            seq_init = tmp_d[best_seq_id][0]
#            prev_best_score = best_score


input_pdb_path = str(sys.argv[1])

if  os.path.isfile(input_pdb_path): 
    file = open(input_pdb_path,'r')
    pdb_txt = file.read()


    print("inner contancts:" + str(get_nconts(pdb_txt, "A", 6.5, 0)))
    print("intra contancts:" + str(get_inter_nconts(pdb_txt,"A","B", 6.5, 0)))


