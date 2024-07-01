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
 


def get_nconts(pdb_txt, chain="A", distance_cutoff=6.0, plddt_cutoff=0): 
    """
    Calculates number of contaict in a protein.

    """
    
    # Get all C-alpha atoms with specific pLDDT cutoff
    ca_data, plddt = [],[]
    for line in pdb_txt.splitlines():
        atom_array = line.split()
        if atom_array[0] == 'ATOM'and atom_array[4] == chain:
            plddt.append(float(atom_array[10]))
        if (atom_array[0] == 'ATOM' and atom_array[2] == 'CB' and float(atom_array[10]) > plddt_cutoff and atom_array[4] == chain) :
            ca_data.append([
            int(atom_array[5]), # residue index 
            np.array(list(map(float, atom_array[6:9]))), #xyz
            float(atom_array[10]) #pLDDT   
            ])
    
    if len(ca_data) == 0:
        mean_plddt = np.mean(np.array(plddt))
        return(1, round(mean_plddt * 0.01, 3))
    else:    
        coords = np.array([item[1] for item in ca_data])  # Extract coordinates
        CA_pLDDT = np.mean(np.array([item[2] for item in ca_data]))
        n_atoms = len(coords)
        #pairs_data = np.zeros((0, 5))

        distances_matrix = np.linalg.norm(coords[:, None] - coords, axis=2)
        row = 0
        for i in range(n_atoms):
            for j in range(i + 4, n_atoms): # do not calc dist between atoms i, ... i+4
                if distances_matrix[i, j] < distance_cutoff:
                    #pairs_data = np.append(pairs_data, [[row, ca_data[i][0], ca_data[j][0], np.mean([ca_data[i][2], ca_data[j][2]]), distances_matrix[i, j]]], axis=0)
                    row += 1
        
        return(row+1, round(CA_pLDDT * 0.01, 3))




def get_inter_nconts(pdb_txt, chainA='A', chainB='B', distance_cutoff=6.0, plddt_cutoff=0): 
    """
    Calculates number of contaict between two protein chains
    returns a tuple (number of contacts, average plddt of residues with plddt > plddt_cutoff)
    """

    # Get all C-beta atoms with specific pLDDT cutoff
    cb_data_A, cb_data_B, = [], []
    for line in pdb_txt.splitlines():
        atom_array = line.split()
        if (atom_array[0] == 'ATOM' and atom_array[2] == 'CB' and float(atom_array[10]) > plddt_cutoff and atom_array[4] == chainA) :
            cb_data_A.append([
            int(atom_array[5]), # residue index 
            np.array(list(map(float, atom_array[6:9]))), #xyz
            float(atom_array[10]) #pLDDT   
            ])
        if (atom_array[0] == 'ATOM' and atom_array[2] == 'CB' and float(atom_array[10]) > plddt_cutoff and atom_array[4] == chainB) :
            cb_data_B.append([
            int(atom_array[5]), # residue index 
            np.array(list(map(float, atom_array[6:9]))), #xyz
            float(atom_array[10]) #pLDDT   
            ])

    if len(cb_data_A) == 0 or len(cb_data_B) == 0: 
        return(1, 1)
    else:    
        Acoords = np.array([item[1] for item in cb_data_A])
        Bcoords = np.array([item[1] for item in cb_data_B])
        CA_pLDDT_A = np.mean(np.array([item[2] for item in cb_data_A]))

        #make pairs of coordinates and calculate distace between them
        n_atoms_A = len(cb_data_A)
        n_atoms_B = len(cb_data_B)
        pairs_data = np.zeros((0, 4))

        distances_matrix = np.linalg.norm(Acoords[:, None] - Bcoords, axis=2)
        contact_map = distances_matrix.copy()
        contact_map[contact_map <= distance_cutoff] = 1
        contact_map[contact_map > distance_cutoff] = 0
        n_contacts = contact_map.sum()
        return(n_contacts, round(CA_pLDDT_A * 0.01, 3))



def cbiplddt(pdb_txt, chainA='A', chainB='B', distance_cutoff=6.0, plddt_cutoff=0):
    """
    Calculates number of contaict between two protein chains and iPLDDT
    """

    # Get all C-beta atoms with specific pLDDT cutoff
    cbeta_atom = []
    for line in pdb_txt.splitlines():
            if line[:4] == 'ATOM' and line[13:15] == "CB":
                cbeta_atom.append(line)
    cbeta_array = [['X' for j in range(8)] for i in range(len(cbeta_atom))]
    for row in range(len(cbeta_atom)):
        cbeta_array[row]
        cbeta_array[row][0] = row					#Index
        cbeta_array[row][1] = (cbeta_atom[row][17:20]).strip()	#Residue Name
        cbeta_array[row][2] = (cbeta_atom[row][20:22]).strip()	#ChainID
        cbeta_array[row][3] = (cbeta_atom[row][22:26]).strip()	#Residue Number
        cbeta_array[row][4] = (cbeta_atom[row][30:38]).strip()	#xyz
        cbeta_array[row][5] = (cbeta_atom[row][38:46]).strip()	#xyz
        cbeta_array[row][6] = (cbeta_atom[row][46:54]).strip()	#xyz
        cbeta_array[row][7] = (cbeta_atom[row][61:66]).strip()	#pLDDT 

    cb_data_A, cb_data_B, = [], []
    for row in range(len(cbeta_array)):
        if (cbeta_array[row][2] == chainA and float(cbeta_array[row][7]) > plddt_cutoff):
            cb_data_A.append(cbeta_array[row])
        if (cbeta_array[row][2] == chainB and float(cbeta_array[row][7]) > plddt_cutoff):
            cb_data_B.append(cbeta_array[row])
    if len(cb_data_A) == 0 or len(cb_data_B) == 0: 
        return(1, 1)
    else:    
        Acoords = np.array([item[4:7] for item in cb_data_A], dtype="float32")
        Bcoords = np.array([item[4:7] for item in cb_data_B], dtype="float32")
        CA_pLDDT_A = np.array([item[7] for item in cb_data_A], dtype="float32").mean()
        distances_matrix = np.linalg.norm(Acoords[:, None] - Bcoords, axis=2)
        contact_map = distances_matrix.copy()
        contact_map[contact_map <= distance_cutoff] = 1
        contact_map[contact_map > distance_cutoff] = 0
        n_contacts = contact_map.sum()
        inteface_ndx = np.where(contact_map)

        return(n_contacts, round(CA_pLDDT_A * 0.01, 3), inteface_ndx)


def iplddt_all_atom(pdb_txt, chainA='A', chainB='B', distance_cutoff=6.0,):
    iplddt_all_atom = 'not ready yet'
    return iplddt_all_atom




input_pdb_path = str(sys.argv[1])

if  os.path.isfile(input_pdb_path): 
    file = open(input_pdb_path,'r')
    pdb_txt = file.read()


    print("inner contancts:" + str(get_nconts(pdb_txt, "A", 6.0, 0)))
    print("intra contancts:" + str(get_inter_nconts(pdb_txt,"A","B", 6.0, 0)))


