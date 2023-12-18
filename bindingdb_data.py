import numpy as np
import pandas as pd
from openbabel import pybel
import math
import pickle
import os

# functions
def get_nodes(ligand, pocket):
    SMARTS = [
        '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
        '[a]',
        '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
        '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
        '[r]'
    ]
    #smarts_labels = ['hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']
    smarts_pattern = []
    for smart in SMARTS:
        smarts_pattern.append(pybel.Smarts(smart))

    p_atoms = [[atom.atomicnum, atom.coords, atom.hyb, atom.heavydegree, atom.heterodegree, atom.partialcharge, 0, 0, 0, 0, 0, 1] for atom in pocket]
    for (x_i, x) in enumerate(smarts_pattern):
        atoms_with_prop = np.array(list(*zip(*x.findall(pocket))), dtype=int) - 1
        for atom_i in atoms_with_prop:
            p_atoms[atom_i][x_i + 6] = 1
    p_atoms = [atoms for atoms in p_atoms if not atoms[0]==1]

    l_atoms = [[atom.atomicnum, atom.coords, atom.hyb, atom.heavydegree, atom.heterodegree, atom.partialcharge, 0, 0, 0, 0, 0, 0] for atom in ligand]
    for (x_i, x) in enumerate(smarts_pattern):
        atoms_with_prop = np.array(list(*zip(*x.findall(ligand))), dtype=int) - 1
        for atom_i in atoms_with_prop:
            l_atoms[atom_i][x_i + 6] = 1
    l_atoms = [atoms for atoms in l_atoms if not atoms[0]==1]

    c_atoms = [l_atom for l_atom in l_atoms]
    for p_atom in p_atoms:
        for l_atom in l_atoms:
            distance = math.dist(p_atom[1], l_atom[1])
            if distance <= 4: # within 4 Angstrom
                c_atoms.append(p_atom)
                break
    return c_atoms

def get_edges(vs):
    edge_list0 = [] # for 2 adjacency matrix model[0] <4
    edge_list1 = [] # for 2 adjacency matrix model[1] <4
    edge_list2 = [] # for fully adjacent layer <4
    for i in range(len(vs)):
        for j in range(i):
            distance = math.dist(vs[i][1], vs[j][1])
            if (vs[i][11] == vs[j][11]):
                if (distance>=0) and (distance<=2.0):
                    edge_list0.append((i,j))
                    edge_list0.append((j,i))
            else:
                if (distance>=0.0) and (distance<=2.0):
                    edge_list0.append((i,j))
                    edge_list0.append((j,i))
                if (distance>2.0) and (distance<=4.0):
                    edge_list1.append((i,j))
                    edge_list1.append((j,i))
            edge_list2.append((i,j))
            edge_list2.append((j,i))
    return edge_list0, edge_list1, edge_list2

data_path = '../kinase/data_kinase.tsv'
bindingdb_data = pd.read_csv(data_path, sep = '\t')

pdbbind_pdb = []
pdbbind_uni = []
with open("../kinase/pdbbind_pdb_uniprot.csv", 'r') as f:
    for l in f:
        line = l.strip().split(',')
        pdbbind_pdb.append(line[0])
        pdbbind_uni.append(line[1])


full_dock = []
full_dock1 = []
full_dock_15 = []
full_dock1_15 = []
exc_dock = []
exc_dock1 = []
exc_dock_15 = []
exc_dock1_15 = []
dataset_name = 'bindingdb'
for _, row in bindingdb_data.iterrows():
    target_name = row['UniProt_ID']
    ligand_name = row['PubChem_ID']
    affinity = row['Label']
    output_dir = '../kinase/ligand_docking/%s/%s' %(target_name, ligand_name)
    #pocket_path = "../kinase/target_pdbs/%s.pdb" %(target_name)
    pocket_path = "../kinase/target_pdbqt/%s.pdbqt" %target_name
    #for j in range(9):
    for j in range(3):
        i = j + 1
        ligand_path = "%s/docking%d.pdbqt" %(output_dir, i)
        """
        if os.path.exists(ligand_path):
            with open(ligand_path, 'r') as f:
                for l in f:
                    if 'RESULT' in l:
                        line = l.split()
                        rmsd = float(line[len(line)-2])
                        break
        """
        name = '%s_%s' %(target_name, ligand_name)
        if os.path.exists(pocket_path) and os.path.exists(ligand_path):
            #if rmsd >= 3:
            #    continue
            ligand = next(pybel.readfile('pdbqt', ligand_path))
            pocket = next(pybel.readfile('pdbqt', pocket_path))
            vs = get_nodes(ligand, pocket)
            es0, es1, es2 = get_edges(vs)
            full_dock.append([name, affinity, vs, es0, es1, es2, dataset_name])
            if float(affinity) <= 15.22:
                full_dock_15.append([name, affinity, vs, es0, es1, es2, dataset_name])
            if i == 1:
                full_dock1.append([name, affinity, vs, es0, es1, es2, dataset_name])
                if float(affinity) <= 15.22:
                    full_dock1_15.append([name, affinity, vs, es0, es1, es2, dataset_name])
            if target_name in pdbbind_uni:
                exc_dock.append([name, affinity, vs, es0, es1, es2, dataset_name])
                if float(affinity) <= 15.22:
                    exc_dock_15.append([name, affinity, vs, es0, es1, es2, dataset_name])
                if i == 1:
                    exc_dock1.append([name, affinity, vs, es0, es1, es2, dataset_name])
                    if float(affinity) <= 15.22:
                        exc_dock1_15.append([name, affinity, vs, es0, es1, es2, dataset_name])
print('full_dock', len(full_dock))
print('full_dock1', len(full_dock1))
print('full_dock_15', len(full_dock_15))
print('full_dock1_15', len(full_dock1_15))
print('exc_dock', len(exc_dock))
print('exc_dock1', len(exc_dock1))
print('exc_dock_15', len(exc_dock_15))
print('exc_dock1_15', len(exc_dock1_15))
with open("database/bindingdb_prank.pickle", 'wb') as f:
    pickle.dump(full_dock, f)
with open("database/bindingdb1_prank.pickle", 'wb') as f:
    pickle.dump(full_dock1, f)
with open("database/bindingdb_15_prank.pickle", 'wb') as f:
    pickle.dump(full_dock_15, f)
with open("database/bindingdb1_15_prank.pickle", 'wb') as f:
    pickle.dump(full_dock1_15, f)
with open("database/bindingdb_nopdbbind_prank.pickle", 'wb') as f:
    pickle.dump(exc_dock, f)
with open("database/bindingdb_nopdbbind1_prank.pickle", 'wb') as f:
    pickle.dump(exc_dock1, f)
with open("database/bindingdb_nopdbbind_15_prank.pickle", 'wb') as f:
    pickle.dump(exc_dock_15, f)
with open("database/bindingdb_nopdbbind1_15_prank.pickle", 'wb') as f:
    pickle.dump(exc_dock1_15, f)
