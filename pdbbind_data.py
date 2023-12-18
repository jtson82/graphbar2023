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


affinity_data = pd.read_csv("/home/json/database/pdbbind/v2020/affinity_data.csv", comment='#')

print(affinity_data['-logKd/Ki'].isnull().any())
with open("/home/json/projects/graphbar/core_pdbbind2013.ids", 'r') as f:
    core2013 = f.read().splitlines()
core2013 = set(core2013)

with open("/home/json/projects/graphbar/core_pdbbind2016.ids", 'r') as f:
    core_set = f.read().splitlines()
core_set = set(core_set)

with open("/home/json/database/pdbbind/v2020/refined_pdbbind2020.ids", 'r') as f:
    refined_set = f.read().splitlines()
refined_set = set(refined_set)

"""
with open("../docking/pdb_kinase.csv", 'r', encoding='utf-8-sig') as f:
    kinase_set = f.read().splitlines()
kinase_set = set(kinase_set)

with open("/home/json/projects/graphbar/docking_dict.pickle", 'rb') as f:
    docking_dict = pickle.load(f)
"""

general_set = set(affinity_data['pdbid'])

affinity_data.loc[np.in1d(affinity_data['pdbid'], list(general_set)), 'set'] = 'general'
affinity_data.loc[np.in1d(affinity_data['pdbid'], list(refined_set)), 'set'] = 'refined'
affinity_data.loc[np.in1d(affinity_data['pdbid'], list(core_set)), 'set'] = 'core'
print(affinity_data.groupby('set').apply(len).loc[['general', 'refined', 'core']])

dataset_path = {'general' : 'general-set-except-refined', 'refined': 'refined-set', 'core': 'refined-set'}
path = "/home/json/database/pdbbind/v2020/"

dock_num = 3
general_data = []
refined_data = []
core_data = []
core2013_data = []
for dataset_name, data in affinity_data.groupby('set'):
    for _, row in data.iterrows():
        ds_path = dataset_path[dataset_name]
        name = row['pdbid']
        if not os.path.isdir('%s/%s/%s' %(path, ds_path, name)):
            ds_path = 'general-set-except-refined'
        affinity = row['-logKd/Ki']
        ligand = next(pybel.readfile('mol2', "%s/%s/%s/%s_ligand.mol2" %(path, ds_path, name, name)))
        pocket = next(pybel.readfile('mol2', "%s/%s/%s/%s_pocket.mol2" %(path, ds_path, name, name)))
        vs = get_nodes(ligand, pocket)
        es0, es1, es2 = get_edges(vs)
        if name in core2013:
            core2013_data.append([name, affinity, vs, es0, es1, es2, dataset_name])
        elif dataset_name == "general":
            general_data.append([name, affinity, vs, es0, es1, es2, dataset_name])
        elif dataset_name == "refined":
            refined_data.append([name, affinity, vs, es0, es1, es2, dataset_name])
        else:
            core_data.append([name, affinity, vs, es0, es1, es2, dataset_name])
        if (name in core2013) and dataset_name == 'core':
            core_data.append([name, affinity, vs, es0, es1, es2, dataset_name])
    
with open("database/core2013_data.pickle", 'wb') as f:
    pickle.dump(core2013_data, f)
print('core2013', len(core2013_data))
with open("database/core_data.pickle", 'wb') as f:
    pickle.dump(core_data, f)
print('core', len(core_data))
with open("database/general_data.pickle", 'wb') as f:
    pickle.dump(general_data, f)
print('general', len(general_data))
with open("database/refined_data.pickle", 'wb') as f:
    pickle.dump(refined_data, f)
print('refined', len(refined_data))



dockings = ['docking_dict_rmsd3', 'docking_dict_rmsd4', 'docking_dict_rmsd5', 'docking_dict']

for docks in dockings:
    docking_list = []
    with open("%s.pickle" %docks, 'rb') as f:
        docking_dict = pickle.load(f)
    for dataset_name, data in affinity_data.groupby('set'):
        for _, row in data.iterrows():
            ds_path = dataset_path[dataset_name]
            name = row['pdbid']
            print(name)
            if not name in docking_dict.keys():
                continue
            if not os.path.isdir('%s/%s/%s' %(path, ds_path, name)):
                ds_path = 'general-set-except-refined'
            affinity = row['-logKd/Ki']
            pocket = next(pybel.readfile('mol2', "%s/%s/%s/%s_pocket.mol2" %(path, ds_path, name, name)))
            cnt = 0
            for dock in docking_dict[name]:
                print(dock)
                ligand = next(pybel.readfile('pdbqt', "/home/json/myprojects/defense/database/docking/%s/%s" %(name, dock)))
                vs = get_nodes(ligand, pocket)
                es0, es1, es2 = get_edges(vs)
                docking_list.append([name, affinity, vs, es0, es1, es2, 'docking'])
                cnt += 1
                if cnt >= dock_num:
                    break
    with open("database/%s.pickle" %docks, 'wb') as f:
        pickle.dump(docking_list, f)
    print(docks, len(docking_list))
