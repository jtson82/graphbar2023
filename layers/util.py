import pickle
import torch
from torch_geometric.data import Data
import numpy as np

def load_data(file_name, feature_type=1):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    atomic = {5:0, 6:1, 7:2, 8:3, 15:4, 16:5, 34:6, 9:7, 17:7, 35:7, 35:7, 53:7, 85:7}
    data_list = []
    for x in data:
        feat_list = []
        coord_list = []
        if feature_type == 0:
            for y in x[2]:
                feat = np.zeros(18)
                if y[0] in atomic.keys():
                    feat[atomic[y[0]]] = 1
                else:
                    feat[8] = 1
                feat[9:18] = y[2:11]
                feat_list.append(feat)
                coord = y[1]
                coord_list.append(coord)
        else: # feature_type == 1
            for y in x[2]:
                feat = np.zeros(27)
                if y[6] == 0:
                    if y[0] in atomic.keys():
                        feat[atomic[y[0]]] = 1
                    else:
                        feat[8] = 1
                else:
                    if y[0] in atomic.keys():
                        feat[atomic[y[0]]+9] = 1
                    else:
                        feat[17] = 1
                feat[18:27] = y[2:11]
                feat_list.append(feat)
                coord = y[1]
                coord_list.append(coord)
        
        pdbid = x[0]
        node = torch.tensor(np.array(feat_list), dtype=torch.float)
        label = torch.tensor(x[1], dtype=torch.float)
        coord = torch.tensor(np.array(coord_list), dtype=torch.float)
        edge0 = torch.tensor(x[3], dtype=torch.long).t()
        edge1 = torch.tensor(x[4], dtype=torch.long).t()
        edge2 = torch.tensor(x[5], dtype=torch.long).t()
        edge_index = (edge0, edge1, edge2)
        data = Data(pdbid = pdbid, x=node, y=label, coord=coord, edge_index=edge_index)
        data_list.append(data)
    return data_list

