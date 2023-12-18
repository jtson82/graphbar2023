import torch
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, global_add_pool
import torch.nn.functional as F
from egnn_pytorch import EGNN_Sparse

class GCN_model(torch.nn.Module):
    def __init__(self, input_dim, num_adjs=2, num_layers=3, fa_layer=False, bias=False):
        super().__init__()
        self.num_adjs = num_adjs
        self.lin0 = torch.nn.Linear(input_dim, 128)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_adjs):
            self.conv = torch.nn.ModuleList()
            for num in range(num_layers):
                if num == num_layers-1:
                    self.conv.append(GCNConv(128, int(256/num_adjs), bias=bias))
                else:
                    self.conv.append(GCNConv(128, 128, bias=bias))
            self.convs.append(self.conv)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 1)
        self.fa_layer = fa_layer

    def forward(self, data):
        node, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.lin0(node).relu()
        xi = [x] * self.num_adjs 
        for adj in range(self.num_adjs):
            for i, conv in enumerate(self.convs[adj]):
                if (i == self.num_adjs) and (self.fa_layer == True):
                    xi[adj] = conv(xi[adj], edge_index[self.num_adjs]).relu()
                else:
                    xi[adj] = conv(xi[adj], edge_index[adj]).relu()
            xi[adj] = global_add_pool(xi[adj], batch)
        x = torch.cat(xi, 1)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


class GAT_model(torch.nn.Module):
    def __init__(self, input_dim, num_adjs=2, num_layers=3, fa_layer=False, bias=False):
        super().__init__()
        self.num_adjs = num_adjs
        self.lin0 = torch.nn.Linear(input_dim, 128)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_adjs):
            self.conv = torch.nn.ModuleList()
            for num in range(num_layers):
                if num == num_layers-1:
                    self.conv.append(GATConv(128, int(256/num_adjs), bias=bias))
                else:
                    self.conv.append(GATConv(128, 128, bias=bias))
            self.convs.append(self.conv)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 1)
        self.fa_layer = fa_layer

    def forward(self, data):
        node, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.lin0(node).relu()
        xi = [x] * self.num_adjs 
        for adj in range(self.num_adjs):
            for i, conv in enumerate(self.convs[adj]):
                if (i == self.num_adjs) and (self.fa_layer == True):
                    xi[adj] = conv(xi[adj], edge_index[self.num_adjs]).relu()
                else:
                    xi[adj] = conv(xi[adj], edge_index[adj]).relu()
            xi[adj] = global_add_pool(xi[adj], batch)
        x = torch.cat(xi, 1)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

class TransConv_model(torch.nn.Module):
    def __init__(self, input_dim, num_adjs=2, num_layers=3, fa_layer=False, bias=False):
        super().__init__()
        self.num_adjs = num_adjs
        self.lin0 = torch.nn.Linear(input_dim, 128)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_adjs):
            self.conv = torch.nn.ModuleList()
            for num in range(num_layers):
                if num == num_layers-1:
                    self.conv.append(TransformerConv(128, int(256/num_adjs), bias=bias))
                else:
                    self.conv.append(TransformerConv(128, 128, bias=bias))
            self.convs.append(self.conv)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 1)
        self.fa_layer = fa_layer

    def forward(self, data):
        node, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.lin0(node).relu()
        xi = [x] * self.num_adjs 
        for adj in range(self.num_adjs):
            for i, conv in enumerate(self.convs[adj]):
                if (i == self.num_adjs) and (self.fa_layer == True):
                    xi[adj] = conv(xi[adj], edge_index[self.num_adjs]).relu()
                else:
                    xi[adj] = conv(xi[adj], edge_index[adj]).relu()
            xi[adj] = global_add_pool(xi[adj], batch)
        x = torch.cat(xi, 1)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

class egnn_model(torch.nn.Module):
    def __init__(self, input_dim, num_adjs=2, num_layers=3, fa_layer=False, bias=False):
        super().__init__()
        self.num_adjs = num_adjs
        self.lin0 = torch.nn.Linear(input_dim, 128)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_adjs):
            self.conv = torch.nn.ModuleList()
            for num in range(num_layers):
                self.conv.append(EGNN_Sparse(128, 128, update_coors = False))
            self.convs.append(self.conv)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 1)
        self.fa_layer = fa_layer

    def forward(self, data):
        node, coords, edge_index, batch = data.x, data.coord, data.edge_index, data.batch
        x = self.lin0(node).relu()
        x = torch.cat([coords, x], dim=-1)
        xi = [x] * self.num_adjs 
        for adj in range(self.num_adjs):
            for i, conv in enumerate(self.convs[adj]):
                if (i == self.num_adjs) and (self.fa_layer == True):
                    xi[adj] = conv(xi[adj], edge_index[self.num_adjs]).relu()
                else:
                    xi[adj] = conv(xi[adj], edge_index[adj]).relu()
            xi[adj] = global_add_pool(xi[adj], batch)
            xi[adj] = xi[:, 3:]
        x = torch.cat(xi, 1)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
