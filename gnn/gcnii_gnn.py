import sys,os
sys.path.append(os.getcwd())

import torch
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv, GINConv
from torch_geometric.utils import to_undirected
from torch_geometric.nn.dense.linear import Linear

class GCNii(torch.nn.Module):
    def __init__(self, in_channels=768, hidden_channel=128, alpha=0.1, theta=0.5, layer=8):
        super(GCNii, self).__init__()
        self.gcnii = GCN2Conv(in_channels, alpha, theta, layer)


    def forward(self, data):
        x, edge_index = data.x, to_undirected(data.edge_index)
        x = self.gcnii(x, x, edge_index)
        x = F.relu(x)
        x = scatter_mean(x, data.batch, dim=0)
        return x
    
