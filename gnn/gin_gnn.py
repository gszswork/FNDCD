import sys,os
sys.path.append(os.getcwd())

import torch
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv, GINConv
from torch_geometric.utils import to_undirected
from torch_geometric.nn.dense.linear import Linear

class GIN(torch.nn.Module):
    def __init__(self, in_channels=768, hidden_channels=128, out_channels=2):
        super(GIN, self).__init__()
        
        # Define the MLP for GINConv
        self.mlp = torch.nn.Sequential(
            Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
        )
        
        # Replace GCNII with GINConv
        self.gin = GINConv(self.mlp)
        
        # Final linear layer for classification
        self.linear = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, to_undirected(data.edge_index)
        
        # Apply GIN layer
        x = self.gin(x, edge_index)
        x = F.relu(x)
        x = scatter_mean(x, data.batch, dim=0)
        # Apply the final classification layer
        return x