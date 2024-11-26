from gnn.bigcn_gnn import BiGCN 
from gnn.gat_gnn import GAT
from gnn.gin_gnn import GIN
from gnn.gcnii_gnn import GCNii

def init_gnn_model(args):
    if args.gnn_model == 'BiGCN':
        return BiGCN(args.in_channels, args.channels/4, args.channels)
    if args.gnn_model == 'GAT':
        return GAT(args.in_channels, args.channels, args.channels)
    if args.gnn_model == 'GIN':
        return GIN(args.in_channels, args.channels, args.channels)
    if args.gnn_model == 'GCNii':
        return GCNii(args.in_channels, args.alpha*4, args.theta, args.layer)
    else: 
        assert 'No valid gnn_model found. '
        