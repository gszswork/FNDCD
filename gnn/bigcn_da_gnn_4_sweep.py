import sys,os
sys.path.append(os.getcwd())
import torch
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch.nn import Linear, ReLU, ModuleList, Softmax, CrossEntropyLoss, Parameter
import numpy as np
from torch_geometric.data import DataLoader
from tqdm import tqdm
from .models import create_model, LSM, CosineLSM
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import GCNConv
from .overloader import overload
import copy
from utils.helper import rand_prop, consis_loss


device =torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TDrumorGCN(torch.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, x_da, edge_index = data.x, data.x_da, data.edge_index
        x = torch.cat((x,x_da),1)
        x1=copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2=copy.copy(x)
        rootindex = data.rootindex
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = torch.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = torch.cat((x,root_extend), 1)
        x= scatter_mean(x, data.batch, dim=0)

        return x

class BUrumorGCN(torch.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, x_da, edge_index = data.x, data.x_da, data.BU_edge_index
        x = torch.cat((x,x_da),1)
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        rootindex = data.rootindex
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = torch.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = torch.cat((x,root_extend), 1)

        x= scatter_mean(x, data.batch, dim=0)
        return x

class BiGCN(torch.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(BiGCN, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        self.fc=torch.nn.Linear((out_feats+hid_feats)*2,4)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = torch.cat((BU_x,TD_x), 1)
        #x=self.fc(x)
        #x = F.log_softmax(x, dim=1)
        return x


class BiGCNNet(torch.nn.Module):
    def __init__(self, in_channels,  num_classes=2, n_train_data=None, args=None, sweep_config=None):
        # Previously, I want to define an out_channels. But this can be done by args.channel.
        # in_channels=5000, args.channels=64
        super().__init__()

        self.num_classes = num_classes
        self.prior_ratio = sweep_config.prior_ratio
        self.n_train_data = n_train_data
        # for grand
        self.dropnode = args.dropnode
        self.temp = args.tem
        self.K = args.sample
        self.order = args.order
        self.lam = args.lam
        self.grand = args.grand
        self.sweep_config = sweep_config

        self.gnn_model = BiGCN(in_feats=in_channels, hid_feats=args.channels, out_feats=args.channels)
        # self.gnn_model = create_model(args.backbone, in_channels, args.channels, args.num_unit, args.dropout, args.dropedge, args.bn)
        # TODO: If the model created contains a MLP output layer?  -NO
        self.structure_model = LSM(in_channels, args.channels, args.dropout, sweep_config.neg_ratio)
        self.out_mlp = torch.nn.Sequential(
            Linear((args.channels + args.channels)*2, 2 * args.channels),
            ReLU(),
            Linear(2 * args.channels, num_classes)
        )
        if n_train_data:
            self.e_logits = Parameter(torch.ones([n_train_data, 2], dtype=torch.float))

        self.CELoss = torch.nn.CrossEntropyLoss(reduction='none')
        self.KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean')
        self.Softmax = torch.nn.Softmax(dim=-1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)

    def get_graph_rep(self, graph):
        graph_rep = self.gnn_model(graph)
        return graph_rep
    
    def get_pred(self, graph_rep):
        pred = self.out_mlp(graph_rep)
        return pred 
    
    def forward(self, graph): 
        if self.grand:
            graph.x = rand_prop(graph.x, graph.edge_index, self.order, self.dropnode, self.training)
        graph_rep = self.get_graph_rep(graph)
        # Keep the structure of BiGCN, the prediction labels are returned. 
        # graph_rep = global_max_pool(node_rep, batch)
        pred = self.out_mlp(graph_rep)
        return pred

    def infer_e_gx(self, x, edge_index, batch):
        """
        Infer the environment variable based
        on the structure estimation model (for testing OOD data)
        """
        graph_neglogprob = self.structure_model.get_reg_loss(x, edge_index, batch)
        graph_prob = torch.exp(-graph_neglogprob)
        e_inferred = graph_prob / (graph_prob + 0.5)
        return e_inferred

    def infer_e_gxy(self, x, edge_index, edge_attr, batch, y):
        """
        Infer the environment variable based
        on the structure estimation and classification model
        (for training OOD data, i.e. the outliers)
        """
        graph_neglogprob = self.structure_model.get_reg_loss(x, edge_index, batch)
        graph_prob = torch.exp(-graph_neglogprob)

        y_pred = self.forward(x, edge_index, edge_attr, batch)
        y_neglogprob = self.CELoss(y_pred, y)
        y_prob = torch.exp(-y_neglogprob)

        e_in = graph_prob * y_prob
        e_out = (1 / self.num_classes) * 1 / 2
        e_inferred = e_in / (e_in + e_out)
        return e_inferred

    def get_kl_loss(self, e_logprob):
        e_prior = torch.tensor([[1 - self.prior_ratio, self.prior_ratio]], dtype=torch.float, \
                               device=e_logprob.device).expand(e_logprob.size(0), -1)
        kl_loss = self.KLDivLoss(e_logprob, e_prior)
        return kl_loss

    def get_pred_loss(self,graph):
        pred = self.forward(graph)
        loss = torch.mean(self.CELoss(pred, graph.y))
        return loss

    #def get_pred_loss(self, x, edge_index, edge_attr, batch, y):
    #    graph_rep = self.get_graph_rep(x, edge_index, edge_attr, batch)
    #    pred = self.get_pred(graph_rep)
    #    loss = torch.mean(self.CELoss(pred, y))
    #    return loss

    def get_grand_pred_loss(self, x, edge_index, edge_attr, batch, y):
        output_list = []
        for k in range(self.K):
            output_list.append(torch.log_softmax(self(x, edge_index, edge_attr, batch), dim=-1))
        loss_train = 0.
        for k in range(self.K):
            loss_train += F.nll_loss(output_list[k], y)
        loss_train = loss_train / self.K
        loss_consis = consis_loss(output_list, self.temp)
        return loss_train + self.lam * loss_consis


    def get_graphde_a_loss(self, graph):
        x, x_da = graph.x, graph.x_da
        x = torch.cat((x, x_da), 1) # concatenate the x_feature and GPT-augmented feature. 
        graph_neglogprob = self.structure_model.get_reg_loss(x, graph.edge_index, graph.batch)
        graph_prob = torch.exp(-graph_neglogprob)

        y_pred = self.forward(graph)
        #y_pred = self.forward(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        y_neglogprob = self.CELoss(y_pred, graph.y)
        y_prob = torch.exp(-y_neglogprob)

        e_in = graph_prob * y_prob
        e_out = (1 / self.num_classes) * 1 / 2 
        # E_out = p(y|X,A,e=0) * p(A|X, e=0)   formula (10). 
        # 
        e_inferred = e_in / (e_in + e_out)

        logprob = torch.unsqueeze(-graph_neglogprob - y_neglogprob, dim=1)
        log_uniform = torch.tensor([[e_out]], dtype=torch.float, device=logprob.device) \
            .log().expand(logprob.size(0), -1)
        divider = torch.logsumexp(torch.cat([logprob, log_uniform], dim=1), 1, True)
        e_in_log = logprob - divider
        e_out_log = log_uniform - divider
        e_inferred_logprob = torch.cat([e_in_log, e_out_log], dim=1)

        # calculate loss
        kl_loss = self.get_kl_loss(e_inferred_logprob)
        inlier_pred_loss = torch.mean(e_inferred * y_neglogprob)
        inlier_reg_loss = torch.mean(e_inferred * graph_neglogprob)

        # the outlier prob is assumed to be uniform
        # The outlier prediction is to predict, if a test sample is OOD sample. Not classify it into T, F, N, U. 
        uni_logprob_pred = torch.full((len(graph.y),), 1 / self.num_classes, device=graph.y.device).log()
        outlier_pred_loss = torch.mean((1 - e_inferred) * -uni_logprob_pred)
        uni_logprob_reg = torch.full((len(graph.y),), 1 / 2, device=graph.x.device).log()
        outlier_reg_loss = torch.mean((1 - e_inferred) * -uni_logprob_reg)

        inlier_loss = inlier_pred_loss + inlier_reg_loss
        outlier_loss = outlier_pred_loss + outlier_reg_loss
        return inlier_loss + outlier_loss + kl_loss

 