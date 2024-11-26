# ######################################
# Causal Debiasing Framework (plugging-in)
# ######################################

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



class Causal_NN(torch.nn.Module):
    def __init__(self, gnn_model, args=None):
        """
        in_channels: input feature dim, 
        num_classes: classification dim, 
        gnn_model: The graph encoder, output dimension should be aligned with args.channel. 
        """
        super().__init__()

        self.num_classes = args.out_channels
        self.prior_ratio = args.prior_ratio

        # for grand
        self.dropnode = args.dropnode
        self.temp = args.tem
        self.K = args.sample
        self.order = args.order
        self.lam = args.lam
        self.grand = args.grand

        self.gnn_model = gnn_model
        self.structure_model = LSM(args.in_channels, args.channels, args.dropout, args.neg_ratio)
        self.out_mlp = torch.nn.Sequential(
            Linear(args.channels, 2 * args.channels),
            ReLU(),
            Linear(2 * args.channels, self.num_classes)
        )
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
   

    def get_graphde_a_loss1(self, graph):
        graph_neglogprob = self.structure_model.get_reg_loss(graph.x, graph.edge_index, graph.batch)
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

    def get_graphde_a_loss(self, x, edge_index, edge_attr, batch, y):
        graph_neglogprob = self.structure_model.get_reg_loss(x, edge_index, batch)
        graph_prob = torch.exp(-graph_neglogprob)

        y_pred = self.forward(x, edge_index, edge_attr, batch)
        y_neglogprob = self.CELoss(y_pred, y)
        y_prob = torch.exp(-y_neglogprob)

        e_in = graph_prob * y_prob
        e_out = (1 / self.num_classes) * 1 / 2
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
        uni_logprob_pred = torch.full((len(y),), 1 / self.num_classes, device=y.device).log()
        outlier_pred_loss = torch.mean((1 - e_inferred) * -uni_logprob_pred)
        uni_logprob_reg = torch.full((len(y),), 1 / 2, device=x.device).log()
        outlier_reg_loss = torch.mean((1 - e_inferred) * -uni_logprob_reg)

        inlier_loss = inlier_pred_loss + inlier_reg_loss
        outlier_loss = outlier_pred_loss + outlier_reg_loss

        return inlier_loss + outlier_loss + kl_loss

