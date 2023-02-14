import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
from collections import OrderedDict
from pathlib import Path
import scipy.sparse as sp
from .data import sparse_mx_to_torch_sparse_tensor, normalize_2d
from .model_init import weights_init_kaiming
from .gcn_base import GCNBase


class ADJLayer(nn.Module):
    def __init__(self, fea_dims, alpha):
        super().__init__()
        self.in_features = fea_dims[0]
        self.out_features = fea_dims[-1]
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(fea_dims[0], fea_dims[1])))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(fea_dims[1], 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.bnneck1 = nn.BatchNorm1d(fea_dims[0])
        self.bnneck1.bias.requires_grad_(False)
        self.bnneck1.apply(weights_init_kaiming)

        self.bnneck2 = nn.BatchNorm1d(fea_dims[1])
        self.bnneck2.bias.requires_grad_(False)
        self.bnneck2.apply(weights_init_kaiming)

        self.bnneck3 = nn.BatchNorm1d(1)
        self.bnneck3.bias.requires_grad_(False)
        self.bnneck3.apply(weights_init_kaiming)

    def forward(self, adj, adj_mean):
        N = adj.shape[0]
        adj = adj.view(N*N, -1)
        adj = self.bnneck1(adj)
        adj = torch.matmul(adj, self.W)
        adj = self.leakyrelu(self.bnneck2(adj))
        adj = torch.matmul(adj, self.a)
        
        e = self.leakyrelu(self.bnneck3(adj))
        e = e.view(N, N)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj_mean > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        return attention


class ADJAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, fea_dims, alpha):
        super().__init__()
        self.in_features = fea_dims[0]
        self.out_features = fea_dims[-1]
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(self.in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.adj_layer = ADJLayer([32, 16], self.alpha)

    def forward(self, h, adj, adj_mean):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        attention = self.adj_layer(adj, adj_mean)
        h_prime = torch.matmul(attention, Wh)
        return h_prime


class GATModel(GCNBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.nheads = 6
        fea_dims = [2048, 512]
        self.alpha = 0.2
        self.nhis = fea_dims[-1]

        self.attentions = nn.ModuleList([ADJAttentionLayer(fea_dims, alpha=self.alpha) for _ in range(self.nheads)])

        self.bnneck = nn.BatchNorm1d(self.nhis * self.nheads)
        self.bnneck.bias.requires_grad_(False)
        self.bnneck.apply(weights_init_kaiming)

    def forward(self, input):
        x, adj, adj_mean = input
        x = torch.cat([att(x, adj, adj_mean) for att in self.attentions], dim=1)
        x = self.bnneck(x)
        x = F.elu(x)
        x = self.classifier(x, adj, adj_mean)
        return x
    
    def inference(self, input):
        x, adj, adj_mean = input
        x = torch.cat([att(x, adj, adj_mean) for att in self.attentions], dim=1)
        x = self.bnneck(x)
        x = F.elu(x)
        return x
    
    def init_classifier(self, nclass):
        self.classifier = ADJAttentionLayer([self.nhis * self.nheads, nclass], alpha=self.alpha)