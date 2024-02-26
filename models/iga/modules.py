'''
Author: Kang Yang
Date: 2023-03-09 14:31:42
LastEditors: Kang Yang
LastEditTime: 2023-03-09 14:35:22
FilePath: /project-b/models/iga/modules.py
Description: 
Copyright (c) 2023 by Kang Yang, All Rights Reserved. 
'''
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, LGConv

class GraphEncoder(nn.Module):
    def __init__(
        self, 
        input_dim, 
        gcn_output_dim, 
        dense_output_dim, 
        accum, 
        device, 
        drop_prob,
        gcn = GATConv,
        gcn_params = None,
        gcn_laysers_num = 2,
        act_dense =lambda x: x,
        bias=False
        ) -> None:
        super(GraphEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.gcn_output_dim = gcn_output_dim
        self.dense_output_dim = dense_output_dim
        self.accum = accum
        
        self.device = device
        self.dropout_prob = drop_prob
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.dense_activate = act_dense
        self.activate = nn.ReLU()
        self.bias = bias
        self.gcn = gcn
        self.gcn_laysers_num = gcn_laysers_num
        self.gcn_params = gcn_params
        
        
        # init gcn layers
        self.gcn_list = list()
        for idx in range(self.gcn_laysers_num):
            if idx == 0:
                if self.gcn == GATConv:
                    self.gcn_list.append(self.gcn(self.input_dim, self.gcn_output_dim, heads=self.gcn_params['head']))
                elif self.gcn == LGConv:
                    self.gcn_list.append(self.gcn())
            else:
                if self.gcn == GATConv:
                    self.gcn_list.append(self.gcn(self.gcn_output_dim * self.gcn_params['head'], self.gcn_output_dim, heads=self.gcn_params['head']))
                elif self.gcn == LGConv:
                    self.gcn_list.append(self.gcn())
        self.gcn_list = nn.ModuleList(self.gcn_list)
        if self.accum == 'sum':
            if self.gcn == GATConv:
                self.dense_layer = nn.Linear(self.gcn_output_dim * self.gcn_params['head'], self.dense_output_dim, bias=self.bias)
            elif self.gcn == LGConv:
                self.dense_layer = nn.Linear(self.input_dim, self.dense_output_dim, bias=self.bias)
        else:
            if self.gcn == GATConv:    
                self.dense_layer = nn.Linear(self.gcn_output_dim * self.gcn_laysers_num * self.gcn_params['head'], self.dense_output_dim, bias=self.bias)
            elif self.gcn == LGConv:
                self.dense_layer = nn.Linear(self.input_dim * self.gcn_laysers_num, self.dense_output_dim, bias=self.bias)
        
        # init weight
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch

        embeddings = []
        if self.accum == 'sum':
            for layer in self.gcn_list:
                x = layer(x, edge_index)
                embeddings.append(x)
            
            embeddings = torch.stack(embeddings)    
            embeddings = torch.sum(embeddings, dim=0)
            
        else:
            for layer in self.gcn_list:
                x = layer(x, edge_index)
                embeddings.append(x)
            
            embeddings = torch.cat(embeddings, dim=-1)
            
        embeddings = self.dropout(embeddings)
        embeddings = self.dense_activate(self.dense_layer(embeddings))

        return embeddings, batch