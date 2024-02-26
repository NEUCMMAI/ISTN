import multiprocessing as mp

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType
from torch_geometric.data import Batch
from torch_geometric.nn import RGCNConv

from models.igmc.modules import IGMCBase
from models.igmc.utils import (MyDynamicDataset, SparseColIndexer, SparseRowIndexer,
                    construct_pyg_graph, links2subgraphs,
                    subgraph_extraction_labeling)


class IGMC(GeneralRecommender):

    input_type = InputType.PAIRWISE
    
    def __init__(self, config, dataset, vaild_data, raw_dataset, regression=True, only_eval_target=False):
        super(IGMC, self).__init__(config, raw_dataset)
        
        self.RATING = config['RATING_FIELD']
        
        self.h = config['h']
        self.sample_ratio = config['sample_ratio']
        self.max_nodes_per_hop = config['max_nodes_per_hop']
        self.max_num = config['max_num']
        self.device = config['device']
        self.ARR = config['ARR']
        self.dynamic_parallel = config['dynamic_parallel']
        self.A = None
        self.only_eval_target = only_eval_target
        
        self.regression = regression
        
        self.interaction_matrix = dataset.dataset.inter_matrix(form='csr').astype(np.float32)  # csr
        
        self.train_dataset, self.train_graph = self.init_graph(raw_dataset, dataset, vaild_data)
        
        
        self.igmc = IGMCBase(
            self.train_dataset, 
            gconv = RGCNConv,
            latent_dim=[32, 32, 32, 32], 
            num_relations=config['num_relations'], 
            num_bases=4, 
            regression=False,  
            adj_dropout=config['adj_dropout'], 
            force_undirected=config['force_undirected'], 
            side_features=config['use_features'], 
            n_side_features=0, 
            multiply_by=config['multiply_by']
        )
        
        self.igmc.to(self.device).reset_parameters()
        self.loss_function = torch.nn.CrossEntropyLoss()
        
        
    def init_graph(self, raw_dataset, dataset, vaild_data):
        
        self.num_users, self.num_items = raw_dataset.user_num, raw_dataset.item_num
        
        self.A = self.interaction_matrix

        u_features = None
        v_features = None
        self.class_values = None
        
        self.Arow = SparseRowIndexer(self.A)
        self.Acol = SparseColIndexer(self.A.tocsc())
        g_dict = None
        
        self.dynamic_graph = False

        if(self.n_items * self.n_users < 1500**2):
            
            users = np.array([x for x in range(self.num_users)])
            items = np.array([x for x in range(self.num_items)])
            
            users = np.repeat(users, len(items))
            items = np.expand_dims(items, axis=0)
            items = np.repeat(items, len(users), axis=0)
            items = items.flatten()
                
            links = (users, items)
            labels = torch.zeros(len(users), dtype=torch.long)
        
            g_dict = links2subgraphs(self.Arow, self.Acol, links, labels, self.h, self.sample_ratio, self.max_nodes_per_hop,
                                                u_features, v_features, self.class_values, True, pools = 4)
        else:
            self.dynamic_graph = True
            
        zero_data = (torch.tensor([0]),torch.tensor([0]),torch.tensor([0]))
        
        dataset = {
                'num_node_features': self.dynamic_construct_graph(*zero_data)[(0,0)].num_node_features
        }
        return dataset, g_dict
    
    def dynamic_construct_graph(self, users ,items, labels, parallel=False):
        
        train_graph = dict()
        if not parallel:
            for idx in range(len(users)):
                    user = users[idx].item()
                    item = items[idx].item()
                    label = labels[idx].item()
                    graph_data = subgraph_extraction_labeling(
                            (user, item), self.Arow, self.Acol, self.h, self.sample_ratio, self.max_nodes_per_hop, None, 
                            None, self.class_values, label)
                    train_graph[(user, item)] = construct_pyg_graph(*graph_data[:-2])
                    
        else:
            links = (users.cpu(), items.cpu())
            train_graph = links2subgraphs(self.Arow, self.Acol, links, labels.cpu(), self.h, self.sample_ratio, self.max_nodes_per_hop,
                                                None, None, self.class_values, parallel=parallel, show_bar = False)
        
        return train_graph
        
        
    def forward(self, batch):
        return self.igmc(batch)
        
    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        users = torch.cat((user, user))
        items = torch.cat((pos_item, neg_item))
        
        ratings = torch.zeros(len(user) * 2, dtype=torch.long).to(self.device)
        ratings[:len(user)] = 1
        
        if self.dynamic_graph is True:
            self.train_graph = self.dynamic_construct_graph(users, items, ratings, self.dynamic_parallel)
        
        batch = list()
        for idx in range(len(users)):
            user = users[idx].item()
            item = items[idx].item()
            data = self.train_graph[(user, item)]
            batch.append(data)
        batch = Batch.from_data_list(batch).to(self.device)
        out = self.forward(batch)
        loss = self.loss_function(out, ratings)
        
        if self.ARR != 0:
            for gconv in self.igmc.convs:
                w = torch.matmul(
                    gconv.comp,
                    gconv.weight.view(gconv.num_bases, -1)
                ).view(gconv.num_relations, gconv.in_channels, gconv.out_channels)
                reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
                loss += self.ARR * reg_loss
        
        return loss
        
        
    def predict(self, interaction):
        
        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]
        ratings = torch.zeros(len(users), dtype=torch.long).to(self.device)
        
        if self.dynamic_graph is True:
            self.train_graph = self.dynamic_construct_graph(users, items, ratings, self.dynamic_parallel)
        
        batch = list()
        for idx in range(len(interaction)):
            user = interaction[self.USER_ID][idx].item()
            item = interaction[self.ITEM_ID][idx].item()
            data = self.train_graph[(user, item)]
            batch.append(data)
        batch = Batch.from_data_list(batch).to(self.device)
        out = self.forward(batch)
        return out[:, 1]
        
    def full_sort_predict(self, interaction):
        
        raw_users = interaction[self.USER_ID].cpu().numpy()
        items = np.array([x for x in range(self.num_items)])
        
        users = np.repeat(raw_users, self.num_items)
        items = np.expand_dims(items, axis=0)
        items = np.repeat(items, len(raw_users), axis=0)
        items = items.flatten()
        
        ratings = torch.zeros(len(users), dtype=torch.long).to(self.device)
        
        if self.dynamic_graph is True:
            self.train_graph = self.dynamic_construct_graph(users, items, ratings, self.dynamic_parallel)
        
        batch = list()
        for idx in range(len(users)):
            user = users[idx]
            item = items[idx]
            data = self.train_graph[(user, item)]
            batch.append(data)
        batch = Batch.from_data_list(batch).to(self.device)
        out = self.forward(batch)
        return out[:, 1]
