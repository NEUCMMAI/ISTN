'''
Author: Kang Yang
Date: 2023-03-09 14:31:04
LastEditors: Kang Yang
LastEditTime: 2023-03-13 13:01:27
FilePath: /project-b/models/iga/recbole_entrance.py
Description: 
Copyright (c) 2023 by Kang Yang, All Rights Reserved. 
'''
import multiprocessing as mp
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, LGConv
from tqdm import tqdm
import pickle

from models.iga.modules import GraphEncoder
from models.iga.utils import generate_subgraph, fix_pad

class IGA(GeneralRecommender):

    input_type = InputType.PAIRWISE
   
    def __init__(self, config, dataset, vaild_data = None, raw_dataset = None, regression=True, only_eval_target=False):
        super(IGA, self).__init__(config, raw_dataset)
        
        self.h = config['h']
        self.sample_ratio = config['sample_ratio']
        self.max_nodes_per_hop = config['max_nodes_per_hop']
        self.conv_layer = config['conv_layer']
        self.train_batch_size = config['train_batch_size']

        self.interaction_matrix = raw_dataset.inter_matrix(form='coo').astype(np.float32)  # csr
        
        self.__init_subgraph(pools=8)
        
        self.graph_encoder = GraphEncoder(
            input_dim = 2*(self.h+1) + 1, 
            gcn_output_dim = config['gcn_output_dim'], 
            dense_output_dim = config['dense_output_dim'], 
            accum = config['accum'], 
            device = self.device, 
            drop_prob = config['drop_prob'],
            gcn = eval(self.conv_layer['layer_name']),
            gcn_params = self.conv_layer['layer_pramas'],
            gcn_laysers_num = 2,
            act_dense =lambda x: x,
            bias=False
        )
        
        self.readout_attention = nn.MultiheadAttention(config['dense_output_dim'], 
                                                       config['read_out_head'])
        self.predict_layer = nn.Linear(config['dense_output_dim']*2, config['class_num'])
        self.activate_function = nn.LeakyReLU()
        
        self.loss = nn.CrossEntropyLoss()
        
        self.all_item_embeddings = None
        
        self.massive_users = False
        self.massive_items = False
        
        if (self.n_users/self.n_items > 200):
            self.massive_users = True
            
        if (self.n_items/self.n_users > 200):
            self.massive_items = True
        
    def __init_subgraph(self, pools = mp.cpu_count()):

        self.__generate_node_feature()
        
        self.A = self.interaction_matrix
        
        self.user_subgraph = dict()
        self.item_subgraph = dict()
        
        if self.n_users <= 2000:
            for user in tqdm(range(self.n_users)):
                data, node = generate_subgraph(user, self.A, self.h, self.sample_ratio, 
                                               self.max_nodes_per_hop, self.items_degree, self.users_degree)
                self.user_subgraph[user] = data
        else:
            torch.multiprocessing.set_sharing_strategy('file_system')
            mp.freeze_support()
            pool = mp.Pool(pools)
            results = pool.starmap_async(
                generate_subgraph, 
                [(user, self.A, self.h, self.sample_ratio, self.max_nodes_per_hop, 
                self.items_degree, self.users_degree) for user in range(self.n_users)],
                chunksize=10
            )
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready(): break
                remaining = results._number_left
                time.sleep(1)
            results = results.get()
            pool.close()
            pbar.close()
            pbar = tqdm(total=len(results))
            for result in results:
                data, node = result[0], result[1]
                self.user_subgraph[node] = data
                pbar.update(1)
            pbar.close()
            pool.join()

            with open('user_subgraph.tmp', 'wb') as f:
                pickle.dump(self.user_subgraph, f)

            del pool
            del results
            del self.user_subgraph

        if self.n_items <= 2000:  
            for item in tqdm(range(self.n_items)):
                data, node = generate_subgraph(item, self.A, self.h, self.sample_ratio, 
                                               self.max_nodes_per_hop, self.items_degree, self.users_degree, isItem = True)
                self.item_subgraph[item] = data
        else:
            torch.multiprocessing.set_sharing_strategy('file_system')
            mp.freeze_support()
            pool = mp.Pool(pools)
            results = pool.starmap_async(
                generate_subgraph, 
                [(item, self.A, self.h, self.sample_ratio, 
                  self.max_nodes_per_hop, self.items_degree, 
                  self.users_degree, True) for item in range(self.n_items)],
                chunksize=10
            )
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready(): break
                remaining = results._number_left
                time.sleep(1)
            results = results.get()
            pool.close()
            pbar.close()
            pbar = tqdm(total=len(results))
            for result in results:
                data, node = result[0], result[1]
                self.item_subgraph[node] = data
                pbar.update(1)
            pbar.close()
            pool.join()
            if self.n_users > 2000:
                with open('user_subgraph.tmp', 'rb') as f:
                    self.user_subgraph = pickle.load(f)

                # 删除文件
                try:
                    os.remove('user_subgraph.tmp')
                    print(f"{'user_subgraph.tmp'} 文件已删除")
                except OSError as e:
                    print(f"删除 user_subgraph.tmp 文件失败: {e.strerror}")

        
    def __generate_node_feature(self):
        graph_users = torch.tensor(self.interaction_matrix.row).long()
        graph_items = torch.tensor(self.interaction_matrix.col).long()
        
        node_id_users, counts_users= torch.unique(graph_users, return_counts = True)
        node_id_items, counts_items= torch.unique(graph_items, return_counts = True)
        
        max_user_degree = torch.max(counts_users).item()
        max_user_degree = torch.max(counts_items).item()
        max_degree = max([max_user_degree, max_user_degree])
        
        
        self.users_degree = torch.zeros(self.n_users, dtype = torch.float)
        self.users_degree[node_id_users] = counts_users / max_degree
        self.users_degree = self.users_degree.numpy()
        
        self.items_degree = torch.zeros(self.n_items, dtype = torch.float)
        self.items_degree[node_id_items] = counts_items / max_degree
        self.items_degree = self.items_degree.numpy()
        
    def graph_readout(self, embeddings, batch):
        idx, idx_num = torch.unique(batch, return_counts=True)
        # idx_num = torch.cat([torch.tensor([0]).to(self.device), torch.cumsum(idx_num, dim=0)])
        target_length = torch.max(idx_num)
        node_embeddings_list = torch.split(embeddings, idx_num.cpu().numpy().tolist(), dim=0)
        node_embeddings_list = [fix_pad(x, target_length) for x in node_embeddings_list]
        node_embeddings = torch.stack(node_embeddings_list)
        masks = torch.zeros(node_embeddings.shape[0], target_length, dtype=torch.float)
        for e_id, src_len in enumerate(idx_num):
            masks[e_id, src_len+1:] = 1
        masks = masks.bool().to(self.device)
        # For compatibility with lower versions of pytorch ↓
        query, key, value = [x.transpose(1, 0) for x in (node_embeddings, node_embeddings, node_embeddings)]
        
        attn_output, attn_output_weights = self.readout_attention(query, key, value, key_padding_mask = masks)
        attn_output = attn_output.transpose(1, 0)
        # For compatibility with lower versions of pytorch ↑
        node_embeddings = torch.sum(torch.mul(attn_output, ~masks.unsqueeze(-1)), dim=1)
        # node_embeddings = F.softmax(node_embeddings, dim=-1)
        node_embeddings = F.leaky_relu(node_embeddings)
        return node_embeddings
        
    def forward(self, user_batch, item_batch):
        user_embeddings = self.graph_readout(*self.graph_encoder(user_batch))
        item_embeddings = self.graph_readout(*self.graph_encoder(item_batch))
        
        scores = self.predict_layer(torch.cat([user_embeddings, item_embeddings], dim=-1))
        
        return scores
    
    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        target = torch.zeros(len(user) * 2, dtype=torch.long).to(self.device)
        target[:len(user)] = 1
        
        users = torch.cat((user, user))
        items = torch.cat((pos_item, neg_item))
        
        user_batch = list()
        for idx in range(len(users)):
            user = users[idx].item()
            user_data = self.user_subgraph[user]
            user_batch.append(user_data)
        user_batch = Batch.from_data_list(user_batch).to(self.device)
        
        item_batch = list()
        for idx in range(len(items)):
            item = items[idx].item()
            item_data = self.item_subgraph[item]
            item_batch.append(item_data)
        item_batch = Batch.from_data_list(item_batch).to(self.device)
        
        predict = self.activate_function(self.forward(user_batch, item_batch))

        loss = self.loss(predict, target)
        
        self.logger.debug(predict)
        self.logger.debug('loss: {:f}'.format(loss.item()))
        
        return loss

    
    def predict(self, interaction):
        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]
        
        user_batch = list()
        for idx in range(len(users)):
            user = users[idx].item()
            user_data = self.user_subgraph[user]
            user_batch.append(user_data)
        user_batch = Batch.from_data_list(user_batch).to(self.device)
        
        item_batch = list()
        for idx in range(len(items)):
            item = items[idx].item()
            item_data = self.item_subgraph[item]
            item_batch.append(item_data)
        item_batch = Batch.from_data_list(item_batch).to(self.device)
        
        predict = self.activate_function(self.forward(user_batch, item_batch))
        
        score = predict[:, 1]
        
        return score
    
    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        
        if self.all_item_embeddings is None : # all item is too big
            if self.massive_users is False: #too many 1-hop nodes
                item_batch = list()
                for idx in range(self.n_items):
                    item_data = self.item_subgraph[idx]
                    item_batch.append(item_data)
                item_batch = Batch.from_data_list(item_batch).to(self.device)
                self.all_item_embeddings = self.graph_readout(*self.graph_encoder(item_batch))
                self.all_item_embeddings = self.all_item_embeddings.repeat(len(users),1)
            else:
                item_embeddings_list = list()
                for idx in range(self.n_items):
                    item_batch = list()
                    item_data = self.item_subgraph[idx]
                    item_batch.append(item_data)
                    item_batch = Batch.from_data_list(item_batch).to(self.device)
                    item_embeddings_list.append(self.graph_readout(*self.graph_encoder(item_batch))) 
                self.all_item_embeddings = torch.cat(item_embeddings_list, dim=0)
                self.all_item_embeddings = self.all_item_embeddings.repeat(len(users),1)
        

        user_batch = list()
        for idx in range(len(users)):
            user = users[idx].item()
            user_data = self.user_subgraph[user]
            user_batch.append(user_data)
        user_batch = Batch.from_data_list(user_batch).to(self.device)
        user_embeddings = self.graph_readout(*self.graph_encoder(user_batch))
        user_embeddings = torch.stack(torch.split(user_embeddings.repeat(self.n_items,1), len(users), dim=0), dim = 1).view(len(users)*self.n_items ,-1)
        
        
        scores = self.activate_function(self.predict_layer(torch.cat([user_embeddings, self.all_item_embeddings], dim=-1)))
        
        return scores[:, 1]