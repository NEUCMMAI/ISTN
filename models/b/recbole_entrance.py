import multiprocessing as mp
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, LGConv
from tqdm import tqdm

from models.b.modules import DIFTransformerEncoder, GraphEncoder
from models.b.utils import sequence_mask, generate_subgraph, fix_pad


class B(SequentialRecommender):

    def __init__(self, config, dataset, vaild_data = None, raw_dataset = None, regression=True, only_eval_target=False):
        super(B, self).__init__(config, raw_dataset)

        self.h = config['h']
        self.sample_ratio = config['sample_ratio']
        self.max_nodes_per_hop = config['max_nodes_per_hop']
        self.conv_layer = config['conv_layer']
        self.train_batch_size = config['train_batch_size']
        self.dense_output_dim= config['dense_output_dim']

        self.n_users = raw_dataset.num(self.USER_ID)

        self.interaction_matrix = raw_dataset.inter_matrix(form='coo').astype(np.float32)  # csr
        
        self.__init_subgraph(pools=8)
        self.__init_item_subgraph()


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
        
        self.position_embedding = nn.Embedding(self.max_seq_length, config['dense_output_dim'])

        self.trm_n_layers = config['trm_n_layers']
        self.trm_n_heads = config['trm_n_heads']
        self.trm_hidden_size = config['dense_output_dim']  # same as embedding_size
        self.trm_inner_size = config['trm_inner_size']  # the dimensionality in feed-forward layer
        self.trm_attribute_hidden_size = config['trm_attribute_hidden_size']
        self.trm_hidden_dropout_prob = config['trm_hidden_dropout_prob']
        self.trm_attn_dropout_prob = config['trm_attn_dropout_prob']
        self.trm_hidden_act = config['trm_hidden_act']
        self.trm_layer_norm_eps = config['trm_layer_norm_eps']
        self.trm_selected_features = config['trm_selected_features']
        self.trm_fusion_type = config['trm_fusion_type']

        self.trm_encoder = DIFTransformerEncoder(
            n_layers=self.trm_n_layers,
            n_heads=self.trm_n_heads,
            hidden_size=self.trm_hidden_size,
            attribute_hidden_size=self.trm_attribute_hidden_size,
            feat_num=len(self.trm_selected_features),
            inner_size=self.trm_inner_size,
            hidden_dropout_prob=self.trm_hidden_dropout_prob,
            attn_dropout_prob=self.trm_attn_dropout_prob,
            hidden_act=self.trm_hidden_act,
            layer_norm_eps=self.trm_layer_norm_eps,
            fusion_type=self.trm_fusion_type,
            max_len=self.max_seq_length
        )

        self.LayerNorm = nn.LayerNorm(config['dense_output_dim'], eps=self.trm_layer_norm_eps)
        self.dropout = nn.Dropout(self.trm_hidden_dropout_prob)

        self.loss_type = config['loss_type']

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
            self.attribute_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
    def __init_item_subgraph(self):
        item_batch = list()
        for idx in range(self.n_items):
        # for idx in range(200):
            item_data = self.item_subgraph[idx]
            item_batch.append(item_data)
        self.item_batch = Batch.from_data_list(item_batch).to(self.device)

    def __init_subgraph(self, pools = mp.cpu_count()):

        self.__generate_node_feature()
        
        self.A = self.interaction_matrix
        
        # self.user_subgraph = dict()
        self.item_subgraph = dict()
        
        # if self.n_users <= 2000:
        #     for user in tqdm(range(self.n_users)):
        #         data, node = generate_subgraph(user, self.A, self.h, self.sample_ratio, 
        #                                        self.max_nodes_per_hop, self.items_degree, self.users_degree)
        #         self.user_subgraph[user] = data
        # else:
        #     torch.multiprocessing.set_sharing_strategy('file_system')
        #     mp.freeze_support()
        #     pool = mp.Pool(pools)
        #     results = pool.starmap_async(
        #         generate_subgraph, 
        #         [(user, self.A, self.h, self.sample_ratio, self.max_nodes_per_hop, 
        #         self.items_degree, self.users_degree) for user in range(self.n_users)],
        #         chunksize=10
        #     )
        #     remaining = results._number_left
        #     pbar = tqdm(total=remaining)
        #     while True:
        #         pbar.update(remaining - results._number_left)
        #         if results.ready(): break
        #         remaining = results._number_left
        #         time.sleep(1)
        #     results = results.get()
        #     pool.close()
        #     pbar.close()
        #     pbar = tqdm(total=len(results))
        #     for result in results:
        #         data, node = result[0], result[1]
        #         self.user_subgraph[node] = data
        #         pbar.update(1)
        #     pbar.close()
        #     pool.join()

        #     with open('user_subgraph.tmp', 'wb') as f:
        #         pickle.dump(self.user_subgraph, f)

        #     del pool
        #     del results
        #     del self.user_subgraph

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
            # if self.n_users > 2000:
            #     with open('user_subgraph.tmp', 'rb') as f:
            #         self.user_subgraph = pickle.load(f)

            #     # 删除文件
            #     try:
            #         os.remove('user_subgraph.tmp')
            #         print(f"{'user_subgraph.tmp'} 文件已删除")
            #     except OSError as e:
            #         print(f"删除 user_subgraph.tmp 文件失败: {e.strerror}")

        
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
    # @profile
    def graph_readout(self, embeddings, batch):
        idx, idx_num = torch.unique(batch, return_counts=True)
        # idx_num = torch.cat([torch.tensor([0]).to(self.device), torch.cumsum(idx_num, dim=0)])
        node_embeddings_list = torch.split(embeddings, idx_num.cpu().numpy().tolist(), dim=0)

        target_length = torch.max(idx_num)
        node_embeddings_list = [fix_pad(x, target_length) for x in node_embeddings_list]
        node_embeddings = torch.stack(node_embeddings_list)

        
        # node_embeddings = torch.nn.utils.rnn.pad_sequence(node_embeddings_list, batch_first=True, padding_value=0.0)

        masks = torch.zeros(node_embeddings.shape[0], target_length, dtype=torch.float)
        for e_id, src_len in enumerate(idx_num):
            masks[e_id, src_len+1:] = 1
        masks = masks.bool().to(self.device)

        # masks = sequence_mask(idx_num)

        # For compatibility with lower versions of pytorch ↓
        query, key, value = [x.transpose(1, 0) for x in (node_embeddings, node_embeddings, node_embeddings)]
        
        attn_output, attn_output_weights = self.readout_attention(query, key, value, key_padding_mask = masks)
        attn_output = attn_output.transpose(1, 0)
        # For compatibility with lower versions of pytorch ↑
        node_embeddings = torch.sum(torch.mul(attn_output, ~masks.unsqueeze(-1)), dim=1)
        # node_embeddings = F.softmax(node_embeddings, dim=-1)
        node_embeddings = F.leaky_relu(node_embeddings)
        return node_embeddings
    
    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    def forward(self, item_seq, item_seq_len):
        item_batch = list()
        for idx in range(item_seq.shape[0]):
            for jdx in range(item_seq.shape[1]):
                item_data = self.item_subgraph[item_seq[idx, jdx].item()]
                item_batch.append(item_data)
        item_batch = Batch.from_data_list(item_batch).to(self.device).clone().detach()
        item_embeddings = self.graph_readout(*self.graph_encoder(item_batch))
        item_embeddings = item_embeddings.reshape((-1, self.max_seq_length, self.dense_output_dim))
        
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        position_embedding = torch.zeros_like(position_embedding) # ab

        item_embeddings = item_embeddings
        input_emb = self.LayerNorm(item_embeddings)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq).clone().detach()
        feature_emb = torch.tensor([0])
        trm_output = self.trm_encoder(input_emb,feature_emb,position_embedding, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        # print(seq_output)
        # exit()
        return seq_output
    # @profile
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]

            pos_item_batch = list()
            for idx in range(len(pos_items)):
                item = pos_items[idx].item()
                item_data = self.item_subgraph[item]
                pos_item_batch.append(item_data)
            pos_item_batch = Batch.from_data_list(pos_item_batch).to(self.device)
            pos_items_emb = self.graph_readout(*self.graph_encoder(pos_item_batch))

            neg_item_batch = list()
            for idx in range(len(neg_items)):
                item = pos_items[idx].item()
                item_data = self.item_subgraph[item]
                neg_item_batch.append(item_data)
            neg_item_batch = Batch.from_data_list(neg_item_batch).to(self.device)
            neg_items_emb = self.graph_readout(*self.graph_encoder(neg_item_batch))

            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            with torch.no_grad():
                test_item_emb = self.graph_readout(*self.graph_encoder(self.item_batch))
            # test_item_emb = self.graph_readout(*self.graph_encoder(self.item_batch))
            

            # test_item_emb = item_embeddings
            # test_item_emb = torch.rand((self.n_items, self.dense_output_dim), dtype=torch.float32).to(self.device)


            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item = interaction[self.ITEM_ID]

        item_batch = list()
        for idx in range(test_item.shape[0]):
            item = test_item[idx].item()
            item_data = self.item_subgraph[item]
            item_batch.append(item_data)
        item_batch = Batch.from_data_list(item_batch).to(self.device)
        item_embeddings = self.graph_readout(*self.graph_encoder(item_batch))

        test_item_emb = item_embeddings
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        # item_batch = list()
        # for idx in range(self.n_items):
        #     item_data = self.item_subgraph[idx]
        #     item_batch.append(item_data)
        # item_batch = Batch.from_data_list(item_batch).to(self.device)
        item_embeddings = self.graph_readout(*self.graph_encoder(self.item_batch))

        test_items_emb = item_embeddings
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores