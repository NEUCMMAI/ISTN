import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType

from models.retagnn.modules import GNN_SR_Net, RAGCNConv

class RetaGNN(SequentialRecommender):
    
    input_type = InputType.PAIRWISE
    
    def __init__(self, config, dataset, vaild_data = None, raw_dataset = None, regression=True, only_eval_target=False):
        super(RetaGNN, self).__init__(config, raw_dataset)
        
        self.L = config['L']
        self.S = config['S']
        self.short_term_window_num = config['short_term_window_num']
        self.FEATURE_FIELD = config["item_attribute"]

        self.dataset = dataset.dataset
        
        self.n_users, self.n_items = raw_dataset.user_num, raw_dataset.item_num
        
        self.interaction_matrix = dataset.dataset.inter_matrix(form='coo').astype(np.float32)
        # self.item_feat_categories = self.dataset.item_feat['categories']
        # self.item_categories = self.dataset.field2token_id['categories']
        self.item_feat_categories = self.dataset.item_feat[self.FEATURE_FIELD]
        self.item_categories = self.dataset.field2token_id[self.FEATURE_FIELD]

        self.u2v, self.v2u, self.v2vc = self.extract_inter_dict()
        
        # self.retagnn = GNN_SR_Net(config, self.n_items, self.n_users + self.n_items + len(self.item_categories), config['relation_num'], rgcn=RAGCNConv, device=self.device)
        self.retagnn = GNN_SR_Net(config, self.n_items, self.n_users + self.n_items, config['relation_num'], rgcn=RAGCNConv, device=self.device)

        short_term_window_size = int(self.L / self.short_term_window_num)
        self.short_term_window = [0] + [i + short_term_window_size for i in range(self.short_term_window_num-1)] + [-1]
   
    # @profile
    def complete_idx(self, idx_dict, complete_num, offset = 0):
                if(len(idx_dict)) == complete_num:
                    return idx_dict
                dict_idx = np.array(list(idx_dict.keys()))
                c_idx = np.arange(0, complete_num, 1) + offset
                lost_idx = np.setdiff1d(c_idx, dict_idx, assume_unique=True)
                for idx in lost_idx:
                    idx_dict[idx] = list()
                return idx_dict
     
    def extract_inter_dict(self):
        
        u2v = dict()
        v2u = dict()
        v2vc = dict()
        
        for u, v in zip(self.interaction_matrix.row, self.interaction_matrix.col):
            u = u + self.n_items
            if u not in u2v.keys():
                u2v[u] = list()
                u2v[u].append(v)
            else:
                u2v[u].append(v)
            if v not in v2u.keys():
                v2u[v] = list()
                v2u[v].append(u)
            else:
                v2u[v].append(u)
                    
        u2v = self.complete_idx(u2v, self.n_users, offset=self.n_items)
        v2u = self.complete_idx(v2u, self.n_items)

        zero_list = list()
        for v in range(self.n_items):
            if (len(self.item_feat_categories.shape) > 1):
                v2vc[v] = self.item_feat_categories[v,0] + self.n_users + self.n_items
            else:
                v2vc[v] = self.item_feat_categories[v] + self.n_users + self.n_items
                
        return u2v, v2u, v2vc     
        
    def extract_subgraph(self, user_no, seq_no, sub_seq_no=None, node2ids=None, hop=2):
        if sub_seq_no is not None:
            origin_seq_no = seq_no
            seq_no = sub_seq_no

        if node2ids is  None:
            node2ids = dict()
            
        edge_index, edge_type = list(), list()
        update_set, index, memory_set = list(), 0, list()
        
        for i in range(hop):
            if i == 0:
                #uv
                for j in range(user_no.shape[0]):
                    if user_no[j] not in node2ids:
                        node2ids[user_no[j]] = index
                        index +=1
                    user_node_ids = node2ids[user_no[j]]
                    for k in range(seq_no[j,:].shape[0]):
                        if seq_no[j,k] not in node2ids:
                            node2ids[seq_no[j,k]] = index
                            index +=1
                        item_node_ids = node2ids[seq_no[j,k]]
                        edge_index.append([user_node_ids,item_node_ids])
                        edge_type.append(0)
                        edge_index.append([item_node_ids,user_node_ids])
                        edge_type.append(1)

                        # #vc
                        # vc = self.v2vc[seq_no[j,k]]
                        # if vc not in node2ids:
                        #     node2ids[vc] = index
                        #     index +=1      
                        # vc_node_ids = node2ids[vc] 
                        # edge_index.append([item_node_ids,vc_node_ids])
                        # edge_type.append(2)
                        # edge_index.append([vc_node_ids,item_node_ids])
                        # edge_type.append(3)       
                        # # update       
                        # update_set.append(seq_no[j,k])
                        # #memory
                        # memory_set.append(user_no[j])

                update_set = list(set(update_set)) #v
                memory_set = set(memory_set) #u

                if sub_seq_no is not None:
                    for j in range(user_no.shape[0]):
                        user_node_ids = node2ids[user_no[j]]
                        for k in range(origin_seq_no[j,:].shape[0]):
                            if origin_seq_no[j,k] not in node2ids:
                                node2ids[origin_seq_no[j,k]] = index
                                index +=1
                            if origin_seq_no[j,k] not in set(update_set):
                                item_node_ids = node2ids[origin_seq_no[j,k]]
                                edge_index.append([user_node_ids,item_node_ids])
                                edge_type.append(0)
                                edge_index.append([item_node_ids,user_node_ids])
                                edge_type.append(1)

                                # #vc
                                # vc = self.v2vc[origin_seq_no[j,k]]
                                # if vc not in node2ids:
                                #     node2ids[vc] = index
                                #     index +=1      
                                # vc_node_ids = node2ids[vc] 
                                # edge_index.append([item_node_ids,vc_node_ids])
                                # edge_type.append(2)
                                # edge_index.append([vc_node_ids,item_node_ids])
                                # edge_type.append(3)      
                                
            elif i != 0 and i % 2 != 0:
                # vu
                new_update_set,new_memory_set = list(),list()
                for j in range(len(update_set)):
                    item_node_ids = node2ids[update_set[j]]
                    u_list = self.v2u[update_set[j]]
                    for k in range(len(u_list)):
                        if u_list[k] not in node2ids:
                            node2ids[u_list[k]] = index
                            index +=1
                        if u_list[k] not in memory_set:
                            user_node_ids = node2ids[u_list[k]]
                            edge_index.append([item_node_ids,user_node_ids])
                            edge_type.append(1)                       
                            edge_index.append([user_node_ids,item_node_ids])
                            edge_type.append(0)            
                            new_update_set.append(u_list[k])
                memory_set = set(update_set) #v
                update_set = new_update_set #u
            elif i != 0 and i % 2 == 0:
                #uv      
                for j in range(len(update_set)):
                    user_node_ids = node2ids[update_set[j]]
                    v_list = self.u2v[update_set[j]]
                    for k in range(len(v_list)):
                        if v_list[k] not in node2ids:
                            node2ids[v_list[k]] = index
                            index +=1
                        if v_list[k] not in memory_set:
                            item_node_ids = node2ids[v_list[k]]
                            edge_index.append([item_node_ids,user_node_ids])
                            edge_type.append(1)                       
                            edge_index.append([user_node_ids,item_node_ids])
                            edge_type.append(0)            
                            new_update_set.append(v_list[k])
                memory_set = set(update_set)  #u
                update_set = new_update_set   #v
        
        edge_index = torch.t(torch.tensor(edge_index).to(self.device))
        edge_type = torch.tensor(edge_type).to(self.device)
        node_no = torch.tensor(sorted(list(node2ids.values()))).to(self.device)
        # new_user_no,new_seq_no
        new_user_no,new_seq_no = list(),list()
        for i in range(user_no.shape[0]):
            new_user_no.append(user_no[i])
        for i in range(seq_no.shape[0]):
            new_seq_no.append([seq_no[i,j] for j in range(seq_no[i,:].shape[0])])
        new_user_no,new_seq_no = np.array(new_user_no),np.array(new_seq_no)

        return new_user_no,new_seq_no,edge_index,edge_type,node_no,node2ids   
        
    def id2node_id(self, input, node_dict):
        input_device = input.device
        input = input.cpu()
        input.map_(input, lambda x, *y:node_dict[x])
        return input.to(input_device)
    
    def forward(self):
        pass
        
        
    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID] + self.n_items
        item_seq = interaction[self.ITEM_SEQ]
        neg_items = interaction[self.NEG_ITEM_ID]
        pos_items = interaction[self.POS_ITEM_ID]
        batch_users, batch_sequences, edge_index, edge_type, node_no, node2ids = self.extract_subgraph(user.cpu().numpy(), item_seq.cpu().numpy())
        
        short_term_part = []
        for i in range(len(self.short_term_window)):
            if i != len(self.short_term_window)-1:
                sub_seq_no = batch_sequences[:,self.short_term_window[i]:self.short_term_window[i+1]]
                _,_,edge_index,edge_type,_,_ = self.extract_subgraph(batch_users, batch_sequences, sub_seq_no=sub_seq_no, node2ids=node2ids)
                short_term_part.append((edge_index,edge_type))
                
        items_to_predict = torch.cat((pos_items, neg_items)).view(-1, 1)
        user = torch.cat((user, user))
        item_seq = torch.cat((item_seq, item_seq))
        
        user = self.id2node_id(user, node2ids)
        item_seq = self.id2node_id(item_seq, node2ids)
        
        X_user_item = [user, item_seq, items_to_predict]
        X_graph_base = [edge_index, edge_type, node_no, short_term_part]
        pred_score, user_emb, item_embs_conv = self.retagnn(X_user_item, X_graph_base, for_pred=False)
        
        (targets_pred, negatives_pred) = torch.split(
                    pred_score, [pos_items.size(0), neg_items.size(0)], dim=0)
        # BPR loss
        loss = -torch.log(torch.sigmoid(targets_pred - negatives_pred) + 1e-8)
        loss = torch.mean(torch.sum(loss))
        
        # RAGCN losss(long term)
        reg_loss = 0
        for gconv in self.retagnn.conv_modulelist:
            w = torch.matmul(gconv.att_r, gconv.basis.view(gconv.num_bases, -1)).view(gconv.num_relations, gconv.in_channels, gconv.out_channels)
            reg_loss += torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
        reg_loss = reg_loss/len(self.retagnn.conv_modulelist)
        
        # RAGCN losss(short term)
        short_reg_loss = 0
        for gconv in self.retagnn.short_conv_modulelist:
            w = torch.matmul(gconv.att_r, gconv.basis.view(gconv.num_bases, -1)).view(gconv.num_relations, gconv.in_channels, gconv.out_channels)
            short_reg_loss += torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
        short_reg_loss = short_reg_loss/len(self.retagnn.short_conv_modulelist)
        
        reg_loss += short_reg_loss
        loss += 900*reg_loss
        
        torch.cuda.empty_cache()
        
        return loss
        
    
    def predict(self, interaction):
        user = interaction[self.USER_ID] + self.n_items
        item_seq = interaction[self.ITEM_SEQ]
        test_item = interaction[self.ITEM_ID]
        
        batch_users, batch_sequences, edge_index, edge_type, node_no, node2ids = self.extract_subgraph(user.cpu().numpy(), item_seq.cpu().numpy())
        
        short_term_part = []
        for i in range(len(self.short_term_window)):
            if i != len(self.short_term_window)-1:
                sub_seq_no = batch_sequences[:,self.short_term_window[i]:self.short_term_window[i+1]]
                _,_,edge_index,edge_type,_,_ = self.extract_subgraph(batch_users, batch_sequences, sub_seq_no=sub_seq_no, node2ids=node2ids)
                short_term_part.append((edge_index,edge_type))
                
        user = self.id2node_id(user, node2ids)
        item_seq = self.id2node_id(item_seq, node2ids)
                
        X_user_item = [user, item_seq, test_item]
        X_graph_base = [edge_index, edge_type, node_no, short_term_part]
        rating_pred = torch.diag(self.retagnn(X_user_item, X_graph_base, for_pred=True))
        
        torch.cuda.empty_cache()
        
        return rating_pred
    
    
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID] + self.n_items
        item_seq = interaction[self.ITEM_SEQ]
        
        
        test_item = torch.tensor([x for x in range(self.n_items)]).to(self.device)
        
        batch_users, batch_sequences, edge_index, edge_type, node_no, node2ids = self.extract_subgraph(user.cpu().numpy(), item_seq.cpu().numpy())
        
        short_term_part = []
        for i in range(len(self.short_term_window)):
            if i != len(self.short_term_window)-1:
                sub_seq_no = batch_sequences[:,self.short_term_window[i]:self.short_term_window[i+1]]
                _,_,edge_index,edge_type,_,_ = self.extract_subgraph(batch_users, batch_sequences, sub_seq_no=sub_seq_no, node2ids=node2ids)
                short_term_part.append((edge_index,edge_type))
                
        user = self.id2node_id(user, node2ids)
        item_seq = self.id2node_id(item_seq, node2ids)
                
        X_user_item = [user, item_seq, test_item]
        X_graph_base = [edge_index, edge_type, node_no, short_term_part]
        rating_pred = self.retagnn(X_user_item, X_graph_base, for_pred=True)
        # rating_pred = torch.rand_like(rating_pred)
        
        torch.cuda.empty_cache()
        
        return rating_pred
