import argparse
import random

import numpy as np
import scipy.sparse as ssp
import torch
from torch_geometric.data import Data

class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        indices = np.concatenate(self.indices[row_selector])
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        shape = [indptr.shape[0] - 1, self.shape[1]]
        return ssp.csr_matrix((data, indices, indptr), shape=shape)


class SparseColIndexer:
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []

        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csc_matrix.shape

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))

        shape = [self.shape[0], indptr.shape[0] - 1]
        return ssp.csc_matrix((data, indices, indptr), shape=shape)
    
    
def subgraph_extraction_labeling(node, A, h=1, sample_ratio=1.0, max_nodes_per_hop=None, isItem = False):
    # extract the h-hop enclosing subgraph around link 'ind'
    if isItem:
        Arow, Acol = SparseColIndexer(A.tocsc()), SparseRowIndexer(A.tocsr())
    else:
        Arow, Acol = SparseRowIndexer(A.tocsr()), SparseColIndexer(A.tocsc())
    a_nodes, b_nodes = [node], []
    a_dist, b_dist = [0], []
    a_visited, b_visited = set([node]), set([])
    a_fringe, b_fringe = set([node]), set([])
    for dist in range(1, h+1):
        if dist % 2 == 1:
            b_fringe = neighbors(a_fringe, Arow)
            b_fringe = b_fringe - b_visited
            b_visited = b_visited.union(b_fringe)
            if sample_ratio < 1.0:
                b_fringe = random.sample(b_fringe, int(sample_ratio*len(b_fringe)))
            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(b_fringe):
                    b_fringe = random.sample(b_fringe, max_nodes_per_hop)
            if len(b_fringe) == 0:
                break
            b_nodes = b_nodes + list(b_fringe)
            b_dist = b_dist + [dist] * len(b_fringe)
        else:
            a_fringe = neighbors(b_fringe, Acol)
            a_fringe = a_fringe - a_visited
            a_visited = a_visited.union(a_fringe)
            if sample_ratio < 1.0:
                a_fringe = random.sample(a_fringe, int(sample_ratio*len(a_fringe)))
            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(a_fringe):
                    a_fringe = random.sample(a_fringe, max_nodes_per_hop)
        
            a_nodes = a_nodes + list(a_fringe)
            a_dist = a_dist + [dist] * len(a_fringe)
    
    if isItem:
        subgraph = Arow[a_nodes][b_nodes, :]
    else:
        subgraph = Arow[a_nodes][:, b_nodes]
    
    # prepare pyg graph constructor input
    a, b, _ = ssp.find(subgraph)  # r is 1, 2... (rating labels + 1)
    b += len(a_nodes)
    if isItem:
        node_labels = [x*2+1 for x in a_dist]
        node_labels.extend([x*2 for x in b_dist]) 
        node_labels[0] = 1
    else:
        node_labels = [x*2 for x in a_dist]
        node_labels.extend([x*2+1 for x in b_dist])
    max_node_label = 2*(h+1)
            
    return a, b, node_labels, max_node_label, a_nodes, b_nodes


def construct_pyg_graph(u, v, node_features, edge_type = None):
    u, v = torch.LongTensor(u), torch.LongTensor(v)  
    x = torch.FloatTensor(node_features)
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], 0)
    if edge_type is not None:
        edge_type = torch.LongTensor(edge_type)
        edge_type = torch.cat([edge_type, edge_type])
        data = Data(x, edge_index, edge_type=edge_type)
    else:
        data = Data(x, edge_index)
    
    return data


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    if not fringe:
        return set([])
    return set(A[list(fringe)].indices)

def fix_pad(input, target_length):
    dims = input.shape[1]
    input_length = input.shape[0]
    pad_length = target_length - input_length
    paddings = torch.zeros((pad_length, dims)).to(input.device)
    return torch.cat([input, paddings], dim=0)

def sequence_mask(lengths, max_len=None):
    batch_size=lengths.numel()
    max_len=max_len or lengths.max()
    return (torch.arange(0,max_len,device=lengths.device)
    .type_as(lengths)
    .unsqueeze(0).expand(batch_size,max_len)
    .gt(lengths.unsqueeze(1)))

def generate_subgraph(node, A, h, sample_ratio, max_nodes_per_hop, items_degree, users_degree, isItem = False):
        #item b,a///user a,b
            a, b, node_labels, max_node_label, a_nodes, b_nodes = subgraph_extraction_labeling(node, 
                                                                                               A, 
                                                                                               h=h, 
                                                                                               sample_ratio=sample_ratio, 
                                                                                               max_nodes_per_hop=max_nodes_per_hop, 
                                                                                               isItem = isItem)
            node_feature = np.eye(max_node_label)[node_labels]
            if isItem:
                a_degree = items_degree[a_nodes]
                b_degree = users_degree[b_nodes]
                raw_degree = np.concatenate([a_degree, b_degree]).reshape(node_feature.shape[0], 1)
                node_feature = np.concatenate([node_feature, raw_degree], axis=1)
                data = construct_pyg_graph(b, a, node_feature)
            else:
                a_degree = users_degree[a_nodes]
                b_degree = items_degree[b_nodes]
                raw_degree = np.concatenate([a_degree, b_degree]).reshape(node_feature.shape[0], 1)
                node_feature = np.concatenate([node_feature, raw_degree], axis=1)
                data = construct_pyg_graph(a, b, node_feature)
            
            return data, node