from __future__ import print_function

import multiprocessing as mp
import os
import random
import time
import warnings

import numpy as np
import scipy.sparse as ssp
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))

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


def subgraph_extraction_labeling(ind, Arow, Acol, h=1, sample_ratio=1.0, max_nodes_per_hop=None, 
                                 u_features=None, v_features=None, class_values=None, 
                                 y=1):
    # extract the h-hop enclosing subgraph around link 'ind'
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = set([ind[0]]), set([ind[1]])
    u_fringe, v_fringe = set([ind[0]]), set([ind[1]])
    for dist in range(1, h+1):
        v_fringe, u_fringe = neighbors(u_fringe, Arow), neighbors(v_fringe, Acol)
        u_fringe = u_fringe - u_visited
        v_fringe = v_fringe - v_visited
        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)
        if sample_ratio < 1.0:
            u_fringe = random.sample(u_fringe, int(sample_ratio*len(u_fringe)))
            v_fringe = random.sample(v_fringe, int(sample_ratio*len(v_fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                u_fringe = random.sample(u_fringe, max_nodes_per_hop)
            if max_nodes_per_hop < len(v_fringe):
                v_fringe = random.sample(v_fringe, max_nodes_per_hop)
        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break
        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)
    subgraph = Arow[u_nodes][:, v_nodes]
    # remove link between target nodes
    subgraph[0, 0] = 0
    
    # prepare pyg graph constructor input
    u, v, r = ssp.find(subgraph)  # r is 1, 2... (rating labels + 1)
    v += len(u_nodes)
    r = r - 1  # transform r back to rating label
    num_nodes = len(u_nodes) + len(v_nodes)
    node_labels = [x*2 for x in u_dist] + [x*2+1 for x in v_dist]
    max_node_label = 2*h + 1
    y = y
    
    # get node features
    if u_features is not None:
        u_features = u_features[u_nodes]
    if v_features is not None:
        v_features = v_features[u_nodes]
    node_features = None
    # only output node features for the target user and item
    if u_features is not None and v_features is not None:
        node_features = [u_features[0], v_features[0]]
            
    return u, v, r, node_labels, max_node_label, y, node_features, ind[0], ind[1]


def construct_pyg_graph(u, v, r, node_labels, max_node_label, y, node_features):
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)  
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], 0)
    edge_type = torch.cat([r, r])
    x = torch.FloatTensor(one_hot(node_labels, max_node_label+1))
    data = Data(x, edge_index, edge_type=edge_type)

    if node_features is not None:
        if type(node_features) == list:  # a list of u_feature and v_feature
            u_feature, v_feature = node_features
            data.u_feature = torch.FloatTensor(u_feature).unsqueeze(0)
            data.v_feature = torch.FloatTensor(v_feature).unsqueeze(0)
        else:
            x2 = torch.FloatTensor(node_features)
            data.x = torch.cat([data.x, x2], 1)
    return data


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    if not fringe:
        return set([])
    return set(A[list(fringe)].indices)


def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x

class MyDynamicDataset(Dataset):
    def __init__(self, A, links, labels, h, sample_ratio, max_nodes_per_hop, 
                 u_features, v_features, class_values, max_num=None):
        super(MyDynamicDataset, self).__init__()
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.h = h
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        self.u_features = u_features
        self.v_features = v_features
        self.class_values = class_values
        if max_num is not None:
            np.random.seed(123)
            num_links = len(links[0])
            perm = np.random.permutation(num_links)
            perm = perm[:max_num]
            self.links = (links[0][perm], links[1][perm])
            self.labels = labels[perm]

    def len(self):
        return len(self.links[0])

    def get(self, idx):
        i, j = self.links[0][idx], self.links[1][idx]
        g_label = self.labels[idx]
        tmp = subgraph_extraction_labeling(
            (i, j), self.Arow, self.Acol, self.h, self.sample_ratio, self.max_nodes_per_hop, 
            self.u_features, self.v_features, self.class_values, g_label
        )
        return construct_pyg_graph(*tmp[:-2])
    
    def get_i_j(self, i, j, g_label = 1):
        tmp = subgraph_extraction_labeling(
            (i, j), self.Arow, self.Acol, self.h, self.sample_ratio, self.max_nodes_per_hop, 
            self.u_features, self.v_features, self.class_values, g_label
        )
        return construct_pyg_graph(*tmp[:-2])
    
def links2subgraphs(Arow, 
                    Acol, 
                    links, 
                    labels, 
                    h=1, 
                    sample_ratio=1.0, 
                    max_nodes_per_hop=None, 
                    u_features=None, 
                    v_features=None, 
                    class_values=None, 
                    parallel=True, show_bar = True, pools = mp.cpu_count() - 1):
    # extract enclosing subgraphs
    g_list = {}
    if not parallel:
        if show_bar :
            print('Enclosing subgraph extraction begins...')
            with tqdm(total=len(links[0])) as pbar:
                for i, j, g_label in zip(links[0], links[1], labels):
                    tmp = subgraph_extraction_labeling(
                        (i, j), Arow, Acol, h, sample_ratio, max_nodes_per_hop, u_features, 
                        v_features, class_values, g_label
                    )
                    data = construct_pyg_graph(*tmp[:-2])
                    g_list[(int(i), int(j))] = data
                    pbar.update(1)
        else:
            for i, j, g_label in zip(links[0], links[1], labels):
                    tmp = subgraph_extraction_labeling(
                        (i, j), Arow, Acol, h, sample_ratio, max_nodes_per_hop, u_features, 
                        v_features, class_values, g_label
                    )
                    data = construct_pyg_graph(*tmp[:-2])
                    g_list[(int(i), int(j))] = data
    else:
        torch.multiprocessing.set_sharing_strategy('file_system')
        mp.freeze_support()
        if show_bar :
            print('Enclosing subgraph extraction begins...')
        start = time.time()
        pool = mp.Pool(pools)
        results = pool.starmap_async(
            subgraph_extraction_labeling, 
            [
                ((i, j), Arow, Acol, h, sample_ratio, max_nodes_per_hop, u_features, 
                v_features, class_values, g_label) 
                for i, j, g_label in zip(links[0], links[1], labels)
            ]
        )
        remaining = results._number_left
        if show_bar :
            pbar = tqdm(total=remaining)
        while True:
            if show_bar :
                pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
        if show_bar :
            pbar.close()
        end = time.time()
        if show_bar :
            print("Time elapsed for subgraph extraction: {}s".format(end-start))
        if show_bar :
            print("Transforming to pytorch_geometric graphs...")
        g_list = {}
        if show_bar :
            pbar = tqdm(total=len(results))
        while results:
            tmp = results.pop()
            g_list[(int(tmp[-2]), int(tmp[-1]))] = construct_pyg_graph(*tmp[:-2])
            if show_bar :
                pbar.update(1)
        if show_bar :
            pbar.close()
        end2 = time.time()
        if show_bar :
            print("Time elapsed for transforming to pytorch_geometric graphs: {}s".format(end2-end))
    return g_list
