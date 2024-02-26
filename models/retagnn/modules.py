import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import dropout_adj, softmax

from models.retagnn.utils import uniform


class GNN_SR_Net(nn.Module):
    def __init__(self,config,item_num,node_num,relation_num,rgcn,device):
        super(GNN_SR_Net, self).__init__()
        #paramter setting
        dim = config['retagnn_dim']
        conv_layer_num = config['conv_layer_num']
        self.adj_dropout = config['adj_dropout']
        self.num_bases = config['num_bases']
        self.device = device

        #claim variable
        self.predict_emb_w = nn.Embedding(item_num, 2*conv_layer_num*dim, padding_idx=0).to(device)
        self.predict_emb_b = nn.Embedding(item_num, 1, padding_idx=0).to(device)
        self.predict_emb_w.weight.data.normal_(0, 1.0 / self.predict_emb_w.embedding_dim)
        self.predict_emb_b.weight.data.zero_()
        self.node_embeddings = nn.Embedding(node_num, dim, padding_idx=0).to(device)
        self.node_embeddings.weight.data.normal_(0, 1.0 / self.node_embeddings.embedding_dim)

        #RGCN setting(long term)
        conv_latent_dim = [dim for i in range(conv_layer_num)]
        self.conv_modulelist = torch.nn.ModuleList()
        self.conv_modulelist.append(rgcn(dim,conv_latent_dim[0],relation_num,self.num_bases,device=self.device))
        for i in range(len(conv_latent_dim)-1):
            self.conv_modulelist.append(rgcn(conv_latent_dim[i],conv_latent_dim[i+1],relation_num,self.num_bases,device=self.device))
        
        #RGCN setting(short term)
        conv_latent_dim = [dim for i in range(conv_layer_num)]
        self.short_conv_modulelist = torch.nn.ModuleList()
        self.short_conv_modulelist.append(rgcn(dim,conv_latent_dim[0],relation_num,self.num_bases,device=self.device))
        for i in range(len(conv_latent_dim)-1):
            self.short_conv_modulelist.append(rgcn(conv_latent_dim[i],conv_latent_dim[i+1],relation_num,self.num_bases,device=self.device))

        ##TSAL setting
        self.TSAL_dim = 2*conv_layer_num*dim
        self.head_num = 1 #(default)
        self.attn_drop = 0.0 #(default) 
        self.TSAL_W_Q = Variable(torch.zeros(self.TSAL_dim,self.TSAL_dim).type(torch.FloatTensor), requires_grad=True).to(device)
        self.TSAL_W_K = Variable(torch.zeros(self.TSAL_dim,self.TSAL_dim).type(torch.FloatTensor), requires_grad=True).to(device)
        self.TSAL_W_V = Variable(torch.zeros(self.TSAL_dim,self.TSAL_dim).type(torch.FloatTensor), requires_grad=True).to(device)

        self.TSAL_W_Q = torch.nn.init.xavier_uniform_(self.TSAL_W_Q)
        self.TSAL_W_K = torch.nn.init.xavier_uniform_(self.TSAL_W_K)
        self.TSAL_W_V = torch.nn.init.xavier_uniform_(self.TSAL_W_V)
        self.drop_layer = nn.Dropout(p=self.attn_drop)
        
    
    def Temporal_Self_Attention_Layer(self,input_tensor):
        time_step = input_tensor.size()[1]
        Q_tensor = torch.matmul(input_tensor,self.TSAL_W_Q)  #(N,T,input_dim)->(N,T,input_dim)
        K_tensor = torch.matmul(input_tensor,self.TSAL_W_K)  #(N,T,input_dim)->(N,T,input_dim)
        V_tensor = torch.matmul(input_tensor,self.TSAL_W_V)  #(N,T,input_dim)->(N,T,input_dim)

        Q_tensor_ = torch.cat(torch.split(Q_tensor,int(self.TSAL_dim/self.head_num),2),0)   #(N,T,input_dim)->(N*head_num,T,input_dim/head_num)
        K_tensor_ = torch.cat(torch.split(K_tensor,int(self.TSAL_dim/self.head_num),2),0)   #(N,T,input_dim)->(N*head_num,T,input_dim/head_num)
        V_tensor_ = torch.cat(torch.split(V_tensor,int(self.TSAL_dim/self.head_num),2),0)   #(N,T,input_dim)->(N*head_num,T,input_dim/head_num)

        output_tensor = torch.matmul(Q_tensor_,K_tensor_.permute(0,2,1)) #(N*head_num,T,input_dim/head_num)->(N*head_num,T,T)
        output_tensor = output_tensor/(time_step ** 0.5)

        diag_val = torch.ones_like(output_tensor[0,:,:]).to(self.device) #(T,T)
        tril_tensor = torch.tril(diag_val).unsqueeze(0) #(T,T)->(1,T,T),where tril is lower_triangle matx.
        masks = tril_tensor.repeat(output_tensor.size()[0],1,1) #(1,T,T)->(N*head_num,T,T)
        padding = torch.ones_like(masks) * (-2 ** 32 + 1) 
        output_tensor = torch.where(masks.eq(0),padding,output_tensor) #(N*head_num,T,T),where replace lower_trianlge 0 with 1.
        output_tensor= F.softmax(output_tensor,1) 
        self.TSA_attn = output_tensor #(visiual)

        output_tensor = self.drop_layer(output_tensor) 
        N = output_tensor.size()[0]
        output_tensor = torch.matmul(output_tensor, V_tensor_) #(N*head_num,T,T),(N*head_num,T,input_dim/head_num)->(N*head_num,T,input_dim/head_num)
        output_tensor = torch.cat(torch.split(output_tensor,int(N/self.head_num),0),-1) #(N*head_num,T,input_dim/head_num) -> (N,T,input_dim)
        # Optional: Feedforward and residual
        # if $FLAGS.position_ffn:
        #     output_tensor = self.feedforward(output_tensor)
        # if residual:
        #     output_tensor += input_tensor
        return output_tensor

    def forward(self,X_user_item,X_graph_base,for_pred=False):
        batch_users,batch_sequences,items_to_predict = X_user_item[0],X_user_item[1],X_user_item[2]
        edge_index,edge_type,node_no,short_term_part = X_graph_base[0],X_graph_base[1],X_graph_base[2],X_graph_base[3]
        x = self.node_embeddings(node_no)
 
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(edge_index, edge_type, p=self.adj_dropout, force_undirected=False, num_nodes=len(x), training=not for_pred)
        concat_states = []
        rate = torch.tensor([[1] for i in range(edge_type.size()[0])]).to(self.device)
         
        self.attn_weight_list = list()
        for conv in self.conv_modulelist:
            x = torch.tanh(conv(x, edge_index, edge_type,gate_emd2=None,rate=rate))
            concat_states.append(x)
            self.attn_weight_list.append(conv.attn_weight)
        
        for conv in self.short_conv_modulelist:
            for i in range(len(short_term_part)):
                short_edge_index,short_edge_type = short_term_part[i][0],short_term_part[i][1]
                x = torch.tanh(conv(x, short_edge_index, short_edge_type,gate_emd2=None,rate=rate))
            concat_states.append(x)
 
        concat_states = torch.cat(concat_states, 1)
        user_emb = concat_states[batch_users]
        item_embs_conv = concat_states[batch_sequences]   
        item_embs = self.Temporal_Self_Attention_Layer(item_embs_conv)

        '''
        user_emb : shape(bz,dim)
        item_embs : shape(bz,L,dim)
        items_to_predict(train) : shape(bz,2*H)
        items_to_predict(test) : shape(bz,topk)
        '''

        pe_w = self.predict_emb_w(items_to_predict) 
        pe_b = self.predict_emb_b(items_to_predict) 
        if for_pred:
            pe_w = pe_w.squeeze()
            pe_b = pe_b.squeeze()
            # user-pred_item
            res = user_emb.mm(pe_w.t()) + pe_b 
            # item-item 
            rel_score = torch.matmul(item_embs, pe_w.t().unsqueeze(0)) 
            rel_score = torch.sum(rel_score, dim=1) 
            res += rel_score  
            return res        
        else:
            # user-pred_item
            res = torch.baddbmm(pe_b, pe_w, user_emb.unsqueeze(2)).squeeze()
            # item-item 
            rel_score = item_embs.bmm(pe_w.permute(0, 2, 1))
            rel_score = torch.sum(rel_score, dim=1).view(res.shape)
            res += rel_score
            return res,user_emb,item_embs_conv
        
        
class RAGCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, num_relations, num_bases,device,
                 root_weight=True, bias=True, **kwargs):
        super(RAGCNConv, self).__init__(aggr='add', **kwargs)
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels)).to(self.device)
        self.att_r = Param(torch.Tensor(num_relations, num_bases)).to(self.device)
        self.heads = 1
        self.att = Param(torch.Tensor(1, self.heads, 2 * out_channels)).to(self.device)
        self.gate_layer = nn.Linear(2*out_channels, 1)
        self.relu = nn.ReLU()
        self.negative_slope = 0.2

        if root_weight:
            self.root = Param(torch.Tensor(in_channels, out_channels)).to(self.device)
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels)).to(self.device)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.dropout = 0

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att_r)
        uniform(size, self.root)
        uniform(size, self.bias)
        uniform(size, self.att)


    def forward(self, x, edge_index, edge_type,gate_emd2,rate, edge_norm=None, size=None):
        """"""
        # if size is None and torch.is_tensor(x):
        #     edge_index, _ = remove_self_loops(edge_index)
        #     edge_index, _ = add_self_loops(edge_index,
        #                                    num_nodes=x.size(self.node_dim))
        if gate_emd2 is not None:
            self.gate_emd2 = gate_emd2
        else:
            self.gate_emd2 = None
        if rate is not None:
            self.rate = rate
        else:
            print('[ERROR]: rate is empty.')
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)

    def message(self, x_j, edge_index_j, edge_index_i, x_i,edge_type, size_i, edge_norm):
        '''
        out shape(6,32)
        w1 (it is in if else;) shape(8,16,32)
        w2 (it is in if else;) shape(6,16,32)
        x_j shape(6,16)
        edge_index_j = 0,0,0,1,2,3
        '''
        ## thelta_r * x_j
        w = torch.matmul(self.att_r, self.basis.view(self.num_bases, -1)).to(self.device)

        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            x_j = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        ## thelta_root * x_i
        if self.root is not None:
            if x_i is None:
                x_i = self.root
            else:
                x_i = torch.matmul(x_i, self.root)

        ## attention_ij
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        
        # gate mechanism (belta)
        # gate_emd1 = torch.mean(torch.cat([x_i, x_j], dim=-1).view(-1,2*self.out_channels),0)
        # if self.gate_emd2 is None:
        #     gate_emd = torch.cat([gate_emd1.view(1,-1),gate_emd1.view(1,-1)],1)
        # else:
        #     gate_emd = torch.cat([gate_emd1.view(1,-1),self.gate_emd2.view(1,-1)],1)
        #belta = self.relu(self.gate_layer(gate_emd))
        #belta = belta * self.rate
        #alpha = F.leaky_relu(alpha + belta, self.negative_slope)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i , num_nodes = size_i)
        self.attn_weight =  alpha
 
        # if return_attention_weights:
        #     self.alpha = alpha

        #alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.view(-1, self.heads, 1)
        #out = x_j 
        out = out.view(-1,self.out_channels)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        '''
        x shape:(4,16)
        aggr_out shape:(4,32)
        '''
        if self.root is not None:
            if x is None:
                aggr_out = aggr_out + self.root
            else:
                aggr_out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.num_relations)
