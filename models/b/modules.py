import copy
import math

import torch
import torch.nn as nn

from recbole.model.layers import  VanillaAttention, FeedForward
from torch_geometric.nn import GATConv, LGConv

class DIFMultiHeadAttention(nn.Module):
    """
    DIF Multi-head Self-attention layers, a attention score dropout layer is introduced.
    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor
    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer
    """

    def __init__(self, n_heads, hidden_size,attribute_hidden_size,feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len):
        super(DIFMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attribute_attention_head_size = [int(_ / n_heads) for _ in attribute_hidden_size]
        self.attribute_all_head_size = [self.num_attention_heads * _ for _ in self.attribute_attention_head_size]
        self.fusion_type = fusion_type
        self.max_len = max_len

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.query_p = nn.Linear(hidden_size, self.all_head_size)
        self.key_p = nn.Linear(hidden_size, self.all_head_size)

        self.feat_num = feat_num
        self.query_layers = nn.ModuleList([copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in range(self.feat_num)])
        self.key_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in range(self.feat_num)])

        if self.fusion_type == 'concat':
            self.fusion_layer = nn.Linear(self.max_len*(2+self.feat_num), self.max_len)
        elif self.fusion_type == 'gate':
            self.fusion_layer = VanillaAttention(self.max_len,self.max_len)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_attribute(self, x,i):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attribute_attention_head_size[i])
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor,attribute_table,position_embedding, attention_mask):
        item_query_layer = self.transpose_for_scores(self.query(input_tensor))
        item_key_layer = self.transpose_for_scores(self.key(input_tensor))
        item_value_layer = self.transpose_for_scores(self.value(input_tensor))

        pos_query_layer = self.transpose_for_scores(self.query_p(position_embedding))
        pos_key_layer = self.transpose_for_scores(self.key_p(position_embedding))

        item_attention_scores = torch.matmul(item_query_layer, item_key_layer.transpose(-1, -2))
        pos_scores = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))

        attribute_attention_table = []

        if self.fusion_type != 'no_attr':
            for i, (attribute_query, attribute_key) in enumerate(
                    zip(self.query_layers, self.key_layers)):
                attribute_tensor = attribute_table[i].squeeze(-2)
                attribute_query_layer = self.transpose_for_scores_attribute(attribute_query(attribute_tensor),i)
                attribute_key_layer = self.transpose_for_scores_attribute(attribute_key(attribute_tensor),i)
                attribute_attention_scores = torch.matmul(attribute_query_layer, attribute_key_layer.transpose(-1, -2))
                attribute_attention_table.append(attribute_attention_scores.unsqueeze(-2))
            attribute_attention_table = torch.cat(attribute_attention_table,dim=-2)
            table_shape = attribute_attention_table.shape
            feat_atten_num, attention_size = table_shape[-2], table_shape[-1]
        if self.fusion_type == 'sum':
            attention_scores = torch.sum(attribute_attention_table, dim=-2)
            attention_scores = attention_scores + item_attention_scores + pos_scores
        elif self.fusion_type == 'concat':
            attention_scores = attribute_attention_table.view(table_shape[:-2] + (feat_atten_num * attention_size,))
            attention_scores = torch.cat([attention_scores, item_attention_scores, pos_scores], dim=-1)
            attention_scores = self.fusion_layer(attention_scores)
        elif self.fusion_type == 'gate':
            attention_scores = torch.cat(
                [attribute_attention_table, item_attention_scores.unsqueeze(-2), pos_scores.unsqueeze(-2)], dim=-2)
            attention_scores,_ = self.fusion_layer(attention_scores)
        elif self.fusion_type == 'no_attr':
            attention_scores = item_attention_scores + pos_scores

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)


        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, item_value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class DIFTransformerLayer(nn.Module):
    """
    One decoupled transformer layer consists of a decoupled multi-head self-attention layer and a point-wise feed-forward layer.
    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer
    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.
    """

    def __init__(
        self, n_heads, hidden_size,attribute_hidden_size,feat_num, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps,fusion_type,max_len
    ):
        super(DIFTransformerLayer, self).__init__()
        self.multi_head_attention = DIFMultiHeadAttention(
            n_heads, hidden_size,attribute_hidden_size, feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len,
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states,attribute_embed,position_embedding, attention_mask):
        attention_output = self.multi_head_attention(hidden_states,attribute_embed,position_embedding, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output



class DIFTransformerEncoder(nn.Module):
    r""" One decoupled TransformerEncoder consists of several decoupled TransformerLayers.
        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - attribute_hidden_size(list): the hidden size of attributes. Default:[64]
        - feat_num(num): the number of attributes. Default: 1
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
        - fusion_type(str): fusion function used in attention fusion module. Default: 'sum'
                            candidates: 'sum','concat','gate'
    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        attribute_hidden_size=[64],
        feat_num=1,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12,
        fusion_type = 'sum',
        max_len = None
    ):

        super(DIFTransformerEncoder, self).__init__()
        layer = DIFTransformerLayer(
            n_heads, hidden_size,attribute_hidden_size,feat_num, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,fusion_type,max_len
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states,attribute_hidden_states,position_embedding, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output
        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attribute_hidden_states,
                                                                  position_embedding, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
    

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