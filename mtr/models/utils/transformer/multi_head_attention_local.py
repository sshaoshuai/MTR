# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


"""
Mostly copy-paste from https://github.com/dvlab-research/DeepVision3D/blob/master/EQNet/eqnet/transformer/multi_head_attention.py
"""

import warnings
import torch
from torch.nn import Linear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch.nn as nn

from mtr.ops import attention


class MultiheadAttentionLocal(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        without_weight=False, 
        version='v2', 
        vdim=None
    ):
        super(MultiheadAttentionLocal, self).__init__()
        self.embed_dim = embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        assert version in ['v1', 'v2'], 'only attention_utils_v1 and attention_utils_v2 are available.'
        # self.attention_utils = attention.__all__[version]
        self.attention_version = version

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = Linear(self.vdim, self.vdim, bias=True)
        
        self.without_weight = without_weight
        if self.without_weight:
            self.in_proj_weight = self.in_proj_bias = None 
            constant_(self.out_proj.bias, 0.0)
        else:
            self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def _proj_qkv(self, t, start, end):
        _w = self.in_proj_weight[start:end, :]
        _b = self.in_proj_bias[start:end]
        t = F.linear(t, _w, _b)
        return t

    def forward(
        self,
        query,  # total_q_num, c
        key,  # total_k_num, c
        value,  # total_k_num, c
        index_pair,  # total_q_num, max_memory_num
        query_batch_cnt,  # bs: query_amount of each batch
        key_batch_cnt,  # bs: key_amount of each batch.
        index_pair_batch,  # total_q_num, batch_index of each query.
        attn_mask=None,  # total_q_num, max_memory_num
        vdim=None,

        # positional encoding setting.
        relative_atten_weights=None,  # total_q_num, max_memory_num, nhead

        # crpe module.
        ctx_rpe_query=None,
        ctx_rpe_key=None,
        ctx_rpe_value=None,
        rpe_distance=None,
        **kwargs
    ):
        r""" To reduce memory cost in attention computation, use index to indicate attention pair.
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            index_pair: the associated key indices of each query for computing attention.
            query_batch_cnt: indicate the query_amount in each batch.
            key_batch_cnt: indicate the key / value amount in each batch.
            index_pair_batch: the batch index of each query.
            attn_mask:  mask that prevents attention to certain positions. This is an additive mask
                (i.e. the values will be added to the attention layer).
            relative_atten_weights: Add relative positional encoding.
            ctx_rpe_query / ctx_rpe_key / ctx_rpe_value: nn.Module for providing contextual relative positional
                encoding given rpe_distance between query and keys.
        Shape:
            - Inputs:
            - query: :math:`(N, C)` where N is the total query tokens length, C is
                the embedding dimension.
            - key: :math:`(M, C)`, where M is the total key tokens length, C is
                the embedding dimension.
            - value: :math:`(M, C)` where M is the total value tokens length (equals to ``key''), C is
                the embedding dimension.
            - index_pair: :math:`(N, L)` where N is the total query tokens length (equals to ``query''),
                L is max_key_num for computing attention.
            - query_batch_cnt: :math:`(B)` where B indicate batch_size.
            - key_batch_cnt: :math:`(B)` where B indicate batch_size.
            - index_pair_batch: :math:`(N)` where N is the total query tokens length (equals to ``query'')
            - attn_mask: :math:`(N, L)` where N is the total query tokens length (equals to ``query''),
                L is max_key_num for computing attention.
            - relative_atten_weights: :math:`(N, L, H)` where N is the total query tokens length (equals to ``query''),
                L is max_key_num for computing attention, H is head_num for computing attention.
            - rpe_distance: :math:`(N, L, 3)` where N is the total query tokens length (equals to ``query''),
                L is max_key_num for computing attention.
            - Outputs:
            - attn_output: :math:`(N, C)` where N is the total query tokens length,
                C is the embedding dimension.
            - attn_output_weights: :math:`(N, L, H)` where N is the total query tokens length (equals to ``query''),
                L is max_key_num for computing attention, H is head_num for computing attention.
        """
        total_query_len, embed_dim = query.size()
        max_memory_len = index_pair.shape[1]
        
        if vdim is None:
            assert key.size() == value.size()
            vdim = embed_dim
            v_head_dim = self.head_dim
        else:
            v_head_dim = vdim // self.num_heads
            assert v_head_dim * self.num_heads == vdim 

        scaling = float(self.head_dim) ** -0.5

        # generate qkv features.
        if not self.without_weight:
            q = self._proj_qkv(query, 0, embed_dim)
            q = q * scaling
            k = self._proj_qkv(key, embed_dim, embed_dim * 2)
            v = self._proj_qkv(value, embed_dim * 2, embed_dim * 3)
        else:
            q = query * scaling
            k, v = key, value 

        # -1 in index_pair means this key not joining attention computation.
        used_attn_mask = (index_pair == -1)  # Ignore the -1 pair.
        if attn_mask is not None:
            # attn_mask should have a shape as [total_query_size, max_memory_size]
            attn_mask = attn_mask.to(torch.bool)
            used_attn_mask = torch.logical_or(used_attn_mask, attn_mask)

        q = q.contiguous().view(total_query_len, self.num_heads, self.head_dim)
        k = k.contiguous().view(-1, self.num_heads, self.head_dim)
        v = v.contiguous().view(-1, self.num_heads, v_head_dim)

        # compute attention weight.
        attn_output_weights = attention.__all__[self.attention_version].attention_weight_computation(
            query_batch_cnt, key_batch_cnt, index_pair_batch, index_pair,
            q, k)  # total_query_len, max_memory_len, num_heads
        assert list(attn_output_weights.size()) == [total_query_len, max_memory_len, self.num_heads]

        if ctx_rpe_key is not None:
            rpe_attn_weight = ctx_rpe_key(rpe_distance, k, scaling,
                                          query_batch_cnt, key_batch_cnt,
                                          index_pair_batch, index_pair)
            attn_output_weights = attn_output_weights + rpe_attn_weight
        if ctx_rpe_query is not None:
            rpe_attn_weight = ctx_rpe_query(rpe_distance, q, 1.0, query_batch_cnt)
            attn_output_weights = attn_output_weights + rpe_attn_weight

        if relative_atten_weights is not None:
            # relative_atten_weights: A float tensor with shape [total_query_num, max_memory_num, nhead]
            attn_output_weights = attn_output_weights + relative_atten_weights

        # attn_output_weights: [total_query_num, max_memory_num, nhead]
        used_attn_mask = used_attn_mask.unsqueeze(-1).repeat(1, 1, self.num_heads).contiguous()
        attn_output_weights.masked_fill_(used_attn_mask, float("-inf"))
        attn_output_weights = F.softmax(attn_output_weights, dim=1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        if ctx_rpe_value is not None:
            attn_output = ctx_rpe_value(rpe_distance, attn_output_weights, v,
                                        query_batch_cnt, key_batch_cnt,
                                        index_pair_batch, index_pair)
        else:
            attn_output = attention.__all__[self.attention_version].attention_value_computation(
                query_batch_cnt, key_batch_cnt, index_pair_batch, index_pair,
                attn_output_weights, v)
        assert list(attn_output.size()) == [total_query_len, self.num_heads, v_head_dim]

        attn_output = attn_output.view(total_query_len, vdim)
        
        if self.out_proj is not None:
            attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        return attn_output, attn_output_weights.sum(dim=-1) / self.num_heads