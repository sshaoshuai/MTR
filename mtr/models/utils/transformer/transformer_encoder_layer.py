# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Modified by Shaoshuai Shi 
# All Rights Reserved


"""
Reference: https://github.com/dvlab-research/DeepVision3D/blob/master/EQNet/eqnet/transformer/multi_head_attention.py
"""

from typing import Optional, List


from torch import nn, Tensor
import torch.nn.functional as F
from .multi_head_attention_local import MultiheadAttentionLocal
from .multi_head_attention import MultiheadAttention


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

    
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, use_local_attn=False):
        super().__init__()
        self.use_local_attn = use_local_attn
        
        if self.use_local_attn:
            self.self_attn = MultiheadAttentionLocal(d_model, nhead, dropout=dropout) 
        else:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, 
                     index_pair=None, 
                     query_batch_cnt=None, 
                     key_batch_cnt=None, 
                     index_pair_batch=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, 
                              index_pair=index_pair, query_batch_cnt=query_batch_cnt, 
                              key_batch_cnt=key_batch_cnt, index_pair_batch=index_pair_batch)[0] 
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None, 
                    index_pair=None, 
                    query_batch_cnt=None, 
                    key_batch_cnt=None, 
                    index_pair_batch=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, 
                              index_pair=index_pair, query_batch_cnt=query_batch_cnt, 
                              key_batch_cnt=key_batch_cnt, index_pair_batch=index_pair_batch)[0] 
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, 
                # for local-attn
                index_pair=None, 
                query_batch_cnt=None, 
                key_batch_cnt=None, 
                index_pair_batch=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, 
                                    index_pair=index_pair, query_batch_cnt=query_batch_cnt, 
                                    key_batch_cnt=key_batch_cnt, index_pair_batch=index_pair_batch)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, 
                                 index_pair=index_pair, query_batch_cnt=query_batch_cnt, 
                                 key_batch_cnt=key_batch_cnt, index_pair_batch=index_pair_batch)