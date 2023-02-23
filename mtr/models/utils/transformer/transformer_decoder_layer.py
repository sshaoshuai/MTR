# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Modified by Shaoshuai Shi 
# All Rights Reserved


"""
Modified from https://github.com/IDEA-opensource/DAB-DETR/blob/main/models/DAB_DETR/transformer.py
"""

from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .transformer_encoder_layer import _get_activation_fn
from .multi_head_attention_local import MultiheadAttentionLocal
from .multi_head_attention import MultiheadAttention


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False, use_local_attn=False):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model, without_weight=True)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)

        self.use_local_attn = use_local_attn

        if self.use_local_attn:
            self.cross_attn = MultiheadAttentionLocal(d_model*2, nhead, dropout=dropout, vdim=d_model, without_weight=True)
        else:
            self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model, without_weight=True)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)


        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed = None,
                is_first = False,
                memory_key_padding_mask=None,
                # for local attn
                key_batch_cnt=None,   # (B)
                index_pair=None,  # (N1+N2..., K)
                index_pair_batch=None,  # (N1+N2...),
                memory_valid_mask=None,  # (M1+M2+...)  # TODO DO WE NEED TO MASK OUT INVALID DATA FOR LINEAR LAYER?
                ):
        """

        Args:
            tgt (num_query, B, C):
            memory (M1 + M2 + ..., C):
            pos (M1 + M2 + ..., C):
            query_pos (num_query, B, C):
            query_sine_embed (num_query, B, C):
            is_first (bool, optional):

        Returns:
            _type_: _description_
        """
        num_queries, bs, n_model = tgt.shape
        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=None)[0]
            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)


        if self.use_local_attn:
            # Transform the queries to stack format
            query_batch_cnt = torch.zeros_like(key_batch_cnt)
            query_batch_cnt.fill_(num_queries)

            query_pos = query_pos.permute(1, 0, 2).contiguous().view(-1, n_model)  # (B * num_q, C)
            query_sine_embed = query_sine_embed.permute(1, 0, 2).contiguous().view(-1, n_model)  # (B * num_q, C)
            tgt = tgt.permute(1, 0, 2).contiguous().view(-1, n_model)  # (B * num_q, C)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)

        if self.use_local_attn and memory_valid_mask is not None:
            valid_memory = memory[memory_valid_mask]

            k_content_valid = self.ca_kcontent_proj(valid_memory)
            k_content = memory.new_zeros(memory.shape[0], k_content_valid.shape[-1])
            k_content[memory_valid_mask] = k_content_valid

            v_valid = self.ca_v_proj(valid_memory)
            v = memory.new_zeros(memory.shape[0], v_valid.shape[-1])
            v[memory_valid_mask] = v_valid

            valid_pos = pos[memory_valid_mask]
            k_pos_valid = self.ca_kpos_proj(valid_pos)
            k_pos = pos.new_zeros(memory.shape[0], k_pos_valid.shape[-1])
            k_pos[memory_valid_mask] = k_pos_valid
        else:
            k_content = self.ca_kcontent_proj(memory)
            v = self.ca_v_proj(memory)
            k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)

        if self.use_local_attn:
            num_q_all, n_model = q_content.shape
            num_k_all, _ = k_content.shape

            q = q.view(num_q_all, self.nhead, n_model//self.nhead)
            query_sine_embed = query_sine_embed.view(num_q_all, self.nhead, n_model//self.nhead)
            q = torch.cat([q, query_sine_embed], dim=-1).view(num_q_all, n_model * 2)

            k = k.view(num_k_all, self.nhead, n_model//self.nhead)
            k_pos = k_pos.view(num_k_all, self.nhead, n_model//self.nhead)
            k = torch.cat([k, k_pos], dim=-1).view(num_k_all, n_model * 2)

            assert num_q_all == len(index_pair)

            tgt2 = self.cross_attn(
                query=q, key=k, value=v,
                index_pair=index_pair, query_batch_cnt=query_batch_cnt, key_batch_cnt=key_batch_cnt, index_pair_batch=index_pair_batch,
                attn_mask=None, vdim=n_model
            )[0]
        else:
            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
            query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
            q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)

            k = k.view(hw, bs, self.nhead, n_model//self.nhead)
            k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
            k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

            tgt2 = self.cross_attn(query=q,
                                   key=k,
                                   value=v, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
