"""
Mostly copy-paste from https://github.com/dvlab-research/DeepVision3D/blob/master/EQNet/eqnet/ops/attention/attention_utils_v2.py
"""

import torch
import torch.nn as nn
from torch.autograd import Function, Variable

from . import attention_cuda


""" Attention computation code v1."""
class AttentionWeightComputation(Function):
    """
    Generate the attention weight matrix based on:
        * the generated attention pair index (total_query_num, local_size);
        * query features (total_query_num, nhead, hdim)
        * key features (total_key_num, nhead, hdim)
    Generate the attention weight matrix.
        * (total_query_num, local_size)
    """

    @staticmethod
    def forward(ctx,
                query_batch_cnt: torch.Tensor,
                key_batch_cnt: torch.Tensor,
                index_pair_batch: torch.Tensor,
                index_pair: torch.Tensor,
                query_features: torch.Tensor,
                key_features: torch.Tensor):
        """
        :param ctx:
        :param query_batch_cnt: A integer tensor with shape [bs], indicating the query amount for each batch.
        :param key_batch_cnt: A integer tensor with shape [bs], indicating the key amount of each batch.
        :param index_pair_batch: A integer tensor with shape [total_query_num], indicating the batch
            index of each query.
        :param index_pair: A integer tensor with shape [total_query_num, local_size]
            We ignore those index whose value is -1.
        :param query_features: A float tensor with shape [total_query_num, nhead, hdim]
        :param key_features: A float tensor with shape [total_key_num, nhead, hdim]
        :return:
            output: A float tensor with shape [total_query_num, local_size, nhead]
        """
        assert query_batch_cnt.is_contiguous()
        assert key_batch_cnt.is_contiguous()
        assert index_pair_batch.is_contiguous()
        assert index_pair.is_contiguous()
        assert query_features.is_contiguous()
        assert key_features.is_contiguous()

        b = query_batch_cnt.shape[0]
        total_query_num, local_size = index_pair.size()
        total_key_num, nhead, hdim = key_features.size()

        # Need to ensure that every tensor in query features have an output.
        assert total_query_num == query_features.shape[0]

        output = torch.cuda.FloatTensor(total_query_num, local_size, nhead).zero_()

        attention_cuda.attention_weight_computation_wrapper(
            b, total_query_num, local_size, total_key_num, nhead, hdim,
            query_batch_cnt, key_batch_cnt, index_pair_batch,
            index_pair, query_features, key_features,
            output)
        ctx.for_backwards = (
            b, total_query_num, local_size, total_key_num, nhead, hdim,
            query_batch_cnt, key_batch_cnt, index_pair_batch,
            index_pair, query_features, key_features
        )
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: [total_query_num, local_size, nhead]
        Returns:
            grad_query_features:  [total_query_num, nhead, hdim]
            grad_key_features: [total_key_num, nhead, hdim]
        """
        (b, total_query_num, local_size, total_key_num, nhead, hdim,
         query_batch_cnt, key_batch_cnt, index_pair_batch,
         index_pair, query_features, key_features) = ctx.for_backwards

        grad_query_features = Variable(torch.cuda.FloatTensor(
            total_query_num, nhead, hdim).zero_())
        grad_key_features = Variable(torch.cuda.FloatTensor(
            total_key_num, nhead, hdim).zero_())

        grad_out_data = grad_out.data.contiguous()
        attention_cuda.attention_weight_computation_grad_wrapper(
            b, total_query_num, local_size, total_key_num, nhead, hdim,
            query_batch_cnt, key_batch_cnt, index_pair_batch,
            index_pair, query_features, key_features,
            grad_out_data, grad_query_features.data, grad_key_features.data)
        return None, None, None, None, grad_query_features, grad_key_features


attention_weight_computation = AttentionWeightComputation.apply


class AttentionValueComputation(Function):
    """
    Generate the attention result based on:
        * the generated attention pair index (total_query_num, local_size);
        * value features (total_key_num, nhead, hdim)
        * attn_weight (total_query_num, local_size, nhead)
    Generate the attention result.
        * (total_query_num, nhead, hdim)
    """

    @staticmethod
    def forward(ctx,
                query_batch_cnt: torch.Tensor,
                key_batch_cnt: torch.Tensor,
                index_pair_batch: torch.Tensor,
                index_pair: torch.Tensor,
                attn_weight: torch.Tensor,
                value_features: torch.Tensor):
        """
        :param ctx:
        :param query_batch_cnt: A integer tensor with shape [bs], indicating the query amount for each batch.
        :param key_batch_cnt: A integer tensor with shape [bs], indicating the key amount of each batch.
        :param index_pair_batch: A integer tensor with shape [total_query_num], indicating the batch
            index of each query.
        :param index_pair: A integer tensor with shape [total_query_num, local_size]
            We ignore those index whose value is -1.
        :param attn_weight: A float tensor with shape [total_query_num, local_size, nhead]
        :param value_features: A float tensor with shape [total_key_num, nhead, hdim]
        :return:
            output: A float tensor with shape [total_query_num, nhead, hdim]
        """
        assert query_batch_cnt.is_contiguous()
        assert key_batch_cnt.is_contiguous()
        assert index_pair_batch.is_contiguous()
        assert index_pair.is_contiguous()
        assert attn_weight.is_contiguous()
        assert value_features.is_contiguous()

        b = query_batch_cnt.shape[0]
        total_query_num, local_size = index_pair.size()
        total_key_num, nhead, hdim = value_features.size()

        # Need to ensure that every tensor in query features have an output.
        assert total_query_num == attn_weight.shape[0]

        output = torch.cuda.FloatTensor(total_query_num, nhead, hdim).zero_()

        attention_cuda.attention_value_computation_wrapper(
            b, total_query_num, local_size, total_key_num, nhead, hdim,
            query_batch_cnt, key_batch_cnt, index_pair_batch,
            index_pair, attn_weight, value_features,
            output)
        ctx.for_backwards = (
            b, total_query_num, local_size, total_key_num, nhead, hdim,
            query_batch_cnt, key_batch_cnt, index_pair_batch,
            index_pair, attn_weight, value_features
        )
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: [total_query_num, nhead, hdim]
        Returns:
            grad_attn_weight:  [total_query_num, local_size, nhead]
            grad_value_features: [total_key_num, nhead, hdim]
        """
        (b, total_query_num, local_size, total_key_num, nhead, hdim,
         query_batch_cnt, key_batch_cnt, index_pair_batch,
         index_pair, attn_weight, value_features) = ctx.for_backwards

        grad_attn_weight = Variable(torch.cuda.FloatTensor(
            total_query_num, local_size, nhead).zero_())
        grad_value_features = Variable(torch.cuda.FloatTensor(
            total_key_num, nhead, hdim).zero_())

        grad_out_data = grad_out.data.contiguous()
        attention_cuda.attention_value_computation_grad_wrapper(
            b, total_query_num, local_size, total_key_num, nhead, hdim,
            query_batch_cnt, key_batch_cnt, index_pair_batch,
            index_pair, attn_weight, value_features,
            grad_out_data, grad_attn_weight.data, grad_value_features.data)
        return None, None, None, None, grad_attn_weight, grad_value_features


attention_value_computation = AttentionValueComputation.apply