#ifndef _ATTENTION_FUNC_H
#define _ATTENTION_FUNC_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>


void attention_weight_computation_launcher(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float *query_features, const float* key_features,
    float *output);


int attention_weight_computation_wrapper(
    int b, int total_query_num, int local_size, int total_key_num, int nhead, int hdim,
    at::Tensor query_batch_cnt, at::Tensor key_batch_cnt, at::Tensor index_pair_batch,
    at::Tensor index_pair, at::Tensor query_features, at::Tensor key_features,
    at::Tensor output);


void attention_weight_computation_grad_launcher(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float *query_features, const float* key_features,
    float *grad_out, float* grad_query_features, float* grad_key_features);


int attention_weight_computation_grad_wrapper(
    int b, int total_query_num, int local_size, int total_key_num, int nhead, int hdim,
    at::Tensor query_batch_cnt, at::Tensor key_batch_cnt, at::Tensor index_pair_batch,
    at::Tensor index_pair, at::Tensor query_features, at::Tensor key_features,
    at::Tensor grad_out, at::Tensor grad_query_features, at::Tensor grad_key_features);


void attention_value_computation_launcher(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float *attn_weight, const float* value_features,
    float *output);


int attention_value_computation_wrapper(
    int b, int total_query_num, int local_size, int total_key_num, int nhead, int hdim,
    at::Tensor query_batch_cnt, at::Tensor key_batch_cnt, at::Tensor index_pair_batch,
    at::Tensor index_pair, at::Tensor attn_weight, at::Tensor value_features,
    at::Tensor output);


void attention_value_computation_grad_launcher(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float *attn_weight, const float* value_features,
    float *grad_out, float* grad_attn_weight, float* grad_value_features);


int attention_value_computation_grad_wrapper(
    int b, int total_query_num, int local_size, int total_key_num, int nhead, int hdim,
    at::Tensor query_batch_cnt, at::Tensor key_batch_cnt, at::Tensor index_pair_batch,
    at::Tensor index_pair, at::Tensor attn_weight, at::Tensor value_features,
    at::Tensor grad_out, at::Tensor grad_attn_weight, at::Tensor grad_value_features);

#endif