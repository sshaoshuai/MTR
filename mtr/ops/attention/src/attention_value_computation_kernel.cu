/*
Transformer function helper function.
Written by tomztyang,
2021/08/23
*/

#include <math.h>
#include <stdio.h>

#include "attention_func.h"

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
// #define DEBUG


__global__ void attention_value_computation_forward(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float *attn_weight, const float* value_features,
    float *output) {
    // dim3 blocks(DIVUP(total_query_num * local_size, THREADS_PER_BLOCK), nhead, hdim);
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params attn_weight: [total_query_num, local_size, nhead]
    // params value_features: [total_key_num, nhead, hdim]
    // params output: [total_query_num, nhead, hdim]

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;
    int hdim_idx = blockIdx.z;
    if (index >= total_query_num * local_size ||
        head_idx >= nhead ||
        hdim_idx >= hdim) return;

    if (index_pair[index] == -1){
        // Ignore index.
        return;
    }

    int query_idx = index / local_size;
    int batch_idx = index_pair_batch[query_idx];
    int key_start_idx = 0;
    for (int i = 0; i < batch_idx; i++){
        key_start_idx += key_batch_cnt[i];
    }

    // 1. Obtain value features.
    key_start_idx += index_pair[index];
    value_features += key_start_idx * nhead * hdim + head_idx * hdim + hdim_idx;
    // 2. Obtain attention weight.
    attn_weight += index * nhead + head_idx;
    // 3. Do dot product.
    output += query_idx * nhead * hdim + head_idx * hdim + hdim_idx;
    atomicAdd(
        output,
        attn_weight[0] * value_features[0]);
}


void attention_value_computation_launcher(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float *attn_weight, const float* value_features,
    float *output){
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params attn_weight: [total_query_num, local_size, nhead]
    // params value_features: [total_key_num, nhead, hdim]
    // params output: [total_query_num, nhead, hdim]

    dim3 blocks(DIVUP(total_query_num * local_size, THREADS_PER_BLOCK), nhead, hdim);
    dim3 threads(THREADS_PER_BLOCK);
    attention_value_computation_forward<<<blocks, threads>>>(
        b, total_query_num, local_size, total_key_num, nhead, hdim,
        query_batch_cnt, key_batch_cnt, index_pair_batch,
        index_pair, attn_weight, value_features,
        output);
}


__global__ void attention_value_computation_backward(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float *attn_weight, const float* value_features,
    float *grad_out, float * grad_attn_weight, float * grad_value_features) {
    // dim3 blocks(DIVUP(total_query_num * local_size, THREADS_PER_BLOCK), nhead, hdim);
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params attn_weight: [total_query_num, local_size, nhead]
    // params value_features: [total_key_num, nhead, hdim]
    // params grad_out: [total_query_num, nhead, hdim]
    // params grad_attn_weight: [total_query_num, local_size, nhead]
    // params grad_value_features: [total_key_num, nhead, hdim]

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;
    int hdim_idx = blockIdx.z;
    if (index >= total_query_num * local_size ||
        head_idx >= nhead ||
        hdim_idx >= hdim) return;

    if (index_pair[index] == -1){
        // Ignore index.
        return;
    }

    int query_idx = index / local_size;
    int batch_idx = index_pair_batch[query_idx];
    int key_start_idx = 0;
    for (int i = 0; i < batch_idx; i++){
        key_start_idx += key_batch_cnt[i];
    }

    // 1. Obtain value features.
    key_start_idx += index_pair[index];
    value_features += key_start_idx * nhead * hdim + head_idx * hdim + hdim_idx;
    grad_value_features += key_start_idx * nhead * hdim + head_idx * hdim + hdim_idx;
    // 2. Obtain attention weight.
    attn_weight += index * nhead + head_idx;
    grad_attn_weight += index * nhead + head_idx;

    // 3. Obtain grad out.
    grad_out += query_idx * nhead * hdim + head_idx * hdim + hdim_idx;
    atomicAdd(
        grad_attn_weight,
        grad_out[0] * value_features[0]);
    atomicAdd(
        grad_value_features,
        grad_out[0] * attn_weight[0]);
}


void attention_value_computation_grad_launcher(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float *attn_weight, const float* value_features,
    float *grad_out, float* grad_attn_weight, float* grad_value_features){
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params attn_weight: [total_query_num, local_size, nhead]
    // params value_features: [total_key_num, nhead, hdim]
    // params grad_out: [total_query_num, nhead, hdim]
    // params grad_attn_weight: [total_query_num, local_size, nhead]
    // params grad_value_features: [total_key_num, nhead, hdim]

    dim3 blocks(DIVUP(total_query_num * local_size, THREADS_PER_BLOCK), nhead, hdim);
    dim3 threads(THREADS_PER_BLOCK);
    attention_value_computation_backward<<<blocks, threads>>>(
        b, total_query_num, local_size, total_key_num, nhead, hdim,
        query_batch_cnt, key_batch_cnt, index_pair_batch,
        index_pair, attn_weight, value_features,
        grad_out, grad_attn_weight, grad_value_features);
}