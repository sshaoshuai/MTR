#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "attention_func.h"

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


int attention_weight_computation_wrapper(
    int b, int total_query_num, int local_size, int total_key_num, int nhead, int hdim,
    at::Tensor query_batch_cnt, at::Tensor key_batch_cnt, at::Tensor index_pair_batch,
    at::Tensor index_pair, at::Tensor query_features, at::Tensor key_features,
    at::Tensor output){
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params query_features: [total_query_num, nhead, hdim]
    // params key_features: [total_key_num, nhead, hdim]
    // params output: [total_query_num, local_size, nhead]
    CHECK_INPUT(query_batch_cnt);
    CHECK_INPUT(key_batch_cnt);
    CHECK_INPUT(index_pair_batch);
    CHECK_INPUT(index_pair);
    CHECK_INPUT(query_features);
    CHECK_INPUT(key_features);

    CHECK_INPUT(output);

    const int *query_batch_cnt_data = query_batch_cnt.data<int>();
    const int *key_batch_cnt_data = key_batch_cnt.data<int>();
    const int *index_pair_batch_data = index_pair_batch.data<int>();
    const int *index_pair_data = index_pair.data<int>();

    const float *query_features_data = query_features.data<float>();
    const float *key_features_data = key_features.data<float>();

    float *output_data = output.data<float>();

    attention_weight_computation_launcher(
        b, total_query_num, local_size, total_key_num, nhead, hdim,
        query_batch_cnt_data, key_batch_cnt_data, index_pair_batch_data,
        index_pair_data, query_features_data, key_features_data,
        output_data);

    return 1;
}


int attention_weight_computation_grad_wrapper(
    int b, int total_query_num, int local_size, int total_key_num, int nhead, int hdim,
    at::Tensor query_batch_cnt, at::Tensor key_batch_cnt, at::Tensor index_pair_batch,
    at::Tensor index_pair, at::Tensor query_features, at::Tensor key_features,
    at::Tensor grad_out, at::Tensor grad_query_features, at::Tensor grad_key_features){
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params query_features: [total_query_num, nhead, hdim]
    // params key_features: [total_key_num, nhead, hdim]
    // params grad_out: [total_query_num, local_size, nhead]
    // params grad_query_features: [total_query_num, nhead, hdim]
    // params grad_key_features: [total_key_num, nhead, hdim]
    CHECK_INPUT(query_batch_cnt);
    CHECK_INPUT(key_batch_cnt);
    CHECK_INPUT(index_pair_batch);
    CHECK_INPUT(index_pair);
    CHECK_INPUT(query_features);
    CHECK_INPUT(key_features);

    CHECK_INPUT(grad_out);
    CHECK_INPUT(grad_query_features);
    CHECK_INPUT(grad_key_features);

    const int *query_batch_cnt_data = query_batch_cnt.data<int>();
    const int *key_batch_cnt_data = key_batch_cnt.data<int>();
    const int *index_pair_batch_data = index_pair_batch.data<int>();
    const int *index_pair_data = index_pair.data<int>();

    const float *query_features_data = query_features.data<float>();
    const float *key_features_data = key_features.data<float>();

    float *grad_out_data = grad_out.data<float>();
    float *grad_query_features_data = grad_query_features.data<float>();
    float *grad_key_features_data = grad_key_features.data<float>();

    attention_weight_computation_grad_launcher(
        b, total_query_num, local_size, total_key_num, nhead, hdim,
        query_batch_cnt_data, key_batch_cnt_data, index_pair_batch_data,
        index_pair_data, query_features_data, key_features_data,
        grad_out_data, grad_query_features_data, grad_key_features_data);

    return 1;
}


int attention_value_computation_wrapper(
    int b, int total_query_num, int local_size, int total_key_num, int nhead, int hdim,
    at::Tensor query_batch_cnt, at::Tensor key_batch_cnt, at::Tensor index_pair_batch,
    at::Tensor index_pair, at::Tensor attn_weight, at::Tensor value_features,
    at::Tensor output){
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params attn_weight: [total_query_num, local_size, nhead]
    // params value_features: [total_key_num, nhead, hdim]
    // params output: [total_query_num, nhead, hdim]
    CHECK_INPUT(query_batch_cnt);
    CHECK_INPUT(key_batch_cnt);
    CHECK_INPUT(index_pair_batch);
    CHECK_INPUT(index_pair);
    CHECK_INPUT(attn_weight);
    CHECK_INPUT(value_features);

    CHECK_INPUT(output);

    const int *query_batch_cnt_data = query_batch_cnt.data<int>();
    const int *key_batch_cnt_data = key_batch_cnt.data<int>();
    const int *index_pair_batch_data = index_pair_batch.data<int>();
    const int *index_pair_data = index_pair.data<int>();

    const float *attn_weight_data = attn_weight.data<float>();
    const float *value_features_data = value_features.data<float>();

    float *output_data = output.data<float>();

    attention_value_computation_launcher(
        b, total_query_num, local_size, total_key_num, nhead, hdim,
        query_batch_cnt_data, key_batch_cnt_data, index_pair_batch_data,
        index_pair_data, attn_weight_data, value_features_data,
        output_data);

    return 1;
}


int attention_value_computation_grad_wrapper(
    int b, int total_query_num, int local_size, int total_key_num, int nhead, int hdim,
    at::Tensor query_batch_cnt, at::Tensor key_batch_cnt, at::Tensor index_pair_batch,
    at::Tensor index_pair, at::Tensor attn_weight, at::Tensor value_features,
    at::Tensor grad_out, at::Tensor grad_attn_weight, at::Tensor grad_value_features){
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params attn_weight: [total_query_num, local_size, nhead]
    // params value_features: [total_key_num, nhead, hdim]
    // params grad_out: [total_query_num, nhead, hdim]
    // params grad_attn_weight: [total_query_num, local_size, nhead]
    // params grad_value_features: [total_key_num, nhead, hdim]
    CHECK_INPUT(query_batch_cnt);
    CHECK_INPUT(key_batch_cnt);
    CHECK_INPUT(index_pair_batch);
    CHECK_INPUT(index_pair);
    CHECK_INPUT(attn_weight);
    CHECK_INPUT(value_features);

    CHECK_INPUT(grad_out);
    CHECK_INPUT(grad_attn_weight);
    CHECK_INPUT(grad_value_features);

    const int *query_batch_cnt_data = query_batch_cnt.data<int>();
    const int *key_batch_cnt_data = key_batch_cnt.data<int>();
    const int *index_pair_batch_data = index_pair_batch.data<int>();
    const int *index_pair_data = index_pair.data<int>();

    const float *attn_weight_data = attn_weight.data<float>();
    const float *value_features_data = value_features.data<float>();

    float *grad_out_data = grad_out.data<float>();
    float *grad_attn_weight_data = grad_attn_weight.data<float>();
    float *grad_value_features_data = grad_value_features.data<float>();

    attention_value_computation_grad_launcher(
        b, total_query_num, local_size, total_key_num, nhead, hdim,
        query_batch_cnt_data, key_batch_cnt_data, index_pair_batch_data,
        index_pair_data, attn_weight_data, value_features_data,
        grad_out_data, grad_attn_weight_data, grad_value_features_data);

    return 1;
}