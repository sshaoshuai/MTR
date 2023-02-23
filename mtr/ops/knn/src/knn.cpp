// Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
// Published at NeurIPS 2022
// Written by Li Jiang, Shaoshuai Shi 
// All Rights Reserved


#include "knn_gpu.h"

// input xyz: (n, 3), float
// input query_xyz: (m, 3), float
// input batch_idxs: (n), int
// input query_batch_offsets: (B + 1), int, offsets[-1] = m
// output idx: (n, k), int
void knn_batch(at::Tensor xyz_tensor, at::Tensor query_xyz_tensor, at::Tensor batch_idxs_tensor, at::Tensor query_batch_offsets_tensor, at::Tensor idx_tensor, int n, int m, int k){

    const float *query_xyz = query_xyz_tensor.data<float>();
    const float *xyz = xyz_tensor.data<float>();
    const int *batch_idxs = batch_idxs_tensor.data<int>();
    const int *query_batch_offsets = query_batch_offsets_tensor.data<int>();
    int *idx = idx_tensor.data<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    knn_batch_cuda(n, m, k, xyz, query_xyz, batch_idxs, query_batch_offsets, idx, stream);
}


void knn_batch_mlogk(at::Tensor xyz_tensor, at::Tensor query_xyz_tensor, at::Tensor batch_idxs_tensor, at::Tensor query_batch_offsets_tensor, at::Tensor idx_tensor, int n, int m, int k){

    const float *query_xyz = query_xyz_tensor.data<float>();
    const float *xyz = xyz_tensor.data<float>();
    const int *batch_idxs = batch_idxs_tensor.data<int>();
    const int *query_batch_offsets = query_batch_offsets_tensor.data<int>();
    int *idx = idx_tensor.data<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    knn_batch_mlogk_cuda(n, m, k, xyz, query_xyz, batch_idxs, query_batch_offsets, idx, stream);
}
