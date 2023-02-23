// Motion Transformer (MTR):  Motion Forecasting Transformer with Global Intention Localization and Local Movement Refinement 
// Written by Shaoshuai Shi 
// All Rights Reserved


#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "knn_gpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_batch", &knn_batch, "knn_batch");
    m.def("knn_batch_mlogk", &knn_batch_mlogk, "knn_batch_mlogk");
}
