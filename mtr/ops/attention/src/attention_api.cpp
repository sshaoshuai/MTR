#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "attention_func.h"
#include "attention_func_v2.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_weight_computation_wrapper", &attention_weight_computation_wrapper,
        "attention weight computation forward.");
    m.def("attention_weight_computation_grad_wrapper", &attention_weight_computation_grad_wrapper,
        "attention weight computation backward.");
    m.def("attention_value_computation_wrapper", &attention_value_computation_wrapper,
        "attention result computation forward.");
    m.def("attention_value_computation_grad_wrapper", &attention_value_computation_grad_wrapper,
        "attention result computation backward.");

    m.def("attention_weight_computation_wrapper_v2", &attention_weight_computation_wrapper_v2,
        "attention weight computation forward.");
    m.def("attention_weight_computation_grad_wrapper_v2", &attention_weight_computation_grad_wrapper_v2,
        "attention weight computation backward.");
    m.def("attention_value_computation_wrapper_v2", &attention_value_computation_wrapper_v2,
        "attention result computation forward.");
    m.def("attention_value_computation_grad_wrapper_v2", &attention_value_computation_grad_wrapper_v2,
        "attention result computation backward.");
}