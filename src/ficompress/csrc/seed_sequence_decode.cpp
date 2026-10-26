#include <torch/extension.h>
std::vector<torch::Tensor> seed_sequence_decode_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,bool,bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("decode_frame",&seed_sequence_decode_cuda);}
