#include <torch/extension.h>
std::vector<torch::Tensor> seed_encode_cuda(torch::Tensor,torch::Tensor,int64_t,bool,bool);
std::vector<torch::Tensor> seed_encode_reciprocal_cuda(torch::Tensor,torch::Tensor,int64_t,torch::Tensor);
torch::Tensor seed_prefix_cuda(torch::Tensor);
torch::Tensor seed_probabilities_cdf_cuda(torch::Tensor);
torch::Tensor seed_cdf17_cuda(torch::Tensor);
std::vector<torch::Tensor> seed_encode17_cuda(torch::Tensor,torch::Tensor,int64_t);
std::vector<torch::Tensor> seed_decode17_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);
std::vector<torch::Tensor> seed_bounded_cuda(torch::Tensor,torch::Tensor,double,double);
std::vector<torch::Tensor> seed_bounded_even_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,double,double);
std::vector<torch::Tensor> seed_decode_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);
torch::Tensor seed_pack_cuda(torch::Tensor,torch::Tensor,torch::Tensor,int64_t);
torch::Tensor seed_checker_adjust_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("encode",&seed_encode_cuda,pybind11::arg("q"),pybind11::arg("cdf"),pybind11::arg("streams"),pybind11::arg("scalar")=false,pybind11::arg("with_bypass_counts")=false);
 m.def("encode_reciprocal",&seed_encode_reciprocal_cuda);
 m.def("prefix8",&seed_prefix_cuda);m.def("decode",&seed_decode_cuda);m.def("pack",&seed_pack_cuda);
 m.def("probabilities_cdf",&seed_probabilities_cdf_cuda);
 m.def("cdf17",&seed_cdf17_cuda);m.def("encode17",&seed_encode17_cuda);m.def("decode17",&seed_decode17_cuda);
 m.def("bounded",&seed_bounded_cuda);
 m.def("bounded_after_even",&seed_bounded_even_cuda);
 m.def("checker_adjust",&seed_checker_adjust_cuda);
}
