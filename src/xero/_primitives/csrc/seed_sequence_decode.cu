#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

constexpr uint32_t LOW=1u<<23;
__global__ void decode_frame_kernel(const uint8_t* data,int64_t size,const int32_t* lengths,const int64_t* offsets,
 const int32_t* cdf,int64_t n,int streams,int32_t* states,int32_t* cursors,bool first,bool last,int16_t* out,int32_t* status){
 int s=blockIdx.x;if(threadIdx.x)return;
 int len=lengths[s],at=first?4:cursors[s];int64_t off=offsets[s];
 if(len<4||off<0||off>size||len>size-off||at<4||at>len){status[s]=1;return;}
 const uint8_t* src=data+off;uint32_t state=uint32_t(states[s]);
 if(first){state=0;for(int j=0;j<4;++j)state|=uint32_t(src[j])<<(8*j);}
 if(state<LOW||state>=(1u<<31)){status[s]=2;return;}
 for(int64_t i=n*s/streams;i<n*(s+1)/streams;++i){
  uint32_t slot=state&4095;int label=0;
  while(label<7&&slot>=uint32_t(cdf[9*i+label+1]))++label;
  uint32_t start=cdf[9*i+label],freq=cdf[9*i+label+1]-start;
  if(!freq||freq>4096||start>4096||start+freq>4096||slot<start||slot>=start+freq){status[s]=3;return;}
  out[i]=label==7?-32768:label-3;state=freq*(state>>12)+slot-start;
  while(state<LOW){if(at>=len){status[s]=4;return;}state=(state<<8)|src[at++];}
 }
 states[s]=int32_t(state);cursors[s]=at;
 if(last&&(at!=len||state!=LOW))status[s]=5;
}

std::vector<torch::Tensor> seed_sequence_decode_cuda(torch::Tensor data,torch::Tensor lengths,torch::Tensor offsets,
 torch::Tensor cdf,torch::Tensor states,torch::Tensor cursors,bool first,bool last){
 TORCH_CHECK(data.is_cuda()&&data.is_contiguous()&&data.scalar_type()==torch::kUInt8&&data.dim()==1,"CUDA byte payload required");
 c10::cuda::CUDAGuard guard(data.device());
 TORCH_CHECK(cdf.device()==data.device()&&cdf.is_contiguous()&&cdf.scalar_type()==torch::kInt32&&cdf.dim()==2&&cdf.size(1)==9,"same-device N x9 CDF required");
 for(auto x:{lengths,states,cursors})TORCH_CHECK(x.device()==data.device()&&x.is_contiguous()&&x.scalar_type()==torch::kInt32&&x.dim()==1&&x.numel()==lengths.numel(),"same-device stream int32 vectors required");
 TORCH_CHECK(offsets.device()==data.device()&&offsets.is_contiguous()&&offsets.scalar_type()==torch::kInt64&&offsets.dim()==1&&offsets.numel()==lengths.numel(),"same-device int64 offsets required");
 int64_t streams=lengths.numel();TORCH_CHECK(streams>0&&streams<=65536,"invalid stream count");
 auto out=torch::empty({cdf.size(0)},data.options().dtype(torch::kInt16));auto status=torch::zeros_like(lengths);
 decode_frame_kernel<<<streams,32,0,at::cuda::getCurrentCUDAStream()>>>(data.data_ptr<uint8_t>(),data.numel(),lengths.data_ptr<int32_t>(),offsets.data_ptr<int64_t>(),
  cdf.data_ptr<int32_t>(),cdf.size(0),streams,states.data_ptr<int32_t>(),cursors.data_ptr<int32_t>(),first,last,out.data_ptr<int16_t>(),status.data_ptr<int32_t>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,status};
}
