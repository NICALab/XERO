#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

constexpr uint32_t LOW=1u<<23;

__global__ void prefix8_kernel(const int32_t* f,int32_t* c,int64_t n){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 int v=0;c[9*i]=0;
 #pragma unroll
 for(int j=0;j<8;++j){v+=f[8*i+j];c[9*i+j+1]=v;}
}

template<int K> __global__ void probabilities_cdf_kernel(const float* p,int32_t* c,int64_t n){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 int f[K],sum=0,winner=0;float best=-1.f;
 #pragma unroll
 for(int j=0;j<K;++j){float v=p[K*i+j];if(v>best){best=v;winner=j;}
  f[j]=__float2int_rd(v*float(4096-K))+1;sum+=f[j];}
 f[winner]+=4096-sum;sum=0;c[(K+1)*i]=0;
 #pragma unroll
 for(int j=0;j<K;++j){sum+=f[j];c[(K+1)*i+j+1]=sum;}
}

// One warp per stream: distribute streams over SMs without adding stream
// headers, and prefetch 32 independent CDF entries before serial state updates.
template<bool FAST=false> __global__ void enc_warp_kernel(const int16_t* q,const int32_t* cdf,int64_t n,int streams,int slot_bytes,uint8_t* slots,int32_t* lengths,int32_t* status,int32_t* bypass_counts,const int64_t* divisors=nullptr){
 int stream=blockIdx.x,lane=threadIdx.x;
 int64_t begin=n*stream/streams,end=n*(stream+1)/streams;int pos=slot_bytes;uint32_t state=LOW;
 uint8_t* dst=slots+int64_t(stream)*slot_bytes;int bypass_count=0;
 for(int64_t base=end;base>begin;base-=32){
  int64_t i=base-1-lane;uint32_t start=0,freq=1;
  if(i>=begin){int label=(q[i]>=-3&&q[i]<=3)?q[i]+3:7;bypass_count+=label==7;start=cdf[i*9+label];freq=cdf[i*9+label+1]-start;}
  uint32_t reciprocal=0,shift=0;
  if constexpr(FAST){if(freq>=1&&freq<=4096){uint64_t entry=uint64_t(divisors[freq]);reciprocal=uint32_t(entry);shift=uint32_t(entry>>32);}}
  int count=int(base-begin<32?base-begin:32);
  for(int j=0;j<count;++j){
   uint32_t s=__shfl_sync(0xffffffff,start,j),f=__shfl_sync(0xffffffff,freq,j);
   if(!f||f>4096||s>4096||s+f>4096){if(lane==0)status[stream]=1;return;}
   uint32_t limit=((LOW>>12)<<8)*f;
   while(state>=limit){if(pos<=4){if(lane==0)status[stream]=2;return;}--pos;if(lane==0)dst[pos]=state&255;state>>=8;}
   uint32_t quotient;
   if constexpr(FAST){
    uint32_t m=__shfl_sync(0xffffffff,reciprocal,j),k=__shfl_sync(0xffffffff,shift,j);
    // Renormalization guarantees state <2^19*f; encoding with start+f<=4096
    // preserves state<2^31. The reciprocal table is exact over this domain.
    quotient=f==1?state:(__umulhi(state,m)>>k);
   }else quotient=state/f;
   state=(quotient<<12)+(state-quotient*f)+s;
  }
 }
 if(bypass_counts){for(int shift=16;shift;shift/=2)bypass_count+=__shfl_down_sync(0xffffffff,bypass_count,shift);if(lane==0)bypass_counts[stream]=bypass_count;}
 if(lane==0){pos-=4;for(int j=0;j<4;++j)dst[pos+j]=(state>>(8*j))&255;lengths[stream]=slot_bytes-pos;}
}

__global__ void enc_kernel(const int16_t* q,const int32_t* cdf,int64_t n,int streams,int slot_bytes,uint8_t* slots,int32_t* lengths,int32_t* status,int32_t* bypass_counts){
 int stream=blockIdx.x*blockDim.x+threadIdx.x;if(stream>=streams)return;
 int64_t begin=n*stream/streams,end=n*(stream+1)/streams;int pos=slot_bytes;uint32_t state=LOW;
 uint8_t* dst=slots+int64_t(stream)*slot_bytes;int bypass_count=0;
 for(int64_t i=end;i-->begin;){
  int label=(q[i]>=-3&&q[i]<=3)?q[i]+3:7;bypass_count+=label==7;
  uint32_t start=cdf[i*9+label],freq=cdf[i*9+label+1]-start;
  if(!freq||start+freq>4096){status[stream]=1;return;}
  uint32_t limit=((LOW>>12)<<8)*freq;
  while(state>=limit){if(pos<=4){status[stream]=2;return;}dst[--pos]=state&255;state>>=8;}
  uint32_t quotient=state/freq;state=(quotient<<12)+(state-quotient*freq)+start;
 }
 if(pos<4){status[stream]=3;return;}pos-=4;
 for(int j=0;j<4;++j)dst[pos+j]=(state>>(8*j))&255;
 lengths[stream]=slot_bytes-pos;if(bypass_counts)bypass_counts[stream]=bypass_count;
}

__global__ void pack_kernel(const uint8_t* slots,const int32_t* lengths,const int64_t* offsets,int slot_bytes,uint8_t* out){
 int s=blockIdx.x;int length=lengths[s];
 for(int j=threadIdx.x;j<length;j+=blockDim.x)out[offsets[s]+j]=slots[int64_t(s)*slot_bytes+slot_bytes-length+j];
}

__global__ void dec_kernel(const uint8_t* data,const int32_t* lengths,const int64_t* offsets,const int32_t* cdf,int64_t n,int streams,int16_t* out,int32_t* status){
 int s=blockIdx.x;if(threadIdx.x!=0||s>=streams)return;
 int64_t begin=n*s/streams,end=n*(s+1)/streams;int len=lengths[s],at=4;
 if(len<4){status[s]=1;return;}
 const uint8_t* src=data+offsets[s];uint32_t state=0;
 for(int j=0;j<4;++j)state|=uint32_t(src[j])<<(8*j);
 for(int64_t i=begin;i<end;++i){
  uint32_t slot=state&4095;int label=0;
  while(label<7&&slot>=uint32_t(cdf[i*9+label+1]))++label;
  uint32_t start=cdf[i*9+label],freq=cdf[i*9+label+1]-start;
  if(!freq||slot<start||slot>=start+freq){status[s]=2;return;}
  out[i]=label==7?-32768:label-3;state=freq*(state>>12)+slot-start;
  while(state<LOW){if(at>=len){status[s]=3;return;}state=(state<<8)|src[at++];}
 }
 if(at!=len||state!=LOW)status[s]=4;
}

void check_cdf(torch::Tensor cdf,int64_t n){
 TORCH_CHECK(cdf.is_cuda()&&cdf.is_contiguous()&&cdf.scalar_type()==torch::kInt32,"CUDA int32 contiguous CDF required");
 TORCH_CHECK(cdf.dim()==2&&cdf.size(0)==n&&cdf.size(1)==9,"N x9 CDF required");
}

std::vector<torch::Tensor> seed_encode_cuda(torch::Tensor q,torch::Tensor cdf,int64_t streams,bool scalar,bool with_bypass_counts){
 TORCH_CHECK(q.is_cuda()&&q.is_contiguous()&&q.scalar_type()==torch::kInt16,"CUDA int16 symbols required");
 c10::cuda::CUDAGuard guard(q.device());int64_t n=q.numel();check_cdf(cdf,n);
 TORCH_CHECK(q.device()==cdf.device()&&streams>0&&streams<=65536,"invalid device/streams");
 int slot=2*((n+streams-1)/streams)+8;
 auto slots=torch::empty({streams,slot},q.options().dtype(torch::kUInt8));
 auto lengths=torch::zeros({streams},q.options().dtype(torch::kInt32));auto status=torch::zeros_like(lengths);
 auto counts=with_bypass_counts?torch::zeros_like(lengths):torch::Tensor();int32_t* bp=with_bypass_counts?counts.data_ptr<int32_t>():nullptr;
 if(scalar)enc_kernel<<<(streams+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(q.data_ptr<int16_t>(),cdf.data_ptr<int32_t>(),n,streams,slot,slots.data_ptr<uint8_t>(),lengths.data_ptr<int32_t>(),status.data_ptr<int32_t>(),bp);
 else enc_warp_kernel<false><<<streams,32,0,at::cuda::getCurrentCUDAStream()>>>(q.data_ptr<int16_t>(),cdf.data_ptr<int32_t>(),n,streams,slot,slots.data_ptr<uint8_t>(),lengths.data_ptr<int32_t>(),status.data_ptr<int32_t>(),bp);
 C10_CUDA_KERNEL_LAUNCH_CHECK();if(with_bypass_counts)return {slots,lengths,status,counts};return {slots,lengths,status};
}

std::vector<torch::Tensor> seed_encode_reciprocal_cuda(torch::Tensor q,torch::Tensor cdf,int64_t streams,torch::Tensor divisors){
 TORCH_CHECK(q.is_cuda()&&q.is_contiguous()&&q.scalar_type()==torch::kInt16,"CUDA int16 symbols required");
 c10::cuda::CUDAGuard guard(q.device());int64_t n=q.numel();check_cdf(cdf,n);
 TORCH_CHECK(q.device()==cdf.device()&&streams>0&&streams<=65536,"invalid device/streams");
 TORCH_CHECK(divisors.device()==q.device()&&divisors.is_contiguous()&&divisors.scalar_type()==torch::kInt64&&divisors.numel()==4097,"fixed4097-entry same-device int64 division table required");
 int slot=2*((n+streams-1)/streams)+8;auto slots=torch::empty({streams,slot},q.options().dtype(torch::kUInt8));
 auto lengths=torch::zeros({streams},q.options().dtype(torch::kInt32));auto status=torch::zeros_like(lengths);auto counts=torch::zeros_like(lengths);
 enc_warp_kernel<true><<<streams,32,0,at::cuda::getCurrentCUDAStream()>>>(q.data_ptr<int16_t>(),cdf.data_ptr<int32_t>(),n,streams,slot,slots.data_ptr<uint8_t>(),lengths.data_ptr<int32_t>(),status.data_ptr<int32_t>(),counts.data_ptr<int32_t>(),divisors.data_ptr<int64_t>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {slots,lengths,status,counts};
}

torch::Tensor seed_pack_cuda(torch::Tensor slots,torch::Tensor lengths,torch::Tensor offsets,int64_t total){
 c10::cuda::CUDAGuard guard(slots.device());TORCH_CHECK(total>=0,"invalid total");
 auto out=torch::empty({total},slots.options());
 pack_kernel<<<lengths.numel(),128,0,at::cuda::getCurrentCUDAStream()>>>(slots.data_ptr<uint8_t>(),lengths.data_ptr<int32_t>(),offsets.data_ptr<int64_t>(),slots.size(1),out.data_ptr<uint8_t>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

std::vector<torch::Tensor> seed_decode_cuda(torch::Tensor data,torch::Tensor lengths,torch::Tensor offsets,torch::Tensor cdf){
 c10::cuda::CUDAGuard guard(data.device());int64_t n=cdf.size(0);check_cdf(cdf,n);
 TORCH_CHECK(data.is_cuda()&&data.scalar_type()==torch::kUInt8&&data.is_contiguous(),"invalid payload");
 TORCH_CHECK(lengths.is_cuda()&&lengths.scalar_type()==torch::kInt32&&lengths.is_contiguous(),"invalid lengths");
 TORCH_CHECK(offsets.is_cuda()&&offsets.scalar_type()==torch::kInt64&&offsets.is_contiguous()&&offsets.numel()==lengths.numel(),"invalid offsets");
 TORCH_CHECK(lengths.device()==data.device()&&offsets.device()==data.device()&&cdf.device()==data.device(),"device mismatch");
 int streams=lengths.numel();TORCH_CHECK(streams>0&&streams<=65536,"invalid streams");
 auto out=torch::empty({n},data.options().dtype(torch::kInt16));auto status=torch::zeros_like(lengths);
 dec_kernel<<<streams,32,0,at::cuda::getCurrentCUDAStream()>>>(data.data_ptr<uint8_t>(),lengths.data_ptr<int32_t>(),offsets.data_ptr<int64_t>(),cdf.data_ptr<int32_t>(),n,streams,out.data_ptr<int16_t>(),status.data_ptr<int32_t>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,status};
}

torch::Tensor seed_prefix_cuda(torch::Tensor f){
 TORCH_CHECK(f.is_cuda()&&f.is_contiguous()&&f.scalar_type()==torch::kInt32&&f.dim()==2&&f.size(1)==8,"N x8 contiguous CUDA int32 frequencies required");
 c10::cuda::CUDAGuard guard(f.device());auto c=torch::empty({f.size(0),9},f.options());
 if(f.size(0))prefix8_kernel<<<(f.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(f.data_ptr<int32_t>(),c.data_ptr<int32_t>(),f.size(0));
 C10_CUDA_KERNEL_LAUNCH_CHECK();return c;
}

torch::Tensor seed_probabilities_cdf_cuda(torch::Tensor p){
 TORCH_CHECK(p.is_cuda()&&p.is_contiguous()&&p.scalar_type()==torch::kFloat32&&p.dim()==2&&p.size(1)==8,"N x8 contiguous CUDA float32 probabilities required");
 c10::cuda::CUDAGuard guard(p.device());auto c=torch::empty({p.size(0),9},p.options().dtype(torch::kInt32));
 if(p.size(0))probabilities_cdf_kernel<8><<<(p.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(),c.data_ptr<int32_t>(),p.size(0));
 C10_CUDA_KERNEL_LAUNCH_CHECK();return c;
}

__global__ void enc17_kernel(const int16_t* q,const int32_t* cdf,int64_t n,int streams,int slot_bytes,uint8_t* slots,int32_t* lengths,int32_t* status){
 int s=blockIdx.x,lane=threadIdx.x;int64_t begin=n*s/streams,end=n*(s+1)/streams;
 int pos=slot_bytes;uint32_t state=LOW;uint8_t* dst=slots+int64_t(s)*slot_bytes;
 for(int64_t base=end;base>begin;base-=32){
  int64_t i=base-1-lane;uint32_t start=0,freq=1;bool invalid=false;
  if(i>=begin){int label=q[i];invalid=label<0||label>16;if(!invalid){start=cdf[i*18+label];freq=cdf[i*18+label+1]-start;}}
  if(__any_sync(0xffffffff,invalid)){if(lane==0)status[s]=1;return;}
  int count=int(base-begin<32?base-begin:32);
  for(int j=0;j<count;++j){
   uint32_t a=__shfl_sync(0xffffffff,start,j),f=__shfl_sync(0xffffffff,freq,j);
   if(!f||a+f>4096){if(lane==0)status[s]=2;return;}
   while(state>=((LOW>>12)<<8)*f){if(pos<=4){if(lane==0)status[s]=3;return;}--pos;if(lane==0)dst[pos]=state&255;state>>=8;}
   uint32_t div=state/f;state=(div<<12)+state-div*f+a;
  }
 }
 if(lane==0){pos-=4;for(int j=0;j<4;++j)dst[pos+j]=(state>>(8*j))&255;lengths[s]=slot_bytes-pos;}
}
__global__ void dec17_kernel(const uint8_t* data,const int32_t* lengths,const int64_t* offsets,const int32_t* cdf,int64_t n,int streams,int16_t* out,int32_t* status){
 int s=blockIdx.x;if(threadIdx.x)return;int len=lengths[s],at=4;if(len<4){status[s]=1;return;}
 const uint8_t* src=data+offsets[s];uint32_t state=0;for(int j=0;j<4;++j)state|=uint32_t(src[j])<<(8*j);
 for(int64_t i=n*s/streams;i<n*(s+1)/streams;++i){
  uint32_t slot=state&4095;int label=0;while(label<16&&slot>=uint32_t(cdf[i*18+label+1]))++label;
  uint32_t a=cdf[i*18+label],f=cdf[i*18+label+1]-a;
  if(!f||slot<a||slot>=a+f){status[s]=2;return;}
  out[i]=label;state=f*(state>>12)+slot-a;
  while(state<LOW){if(at>=len){status[s]=3;return;}state=(state<<8)|src[at++];}
 }
 if(at!=len||state!=LOW)status[s]=4;
}
torch::Tensor seed_cdf17_cuda(torch::Tensor p){
 TORCH_CHECK(p.is_cuda()&&p.is_contiguous()&&p.scalar_type()==torch::kFloat32&&p.dim()==2&&p.size(1)==17,"N17 float32 CUDA probabilities required");
 c10::cuda::CUDAGuard guard(p.device());auto c=torch::empty({p.size(0),18},p.options().dtype(torch::kInt32));
 if(p.size(0))probabilities_cdf_kernel<17><<<(p.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(),c.data_ptr<int32_t>(),p.size(0));
 C10_CUDA_KERNEL_LAUNCH_CHECK();return c;
}
std::vector<torch::Tensor> seed_encode17_cuda(torch::Tensor q,torch::Tensor cdf,int64_t streams){
 TORCH_CHECK(q.is_cuda()&&q.is_contiguous()&&q.scalar_type()==torch::kInt16,"CUDA int16 labels required");
 c10::cuda::CUDAGuard guard(q.device());int64_t n=q.numel();
 TORCH_CHECK(cdf.device()==q.device()&&cdf.is_contiguous()&&cdf.scalar_type()==torch::kInt32&&cdf.dim()==2&&cdf.size(0)==n&&cdf.size(1)==18,"N18 same-device CUDA int32 CDF required");
 TORCH_CHECK(streams>0&&streams<=65536,"invalid streams");int slot=2*((n+streams-1)/streams)+8;
 auto slots=torch::empty({streams,slot},q.options().dtype(torch::kUInt8));auto lengths=torch::zeros({streams},q.options().dtype(torch::kInt32));auto status=torch::zeros_like(lengths);
 enc17_kernel<<<streams,32,0,at::cuda::getCurrentCUDAStream()>>>(q.data_ptr<int16_t>(),cdf.data_ptr<int32_t>(),n,streams,slot,slots.data_ptr<uint8_t>(),lengths.data_ptr<int32_t>(),status.data_ptr<int32_t>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {slots,lengths,status};
}
std::vector<torch::Tensor> seed_decode17_cuda(torch::Tensor data,torch::Tensor lengths,torch::Tensor offsets,torch::Tensor cdf){
 TORCH_CHECK(data.is_cuda()&&data.scalar_type()==torch::kUInt8&&data.is_contiguous(),"CUDA byte payload required");c10::cuda::CUDAGuard guard(data.device());
 TORCH_CHECK(cdf.device()==data.device()&&cdf.is_contiguous()&&cdf.scalar_type()==torch::kInt32&&cdf.dim()==2&&cdf.size(1)==18,"N18 CUDA CDF required");
 TORCH_CHECK(lengths.device()==data.device()&&offsets.device()==data.device()&&lengths.is_contiguous()&&offsets.is_contiguous()&&lengths.scalar_type()==torch::kInt32&&offsets.scalar_type()==torch::kInt64&&lengths.numel()==offsets.numel(),"invalid stream tables");
 int streams=lengths.numel();TORCH_CHECK(streams>0&&streams<=65536,"invalid streams");
 auto out=torch::empty({cdf.size(0)},data.options().dtype(torch::kInt16));auto status=torch::zeros_like(lengths);
 dec17_kernel<<<streams,32,0,at::cuda::getCurrentCUDAStream()>>>(data.data_ptr<uint8_t>(),lengths.data_ptr<int32_t>(),offsets.data_ptr<int64_t>(),cdf.data_ptr<int32_t>(),cdf.size(0),streams,out.data_ptr<int16_t>(),status.data_ptr<int32_t>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,status};
}

// Preserve detector-domain FP64 comparisons. Only already-in-bound pixels skip
// transcendental math; no FP16/relative-tolerance relaxation is permitted here.
__global__ void bounded_kernel(const float* source,const float* prediction,int64_t n,double step,double tolerance,int16_t* q,float* rec){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 float sf=source[i],pf=prediction[i];double s=sf,p=pf,bound=tolerance*fabs(s);
 if(fabs(s-p)<=bound){q[i]=0;rec[i]=pf;return;}
 double anchor=p==0.?1.:fabs(p);double raw=nearbyint(log(fabs(s)/anchor)/step);
 int code=isfinite(raw)&&raw>-32768.&&raw<=32767.?int(raw):0;
 float r=code==0?pf:float((p<0.?-anchor:anchor)*exp(double(code)*step));
 if(!isfinite(r)||fabs(s-double(r))>bound){code=-32768;r=sf;}
 q[i]=int16_t(code);rec[i]=r;
}
std::vector<torch::Tensor> seed_bounded_cuda(torch::Tensor s,torch::Tensor p,double step,double tolerance){
 TORCH_CHECK(s.is_cuda()&&p.device()==s.device()&&s.scalar_type()==torch::kFloat32&&p.scalar_type()==torch::kFloat32&&s.is_contiguous()&&p.is_contiguous()&&s.sizes()==p.sizes(),"same-shape contiguous CUDA float32 input required");
 TORCH_CHECK(step>0&&tolerance>0,"positive step/tolerance required");c10::cuda::CUDAGuard guard(s.device());
 auto q=torch::empty_like(s,s.options().dtype(torch::kInt16));auto r=torch::empty_like(s);
 if(s.numel())bounded_kernel<<<(s.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(s.data_ptr<float>(),p.data_ptr<float>(),s.numel(),step,tolerance,q.data_ptr<int16_t>(),r.data_ptr<float>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {q,r};
}

// The first phase and seed predictions are unchanged by even refinement.
// Reuse their exact earlier q/reconstruction; recompute only nonseed odd sites.
__global__ void bounded_even_kernel(const float* source,const float* prediction,const bool* mask,const int16_t* old_q,const float* old_r,int64_t n,int h,int w,double step,double tolerance,int16_t* q,float* rec){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 if(mask[i]||!(((i%w)+(i/w)%h)&1)){q[i]=old_q[i];rec[i]=old_r[i];return;}
 float sf=source[i],pf=prediction[i];double s=sf,p=pf,bound=tolerance*fabs(s);
 if(fabs(s-p)<=bound){q[i]=0;rec[i]=pf;return;}
 double anchor=p==0.?1.:fabs(p);double raw=nearbyint(log(fabs(s)/anchor)/step);
 int code=isfinite(raw)&&raw>-32768.&&raw<=32767.?int(raw):0;
 float r=code==0?pf:float((p<0.?-anchor:anchor)*exp(double(code)*step));
 if(!isfinite(r)||fabs(s-double(r))>bound){code=-32768;r=sf;}
 q[i]=int16_t(code);rec[i]=r;
}
std::vector<torch::Tensor> seed_bounded_even_cuda(torch::Tensor s,torch::Tensor p,torch::Tensor mask,torch::Tensor old_q,torch::Tensor old_r,double step,double tolerance){
 TORCH_CHECK(s.is_cuda()&&s.scalar_type()==torch::kFloat32&&s.is_contiguous()&&s.dim()>=2,"contiguous CUDA float32 source required");
 TORCH_CHECK(p.device()==s.device()&&p.scalar_type()==torch::kFloat32&&p.is_contiguous()&&p.sizes()==s.sizes(),"invalid prediction");
 TORCH_CHECK(mask.device()==s.device()&&mask.scalar_type()==torch::kBool&&mask.is_contiguous()&&mask.sizes()==s.sizes(),"invalid seed mask");
 TORCH_CHECK(old_q.device()==s.device()&&old_q.scalar_type()==torch::kInt16&&old_q.is_contiguous()&&old_q.sizes()==s.sizes(),"invalid previous symbols");
 TORCH_CHECK(old_r.device()==s.device()&&old_r.scalar_type()==torch::kFloat32&&old_r.is_contiguous()&&old_r.sizes()==s.sizes(),"invalid previous reconstruction");
 TORCH_CHECK(step>0&&tolerance>0&&s.size(-1)>0&&s.size(-2)>0&&s.size(-1)<=INT_MAX&&s.size(-2)<=INT_MAX,"invalid quantizer/shape");
 c10::cuda::CUDAGuard guard(s.device());auto q=torch::empty_like(old_q);auto r=torch::empty_like(old_r);
 if(s.numel())bounded_even_kernel<<<(s.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(s.data_ptr<float>(),p.data_ptr<float>(),mask.data_ptr<bool>(),old_q.data_ptr<int16_t>(),old_r.data_ptr<float>(),s.numel(),s.size(-2),s.size(-1),step,tolerance,q.data_ptr<int16_t>(),r.data_ptr<float>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {q,r};
}

// One thread computes the neighbor context once, then handles all8 channels.
// Only FP32 addition changes logits; existing softmax/CDF and rANS are untouched.
__global__ void checker_adjust_kernel(const float* logits,const bool* mask,const int16_t* sq,const int16_t* q,const float* offsets,float* out,int n,int h,int w){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int x=i%w,y=(i/w)%h,context=45;
 if((x+y)&1){
  int count=0,votes=0;int delta[4]={-w,w,-1,1};bool valid[4]={y>0,y<h-1,x>0,x<w-1};
  #pragma unroll
  for(int j=0;j<4;j++)if(valid[j]){
   int at=i+delta[j];int v=mask[at]?sq[at]:q[at];int label=(v>=-3&&v<=3)?v+3:7;
   count+=label!=3;votes+=(label>=4&&label<=6)?1:((label<=2)?-1:0);
  }
  context=count*9+votes+4;
 }
 const float4* a=reinterpret_cast<const float4*>(logits)+2*i;
 const float4* b=reinterpret_cast<const float4*>(offsets)+2*context;
 float4* dst=reinterpret_cast<float4*>(out)+2*i;
 #pragma unroll
 for(int k=0;k<2;k++){
  float4 v=a[k],z=b[k];dst[k]=make_float4(v.x+z.x,v.y+z.y,v.z+z.z,v.w+z.w);
 }
}
torch::Tensor seed_checker_adjust_cuda(torch::Tensor logits,torch::Tensor mask,torch::Tensor sq,torch::Tensor q,torch::Tensor offsets){
 TORCH_CHECK(q.is_cuda()&&q.scalar_type()==torch::kInt16&&q.is_contiguous()&&q.dim()==4,"contiguous BCHW CUDA int16 symbols required");
 TORCH_CHECK(mask.device()==q.device()&&sq.device()==q.device()&&logits.device()==q.device()&&offsets.device()==q.device(),"checkerboard device mismatch");
 TORCH_CHECK(mask.scalar_type()==torch::kBool&&sq.scalar_type()==torch::kInt16&&mask.is_contiguous()&&sq.is_contiguous()&&mask.sizes()==q.sizes()&&sq.sizes()==q.sizes(),"checkerboard mask/seed shape mismatch");
 TORCH_CHECK(logits.scalar_type()==torch::kFloat32&&logits.is_contiguous()&&logits.dim()==5&&logits.size(4)==8&&logits.sizes().slice(0,4)==q.sizes(),"BCHW8 contiguous float32 logits required");
 TORCH_CHECK(offsets.scalar_type()==torch::kFloat32&&offsets.is_contiguous()&&offsets.dim()==2&&offsets.size(0)==46&&offsets.size(1)==8,"46x8 float32 offsets required");
 TORCH_CHECK(q.size(2)>0&&q.size(3)>0&&q.numel()<=2147483647,"unsupported checkerboard image size");
 c10::cuda::CUDAGuard guard(q.device());
 // Contiguous views can still start at an unaligned storage offset.
 if(reinterpret_cast<uintptr_t>(logits.data_ptr<float>())%16)logits=logits.clone();
 if(reinterpret_cast<uintptr_t>(offsets.data_ptr<float>())%16)offsets=offsets.clone();
 auto out=torch::empty_like(logits);
 if(q.numel())checker_adjust_kernel<<<(q.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(logits.data_ptr<float>(),mask.data_ptr<bool>(),sq.data_ptr<int16_t>(),q.data_ptr<int16_t>(),offsets.data_ptr<float>(),out.data_ptr<float>(),int(q.numel()),int(q.size(2)),int(q.size(3)));
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
