#ifdef NVIDIA_DP4A
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#pragma OPENCL FP_CONTRACT OFF
// Exact signed packed integer dots. Portable fallback; NVIDIA optional DP4A.
inline int idot(uint a,uint b,int sum){
#ifdef NVIDIA_DP4A
  int r;asm("dp4a.s32.s32 %0, %1, %2, %3;":"=r"(r):"r"(a),"r"(b),"r"(sum));return r;
#else
  char4 x=as_char4(a),y=as_char4(b);
  return sum+(int)x.s0*y.s0+(int)x.s1*y.s1+(int)x.s2*y.s2+(int)x.s3*y.s3;
#endif
}
__kernel void maxparts(__global const float*x,__global const float*y,__global float*out,int n,int na,__global uint*err){
  __local float a[256];int lane=get_local_id(0);float v=0x1p-40f;
  for(int i=get_global_id(0);i<n;i+=get_global_size(0)){float q=i<na?x[i]:y[i-na];if(!isfinite(q))atomic_or(err,1u);v=fmax(v,fabs(q));}
  a[lane]=v;barrier(CLK_LOCAL_MEM_FENCE);
  for(int k=128;k;k/=2){if(lane<k)a[lane]=fmax(a[lane],a[lane+k]);barrier(CLK_LOCAL_MEM_FENCE);}
  if(lane==0)out[get_group_id(0)]=a[0];
}
__kernel void scalevalue(__global const float*x,__global float*out,int n){
  __local float a[256];int lane=get_local_id(0);float v=0x1p-40f;
  for(int i=lane;i<n;i+=256)v=fmax(v,x[i]);a[lane]=v;barrier(CLK_LOCAL_MEM_FENCE);
  for(int k=128;k;k/=2){if(lane<k)a[lane]=fmax(a[lane],a[lane+k]);barrier(CLK_LOCAL_MEM_FENCE);}
  // 4095 is strictly smaller than 2^12: ceil(max/4095) starts at e-11,
  // and needs another doubling just below the next power of two.
  if(lane==0){int e=(int)((as_uint(a[0])>>23)&255)-127;float s=ldexp(1.f,e-11);if(s*4095.f<a[0])s*=2.f;out[0]=s;}
}
__kernel void pack(__global const float*x,__global const float*y,__global const float*scale,__global uint2*out,int plane,int ci,int ca,__global uint*err){
  int i=get_global_id(0),ci4=(ci+3)/4;if(i>=plane*ci4)return;int pixel=i/ci4,c0=(i%ci4)*4;char4 lo=(char4)(0),hi=(char4)(0);
  for(int j=0;j<4;j++)if(c0+j<ci){int c=c0+j;float v=c<ca?x[c*plane+pixel]:y[(c-ca)*plane+pixel];int q=grid_integer(as_uint(v),(int)(as_uint(scale[0])>>23)-127);if(abs(q)>4095){atomic_or(err,2u);q=0;}int high=(q+64)>>7;hi[j]=(char)high;lo[j]=(char)(q-128*high);}out[i]=(uint2)(as_uint(lo),as_uint(hi));
}
__kernel __attribute__((reqd_work_group_size(8,8,1))) void conv(
  __global const uint2*x,__global const uint*weights,__global float*out,
  __global const float*scale,__global const int*exponent,__global const float*bias,__global const int*lut,
  int h,int w,int ci4,int co,int kh,int kw,int act){
  __local uint2 ax[256];__local uint bw[128];
  int lx=get_local_id(0),ly=get_local_id(1),lane=lx+8*ly,plane=h*w,ktotal=ci4*kh*kw,pixel0=get_group_id(0)*32,channel0=get_group_id(1)*16;
  int slo[8]={0},shi[8]={0};
  for(int kb=0;kb<ktotal;kb+=8){
    for(int j=0;j<4;j++){int a=lane+j*64,pixel=pixel0+a/8,k=kb+a%8;uint2 v=(uint2)(0);
      if(pixel<plane&&k<ktotal){int kk=k/ci4,yy=pixel/w+kk/kw-kh/2,xx=pixel%w+kk%kw-kw/2;if(yy>=0&&xx>=0&&yy<h&&xx<w)v=x[(yy*w+xx)*ci4+k%ci4];}ax[a]=v;}
    for(int j=0;j<2;j++){int b=lane+j*64,ch=channel0+b/8,k=kb+b%8;bw[b]=ch<co&&k<ktotal?weights[ch*ktotal+k]:0;}
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int k=0;k<8;k++)for(int m=0;m<4;m++){uint2 a=ax[(lx+m*8)*8+k];for(int j=0;j<2;j++){uint b=bw[(ly*2+j)*8+k];int i=m*2+j;slo[i]=idot(a.x,b,slo[i]);shi[i]=idot(a.y,b,shi[i]);}}
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for(int m=0;m<4;m++)for(int j=0;j<2;j++){int pixel=pixel0+lx+m*8,ch=channel0+ly*2+j,i=m*2+j;
    if(pixel<plane&&ch<co){int dot=slo[i]+128*shi[i];if(act==-1){out[ch*plane+pixel]=as_float(dot);continue;}float v=as_float(dot_bias(dot,exponent[ch]+(int)(as_uint(scale[0])>>23)-127,as_uint(bias[ch])));
      if(act==1){if(v<=-16.f)v=0;else if(v<16.f)v=(float)lut[convert_int_rte(v*4096.f)+65536]/4096.f;}
      else if(act==2&&v<=0.f)v=as_float(fmul(as_uint(v),0x3dcccccdu));if(act)v=as_float(fmul(integer_round(fmul(as_uint(v),0x45800000u)),0x39800000u));out[ch*plane+pixel]=v;}
  }
}
__kernel void pool(__global const float*x,__global float*out,int c,int h,int w){
  int i=get_global_id(0),hh=h/2,ww=w/2;if(i>=c*hh*ww)return;int co=i/(hh*ww),y=(i%(hh*ww))/ww,z=i%ww,k=co*h*w+2*y*w+2*z;out[i]=as_float(fmul(fadd(fadd(fadd(as_uint(x[k]),as_uint(x[k+1])),as_uint(x[k+w])),as_uint(x[k+w+1])),0x3e800000u));
}
__kernel void up(__global const float*x,__global float*out,int c,int h,int w){
  int i=get_global_id(0),hh=h*2,ww=w*2;if(i>=c*hh*ww)return;int co=i/(hh*ww),y=(i%(hh*ww))/ww,z=i%ww,y0=y?(y-1)/2:0,x0=z?(z-1)/2:0,y1=min(y0+1,h-1),x1=min(x0+1,w-1),off=co*h*w;
  float dy=y?((y&1)?.25f:.75f):0.f,dx=z?((z&1)?.25f:.75f):0.f;
  uint a=fadd(fmul(as_uint(1.f-dx),as_uint(x[off+y0*w+x0])),fmul(as_uint(dx),as_uint(x[off+y0*w+x1]))),b=fadd(fmul(as_uint(1.f-dx),as_uint(x[off+y1*w+x0])),fmul(as_uint(dx),as_uint(x[off+y1*w+x1])));out[i]=as_float(fadd(fmul(as_uint(1.f-dy),a),fmul(as_uint(dy),b)));
}
__kernel void history_features(__global const float*x,__global float*out,int plane,int c,float mean,float inv){
  int i=get_global_id(0);if(i>=plane*c)return;float v=(x[i]-mean)*inv,last=(x[(c-1)*plane+i%plane]-mean)*inv;out[i]=i/plane==c-1?v:v-last;
}
#ifdef NVIDIA_DP4A
inline int4 mma(uint4 a,uint2 b,int4 c){
 int4 d;asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
 :"=r"(d.s0),"=r"(d.s1),"=r"(d.s2),"=r"(d.s3):"r"(a.s0),"r"(a.s1),"r"(a.s2),"r"(a.s3),"r"(b.s0),"r"(b.s1),"r"(c.s0),"r"(c.s1),"r"(c.s2),"r"(c.s3));return d;
}
inline uint2 gathered(__global const uint2*x,int pixel,int k,int h,int w,int ci4,int kh,int kw){
 int total=ci4*kh*kw;if(pixel>=h*w||k>=total)return (uint2)(0);int kk=k/ci4,yy=pixel/w+kk/kw-kh/2,xx=pixel%w+kk%kw-kw/2;
 return yy>=0&&xx>=0&&yy<h&&xx<w?x[(yy*w+xx)*ci4+k%ci4]:(uint2)(0);
}
__kernel __attribute__((reqd_work_group_size(32,1,1))) void conv_mma(
 __global const uint2*x,__global const uint*weights,__global float*out,__global const float*scale,
 __global const int*exponent,__global const float*bias,__global const int*lut,int h,int w,int ci4,int co,int kh,int kw,int act){
 int lane=get_local_id(0),g=lane/4,t=lane%4,plane=h*w,p0=get_group_id(0)*16,c0=get_group_id(1)*8,total=ci4*kh*kw;
 int4 lo=(int4)(0),hi=(int4)(0);
 for(int kb=0;kb<total;kb+=8){
  uint2 a0=gathered(x,p0+g,kb+t,h,w,ci4,kh,kw),a1=gathered(x,p0+g+8,kb+t,h,w,ci4,kh,kw);
  uint2 a2=gathered(x,p0+g,kb+t+4,h,w,ci4,kh,kw),a3=gathered(x,p0+g+8,kb+t+4,h,w,ci4,kh,kw);
  uint2 b=(uint2)(c0+g<co&&kb+t<total?weights[(c0+g)*total+kb+t]:0,c0+g<co&&kb+t+4<total?weights[(c0+g)*total+kb+t+4]:0);
  lo=mma((uint4)(a0.x,a1.x,a2.x,a3.x),b,lo);hi=mma((uint4)(a0.y,a1.y,a2.y,a3.y),b,hi);
 }
 for(int j=0;j<4;j++){int pixel=p0+g+(j/2)*8,ch=c0+2*t+j%2;if(pixel>=plane||ch>=co)continue;
  int dot=lo[j]+128*hi[j];int ex=exponent[ch]+(int)(as_uint(scale[0])>>23)-127;
  // Same explicitly rounded binary64 epilogue as normative strict_v2.
  double cs=ldexp(1.0,ex);float v=convert_float_rte((double)dot*cs+(double)bias[ch]);
  if(act==1){if(v<=-16.f)v=0;else if(v<16.f)v=(float)lut[grid_integer(as_uint(v),-12)+65536]/4096.f;}
  else if(act==2&&v<=0.f)v=as_float(fmul(as_uint(v),0x3dcccccdu));
  if(act)v=as_float(fmul(integer_round(fmul(as_uint(v),0x45800000u)),0x39800000u));out[ch*plane+pixel]=v;
 }
}
#endif
