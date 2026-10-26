// strict_v2 resident entropy and recurrent state. No frame is independent.
#pragma OPENCL FP_CONTRACT OFF
inline uint fadd(uint a,uint b){
#ifdef NVIDIA_DP4A
 float r;asm("add.rn.f32 %0, %1, %2;":"=f"(r):"f"(as_float(a)),"f"(as_float(b)));return as_uint(r);
#else
 return soft_add(a,b);
#endif
}
inline uint fmul(uint a,uint b){
#ifdef NVIDIA_DP4A
 float r;asm("mul.rn.f32 %0, %1, %2;":"=f"(r):"f"(as_float(a)),"f"(as_float(b)));return as_uint(r);
#else
 return soft_mul(a,b);
#endif
}
inline uint fdiv(uint a,uint b){
#ifdef NVIDIA_DP4A
 float r;asm("div.rn.f32 %0, %1, %2;":"=f"(r):"f"(as_float(a)),"f"(as_float(b)));return as_uint(r);
#else
 return soft_div(a,b);
#endif
}
inline uint norm(uint x,uint mean,uint inv){return fmul(fadd(x,mean^0x80000000u),inv);}
inline uint iscale(uint x,uint tol){return max(0x358637bdu,fmul(x&0x7fffffffu,tol));}
inline uint restored(uint p,int q,uint esc,__global const uint2*f,__global uint*err){
 uint a=p&0x7fffffffu;if(a>=0x7f800000u||q< -32768||q>32767){atomic_or(err,32768u);return 0;}
 if(q==-32768)return esc;if(!q)return p;uint sign=a?(p&0x80000000u):0;if(!a)a=0x3f800000u;
 int ex=a>>23;uint pm=(a&0x7fffffu)|(ex?0x800000u:0);uint2 factor=f[q+32767];
 return packed(sign,(ulong)pm*factor.x,(ex?ex-150:-149)+(int)factor.y-31);
}
__kernel void pixels(__global const uint*a,__global const uint*b,__global const uint*c,__global const uint*d,
 __global uint*out,__global uint*second,__global uint*err,
 int mode,int count,int h,int w,int history,uint mean,uint std,uint inv,uint gain,int stride,int lattice,uint tol,int cli){
 int i=get_global_id(0),n=h*w;if(i>=count)return;uint v=0;
 if(mode==0)v=mean;
 else if(mode==1)v=fadd(fmul(fadd(a[i],b[(history-1)*n+i]),std),mean);
 else if(mode==2)v=norm(a[i],mean,inv);
 else if(mode==3){v=norm(a[i],mean,inv);out[n+i]=b[i]?0x3f800000u:0;uint delta=b[i]?fadd(c[i],a[i]^0x80000000u):0;
  uint feature=fdiv(delta,iscale(a[i],tol));out[2*n+i]=feature;if((feature&0x7fffffffu)>=0x7f800000u)atomic_or(err,65536u);
  if(lattice)out[3*n+i]=(i/w%stride==0&&i%w%stride==0)?0x3f800000u:0;
 }else if(mode==4)v=c[i]?d[i]:fadd(a[i],fmul(b[i],iscale(a[i],tol)));
 else if(mode==5){int y=i/w,x=i%w;v=a[i];second[i]=0;if(y%stride==0&&x%stride==0){int j=y/stride*((w+stride-1)/stride)+x/stride;second[i]=b[j];v=cli?cli_restore(a[i],(int)b[j],c[j],(__global const uint2*)d,err):restored(a[i],(int)b[j],c[j],(__global const uint2*)d,err);}}
 else if(mode==6)v=cli?cli_restore(a[i],(int)b[i],c[i],(__global const uint2*)d,err):restored(a[i],(int)b[i],c[i],(__global const uint2*)d,err);
 else if(mode==7){v=a[i];int y=i/w,x=i%w;if(((x+y)&1)&&!c[i]){int ids[4]={y?i-w:-1,x?i-1:-1,x+1<w?i+1:-1,y+1<h?i+w:-1};uint sum=0;
  for(int k=0;k<4;k++){uint delta=ids[k]<0?0:fadd(b[ids[k]],a[ids[k]]^0x80000000u);uint term=fmul(delta,d[k]);sum=k?fadd(sum,term):term;}v=fadd(v,fmul(gain,sum));}}
 else if(mode==8)v=i>=(history-1)*n?a[i]:fadd(a[i],a[(history-1)*n+i%n]^0x80000000u);
 if((v&0x7fffffffu)>=0x7f800000u)atomic_or(err,131072u);out[i]=v;
}
inline int lab(int q){return q>=-3&&q<=3?q+3:7;}
inline int value_at(__global const int*q,__global const int*s,int y,int x,int h,int w,int seedstage){
 if(y<0||x<0||y>=h||x>=w)return 3;int i=y*w+x;return lab(!seedstage&&s[i]?s[i]:q[i]);
}
__kernel void cdf(__global const uint*logits,__global const int*q,__global const int*seed,__global const uint*table,
 __global const uint*offset,__global uint*out,__global uint*err,int h,int w,int plane,int stride,int odd,int seedstage,int fullw){
 int j=get_global_id(0);if(j>=(h*w+1-odd)/2)return;int i=2*j+odd;if(!(w&1)){int yy=j/(w/2);i=yy*w+2*(j%(w/2))+((yy&1)^odd);}int y=i/w,x=i%w,at=j*9;out[at]=0;
 if(!seedstage&&!odd&&seed[i]){for(int k=0;k<8;k++)out[at+k+1]=k>=3?4096:0;return;}
 int ctx=45;if(odd){int ids[4]={value_at(q,seed,y-1,x,h,w,seedstage),value_at(q,seed,y,x-1,h,w,seedstage),value_at(q,seed,y,x+1,h,w,seedstage),value_at(q,seed,y+1,x,h,w,seedstage)};int count=0,vote=0;for(int k=0;k<4;k++){count+=ids[k]!=3;vote+=(ids[k]>=4&&ids[k]<=6)-(ids[k]<=2);}ctx=count*9+vote+4;}
 int pixel=y*stride*fullw+x*stride,values[8],mx=-2147483647,best=0;
 for(int k=0;k<8;k++){uint v=fadd(logits[(k+1)*plane+pixel],offset[ctx*8+k]);if((v&0x7fffffffu)>=0x4b800000u){atomic_or(err,16u);return;}values[k]=grid_integer(v,-4);if(values[k]>mx){mx=values[k];best=k;}}
 uint weights[8],freq[8],sum=0,total=0;for(int k=0;k<8;k++){weights[k]=table[min(256,mx-values[k])];sum+=weights[k];}
 for(int k=0;k<8;k++){freq[k]=weights[k]*4088u/sum+1;total+=freq[k];}freq[best]+=4096-total;
 for(int k=0;k<8;k++)out[at+k+1]=out[at+k]+freq[k];
}
__kernel void rans(__global const uchar*body,__global const uint*offsets,__global uint*state,__global const uint*cdf,
 __global int*labels,__global uint*counts,__global uint*err,int n,int ns,int bytes,int terminal){
 uint s=get_global_id(0);if(s>=ns)return;counts[s]=0;uint begin=offsets[s],end=offsets[s+1],r=state[2*s],at=state[2*s+1];
 if(begin>end||end>bytes||at>end-begin||r<8388608u||r>=2147483648u){atomic_or(err,32u);return;}
 uint count=0;for(uint i=(uint)((ulong)n*s/ns);i<(uint)((ulong)n*(s+1)/ns);i++){
  uint slot=r&4095,k=0,index=9*i;while(k<8&&slot>=cdf[index+k+1])k++;
  if(k>=8||cdf[index]!=0||cdf[index+8]!=4096){atomic_or(err,64u);return;}
  uint low=cdf[index+k],high=cdf[index+k+1];if(high<=low||high>4096||slot<low){atomic_or(err,128u);return;}
  labels[i]=k==7?-32768:(int)k-3;count+=k==7;r=(high-low)*(r>>12)+slot-low;
  while(r<8388608u){if(at>=end-begin){atomic_or(err,256u);return;}r=(r<<8)|body[begin+at++];}
 }state[2*s]=r;state[2*s+1]=at;counts[s]=count;if(terminal&&(at!=end-begin||r!=8388608u))atomic_or(err,512u);
}
__kernel void prefix(__global const uint*counts,__global uint*counter,__global uint*out,__global uint*err,int n,uint available,int terminal){
 if(get_global_id(0))return;uint at=counter[0];if(at>available){atomic_or(err,1024u);return;}
 for(int i=0;i<n;i++){out[i]=at;if(counts[i]>available-at){atomic_or(err,1024u);return;}at+=counts[i];}
 counter[0]=at;if(terminal&&at!=available)atomic_or(err,2048u);
}
__kernel void symbols(__global const int*labels,__global const int*rare,__global const uint*prefix,__global int*q,__global uint*err,int n,int ns,int w,int odd,uint available){
 uint s=get_global_id(0);if(s>=ns)return;uint at=prefix[s];
 for(uint j=(uint)((ulong)n*s/ns);j<(uint)((ulong)n*(s+1)/ns);j++){int v=labels[j];if(v==-32768){if(at>=available){atomic_or(err,4096u);return;}v=rare[at++];}
  if(v< -32768||v>32767){atomic_or(err,8192u);return;}int i=2*j+odd;if(!(w&1)){int y=j/(w/2);i=y*w+2*(j%(w/2))+((y&1)^odd);}q[i]=v;
 }
}
__kernel void escape_count(__global const int*q,__global uint*counts,int n){
 __local uint s[256];uint lane=get_local_id(0),i=get_global_id(0);s[lane]=i<n&&q[i]==-32768;barrier(CLK_LOCAL_MEM_FENCE);
 for(int d=128;d;d/=2){if(lane<d)s[lane]+=s[lane+d];barrier(CLK_LOCAL_MEM_FENCE);}if(!lane)counts[get_group_id(0)]=s[0];
}
__kernel void escape_scatter(__global const int*q,__global const uint*esc,__global const uint*prefix,__global uint*out,__global uint*err,int n,uint available){
 __local uint s[256];uint lane=get_local_id(0),i=get_global_id(0),flag=i<n&&q[i]==-32768;s[lane]=flag;barrier(CLK_LOCAL_MEM_FENCE);
 for(int d=1;d<256;d*=2){uint v=lane>=d?s[lane-d]:0;barrier(CLK_LOCAL_MEM_FENCE);s[lane]+=v;barrier(CLK_LOCAL_MEM_FENCE);}
 if(i<n){out[i]=0;if(flag){uint at=prefix[get_group_id(0)]+s[lane]-1;if(at>=available)atomic_or(err,16384u);else out[i]=esc[at];}}
}
