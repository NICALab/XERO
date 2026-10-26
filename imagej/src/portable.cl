// Common integer-emulated binary32 FMA. No vendor-specific interpolation path.
inline uint portable_fma(uint a,uint b,uint c){
 uint aa=a&0x7fffffffu,bb=b&0x7fffffffu,cc=c&0x7fffffffu,sign=(a^b)&0x80000000u,cs=c&0x80000000u;
 if(aa>=0x7f800000u||bb>=0x7f800000u||cc>=0x7f800000u)return 0x7fc00000u;
 if(!aa||!bb)return soft_add(sign,c);
 ulong product=(ulong)mant(a)*mant(b);int pe=base(a)+base(b);
 if(!cc)return packed(sign,product,pe);
 ulong cm=mant(c);int ce=base(c),top=max(pe+high(product),ce+high(cm));
 ulong av=jam(product,top-62-pe),bv=jam(cm,top-62-ce),sum;
 if(sign==cs)sum=av+bv;else if(av<bv){sum=bv-av;sign=cs;}else sum=av-bv;
 return sum?packed(sign,sum,top-62):0;
}
inline uint cli_restore(uint p,int q,uint esc,__global const uint2*f,__global uint*err){
 uint a=p&0x7fffffffu;if(a>=0x7f800000u||q< -32768||q>32767){atomic_or(err,32768u);return 0;}
 if(q==-32768)return esc;if(!q)return p;uint sign=a?(p&0x80000000u):0;if(!a)a=0x3f800000u;
 int ex=a>>23;uint pm=mant(a);uint2 f64=f[q+32767];uint fe=(f64.y>>20)&2047,fm=(f64.y&0xfffffu)|(fe?0x100000u:0);
 if(fe==2047){atomic_or(err,32768u);return 0;}if(!(fm|f64.x))return sign;
 ulong v=(ulong)pm*f64.x,b=(ulong)pm*fm;uint lo=(uint)v,mid=(uint)(v>>32)+(uint)b,hi=(uint)(b>>32)+(mid<(uint)(v>>32));
 int top=hi?64+31-clz(hi):high(((ulong)mid<<32)|lo),shift=max(0,top-52);
 ulong product=((ulong)mid<<32)|lo;
 if(shift){product=(product>>shift)|((ulong)hi<<(64-shift));uint guard=(lo>>(shift-1))&1,sticky=(lo&((1u<<(shift-1))-1))!=0;if(guard&&(sticky||(product&1)))product++;}
 return packed(sign,product,(ex?ex-150:-149)+(fe?(int)fe-1075:-1074)+shift);
}
__kernel void portable_up(__global const uint*x,__global uint*out,__global const uint4*axes,int c,int h,int w,int hh,int ww,int quarter_grid){
 int i=get_global_id(0);if(i>=c*hh*ww)return;int ch=i/(hh*ww),y=i/ww%hh,z=i%ww,off=ch*h*w;
 uint4 ay=axes[y],ax=axes[hh+z];uint p=x[off+ay.x*w+ax.x],q=x[off+ay.x*w+ax.y],r=x[off+ay.y*w+ax.x],s=x[off+ay.y*w+ax.y];
 if(quarter_grid<0){uint a=soft_add(soft_mul(ax.w,p),soft_mul(ax.z,q)),b=soft_add(soft_mul(ax.w,r),soft_mul(ax.z,s));out[i]=soft_add(soft_mul(ay.w,a),soft_mul(ay.z,b));return;}
 if(quarter_grid&&max(max(p&0x7fffffffu,q&0x7fffffffu),max(r&0x7fffffffu,s&0x7fffffffu))<=0x43000000u){
  float4 v=(float4)(as_float(p),as_float(q),as_float(r),as_float(s))*4096.f;int4 j=convert_int4(v);
  if(all(v==convert_float4(j))){int dx=(int)(as_float(ax.z)*4.f),dy=(int)(as_float(ay.z)*4.f),n=(4-dy)*((4-dx)*j.x+dx*j.y)+dy*((4-dx)*j.z+dx*j.w);if(n){out[i]=as_uint((float)n*(1.f/65536.f));return;}}
 }
 uint a=portable_fma(ax.z,q,soft_mul(ax.w,p)),b=portable_fma(ax.w,r,soft_mul(ax.z,s));out[i]=portable_fma(ay.w,a,soft_mul(ay.z,b));
}
__kernel void history_shift(__global uint*hist,__global const uint*rec,int n,int count,int valid){
 int i=get_global_id(0);if(i>=n)return;for(int c=0;c+1<count;c++)hist[c*n+i]=valid?hist[(c+1)*n+i]:rec[i];hist[(count-1)*n+i]=rec[i];
}
