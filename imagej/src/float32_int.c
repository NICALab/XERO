/* Normative binary32 scalar arithmetic using integer operations only.
 * Finite inputs, RN-even, gradual underflow, signed zero. Overflow gives inf;
 * invalid or NaN gives canonical 0x7fc00000 (codec must reject nonfinite).
 * No float, libm, contraction, hardware rounding mode or subnormal dependency.
 * This is a reference building block, NOT yet wired into the complete codec.
 */
typedef unsigned int u32;
typedef int i32;
typedef unsigned long long u64;
static i32 high(u64 a){return 63-__builtin_clzll(a);}
static u64 jam(u64 a,i32 k){
 if(k<=0)return a<<(-k);
 if(k>=64)return a!=0;
 return (a>>k)|((a&((1ull<<k)-1))!=0);
}
static u64 round_even(u64 a,i32 k){
 if(k<=0)return a<<(-k);
 if(k>64)return 0;
 if(k==64)return a>(1ull<<63);
 u64 q=a>>k,r=a&((1ull<<k)-1),half=1ull<<(k-1);
 return q+(r>half||(r==half&&(q&1)));
}
static u32 packed(u32 sign,u64 n,i32 base){
 if(!n)return sign;
 i32 top=high(n),ex=top+base;
 if(ex< -126)return sign|(u32)round_even(n,-149-base);
 u64 m=round_even(n,top-23);
 if(m==(1ull<<24)){m>>=1;ex++;}
 if(ex>127)return sign|0x7f800000u;
 return sign|((u32)(ex+127)<<23)|((u32)m&0x7fffffu);
}
static u32 mant(u32 a){return (a&0x7fffffu)|((a&0x7f800000u)?0x800000u:0);}
static i32 base(u32 a){i32 e=(a>>23)&255;return e?e-150:-149;}

u32 soft_add(u32 a,u32 b){
 u32 aa=a&0x7fffffffu,bb=b&0x7fffffffu,sa=a&0x80000000u,sb=b&0x80000000u;
 if(aa>=0x7f800000u||bb>=0x7f800000u)return 0x7fc00000u;
 if(!aa&&!bb)return sa&sb;
 if(!aa)return b;if(!bb)return a;
 u32 am=mant(a),bm=mant(b);i32 ae=base(a),be=base(b);
 i32 at=ae+high(am),bt=be+high(bm),top=at>bt?at:bt;
 // 63-bit aligned significands; cancellation is exact when close. When
 // separated by many bits, a sticky low bit retains all rounding information.
 u64 av=jam(am,top-62-ae),bv=jam(bm,top-62-be),m;
 if(sa==sb)m=av+bv;
 else if(av>=bv)m=av-bv;
 else {m=bv-av;sa=sb;}
 if(!m)return 0;
 return packed(sa,m,top-62);
}
u32 soft_mul(u32 a,u32 b){
 u32 aa=a&0x7fffffffu,bb=b&0x7fffffffu,s=(a^b)&0x80000000u;
 if(aa>=0x7f800000u||bb>=0x7f800000u)return 0x7fc00000u;
 return packed(s,(u64)mant(a)*mant(b),base(a)+base(b));
}
u32 soft_div(u32 a,u32 b){
 u32 aa=a&0x7fffffffu,bb=b&0x7fffffffu,s=(a^b)&0x80000000u;
 if(aa>=0x7f800000u||bb>=0x7f800000u||(!aa&&!bb))return 0x7fc00000u;
 if(!bb)return s|0x7f800000u;if(!aa)return s;
 u32 am=mant(a),bm=mant(b);i32 ae=base(a),be=base(b);
 i32 ash=23-high(am),bsh=23-high(bm);am<<=ash;bm<<=bsh;ae-=ash;be-=bsh;
 u64 n=(u64)am<<32,q=n/bm,r=n%bm;
 return packed(s,q|(r!=0),ae-be-32);
}
void scalar_ops(const u32 *a,const u32 *b,u32 *out,u32 n){
 for(u32 i=0;i<n;i++){
  out[i*3]=soft_add(a[i],b[i]);out[i*3+1]=soft_mul(a[i],b[i]);out[i*3+2]=soft_div(a[i],b[i]);
 }
}

void binary_op(const u32 *a,const u32 *b,u32 *out,u32 n,u32 op,u32 na,u32 nb){
 for(u32 i=0;i<n;i++){
  u32 x=a[na==1?0:i],y=b[nb==1?0:i];
  out[i]=op==0?soft_add(x,y):op==1?soft_add(x,y^0x80000000u):op==2?soft_mul(x,y):soft_div(x,y);
 }
}
// NCHW, fixed factor two, pool sums ((UL+UR)+LL)+LR with RN32 at every op.
void resize_bits(const u32 *x,u32 *out,u32 c,u32 h,u32 w,u32 up){
 u32 hh=up?h*2:h/2,ww=up?w*2:w/2;
 const u32 weights[4]={0,0x3e800000u,0x3f000000u,0x3f400000u};
 const u32 complement[4]={0x3f800000u,0x3f400000u,0x3f000000u,0x3e800000u};
 for(u32 ch=0;ch<c;ch++)for(u32 y=0;y<hh;y++)for(u32 z=0;z<ww;z++){
  u32 o=(ch*hh+y)*ww+z;
  if(!up){u32 k=(ch*h+y*2)*w+z*2;
   out[o]=soft_mul(soft_add(soft_add(soft_add(x[k],x[k+1]),x[k+w]),x[k+w+1]),0x3e800000u);
  }else{
   u32 yy=y?2*y-1:0,zz=z?2*z-1:0,y0=yy/4,x0=zz/4,dy=yy%4,dx=zz%4;
   u32 y1=y0+1<h?y0+1:h-1,x1=x0+1<w?x0+1:w-1,k=ch*h*w;
   u32 a=soft_add(soft_mul(complement[dx],x[k+y0*w+x0]),soft_mul(weights[dx],x[k+y0*w+x1]));
   u32 b=soft_add(soft_mul(complement[dx],x[k+y1*w+x0]),soft_mul(weights[dx],x[k+y1*w+x1]));
   out[o]=soft_add(soft_mul(complement[dy],a),soft_mul(weights[dy],b));
  }
 }
}

static i32 grid_integer(u32 bits,i32 exponent){
 u32 a=bits&0x7fffffffu,m=mant(bits);i32 shift=exponent-base(bits);
 u64 q=round_even(m,shift);return bits>>31?-(i32)q:(i32)q;
}
void quantize_bits(const u32 *x,i32 *q,u32 n,i32 exponent){
 for(u32 i=0;i<n;i++)q[i]=grid_integer(x[i],exponent);
}
static u32 integer_round(u32 a){
 u32 magnitude=a&0x7fffffffu,sign=a&0x80000000u;
 if(magnitude>=0x4b000000u)return a;
 if(magnitude<0x3f000000u)return sign;
 i32 e=(i32)(magnitude>>23)-127;
 return packed(sign,round_even(mant(a),23-e),0);
}
static u32 dot_bias(i32 q,i32 exponent,u32 bias){
 u32 am=q<0?(u32)(-(long long)q):(u32)q,bm=mant(bias),sign=q<0?0x80000000u:0,bs=bias&0x80000000u;
 if(!am&&!bm)return 0;
 i32 bx=base(bias),at=am?exponent+high(am):-10000,bt=bm?bx+high(bm):-10000,top=at>bt?at:bt;
 u64 av=am?jam(am,top-62-exponent):0,bv=bm?jam(bm,top-62-bx):0,m;
 if(sign==bs)m=av+bv;
 else if(av>=bv)m=av-bv;
 else {m=bv-av;sign=bs;}
 if(!m)return 0;
 i32 shift=high(m)>52?high(m)-52:0;
 m=round_even(m,shift);
 return packed(sign,m,top-62+shift);
}
void dot_epilogue(const i32 *dots,const i32 *exponents,const u32 *bias,const i32 *table,u32 *out,u32 n,u32 plane,u32 kind){
 for(u32 i=0;i<n;i++){
  u32 c=i/plane,v=dot_bias(dots[i],exponents[c],bias[c]);
  if(kind==1){
   u32 magnitude=v&0x7fffffffu;
   if(v>>31&&magnitude>=0x41800000u)v=0;
   else if(magnitude<0x41800000u){
    i32 q=grid_integer(v,-12),value=table[q+65536];
    v=packed(value<0?0x80000000u:0,value<0?(u64)(-(long long)value):(u64)value,-12);
   }
  }else if(kind==2&&(v>>31||(v&0x7fffffffu)==0))v=soft_mul(v,0x3dcccccdu);
  if(kind)v=soft_mul(integer_round(soft_mul(v,0x45800000u)),0x39800000u);
  out[i]=v;
 }
}
