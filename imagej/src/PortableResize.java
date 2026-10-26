package org.fiarc.strictjava;

/** The CLI/browser binary32 FMA contract, including odd target dimensions. */
public strictfp final class PortableResize {
    static long mant(int a){return (a&0x7fffff)|((a&0x7f800000)!=0?0x800000:0);}
    static int base(int a){int e=(a>>>23)&255;return e==0?-149:e-150;}
    static int high(long a){return 63-Long.numberOfLeadingZeros(a);}
    static long jam(long a,int k){if(k<=0)return a<<-k;if(k>=64)return a==0?0:1;return (a>>>k)|((a&(-1L>>>(64-k)))!=0?1:0);}
    static int fma(int a,int b,int c){
        int aa=a&0x7fffffff,bb=b&0x7fffffff,cc=c&0x7fffffff,sign=(a^b)&0x80000000,cs=c&0x80000000;
        if(aa>=0x7f800000||bb>=0x7f800000||cc>=0x7f800000)return 0x7fc00000;
        if(aa==0||bb==0)return Float.floatToRawIntBits(Float.intBitsToFloat(sign)+Float.intBitsToFloat(c));
        long product=mant(a)*mant(b);int pe=base(a)+base(b);
        if(cc==0)return ExactArithmetic.pack(sign,product,pe);
        long cm=mant(c);int ce=base(c),top=Math.max(pe+high(product),ce+high(cm));
        long av=jam(product,top-62-pe),bv=jam(cm,top-62-ce),sum;
        if(sign==cs)sum=av+bv;else if(Long.compareUnsigned(av,bv)<0){sum=bv-av;sign=cs;}else sum=av-bv;
        return sum==0?0:ExactArithmetic.pack(sign,sum,top-62);
    }
    static int[] axis(int src,int dst){int[] a=new int[dst*4];float scale=(float)src/dst;for(int j=0;j<dst;j++){
        float v=Math.max(0f,(float)((double)scale*(j+.5f)-.5));int lo=(int)v;float delta=v-lo;
        a[j*4]=lo;a[j*4+1]=Math.min(lo+1,src-1);a[j*4+2]=Float.floatToRawIntBits(delta);a[j*4+3]=Float.floatToRawIntBits(1f-delta);
    }return a;}
    static float bilinear(float p,float q,float r,float s,int dx,int ex,int dy,int ey){
        int a=fma(dx,Float.floatToRawIntBits(q),Float.floatToRawIntBits(Float.intBitsToFloat(ex)*p));
        int b=fma(ex,Float.floatToRawIntBits(r),Float.floatToRawIntBits(Float.intBitsToFloat(dx)*s));
        return Float.intBitsToFloat(fma(ey,a,Float.floatToRawIntBits(Float.intBitsToFloat(dy)*Float.intBitsToFloat(b))));
    }
    static float[] resize(float[] x,int c,int h,int w,int hh,int ww){
        int[] ys=axis(h,hh),xs=axis(w,ww);float[] out=new float[c*hh*ww];
        for(int ch=0;ch<c;ch++)for(int y=0;y<hh;y++)for(int z=0;z<ww;z++){
            int a=ch*h*w+ys[y*4]*w,b=ch*h*w+ys[y*4+1]*w,lo=xs[z*4],hi=xs[z*4+1];
            out[(ch*hh+y)*ww+z]=bilinear(x[a+lo],x[a+hi],x[b+lo],x[b+hi],xs[z*4+2],xs[z*4+3],ys[y*4+2],ys[y*4+3]);
        }return out;
    }
}
