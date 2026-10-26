package org.fiarc.strictjava;

/** Exact scale selection and float32 rounding helpers. */
final class ExactArithmetic {
    private ExactArithmetic() {}
    static long round(long a,int k){
        if(k<=0)return a<<-k;
        if(k>64)return 0;
        if(k==64)return Long.compareUnsigned(a,Long.MIN_VALUE)>0?1:0;
        long q=a>>>k,r=a&(-1L>>>(64-k)),half=1L<<(k-1);
        return q+((Long.compareUnsigned(r,half)>0||(r==half&&(q&1)!=0))?1:0);
    }
    static int pack(int sign,long n,int base){
        if(n==0)return sign;
        int h=63-Long.numberOfLeadingZeros(n),e=h+base;
        if(e< -126)return sign|(int)round(n,-149-base);
        long m=round(n,h-23);
        if(m==(1L<<24)){m>>>=1;e++;}
        if(e>127)return sign|0x7f800000;
        return sign|((e+127)<<23)|((int)m&0x7fffff);
    }
    static int exponent(int bits) {
        // Smallest e for which max(abs(x), 2^-40) <= 4095 * 2^e.
        int a=Math.max(bits&0x7fffffff,0x2b800000),be=(a>>>23)-150;
        int m=(a&0x7fffff)|0x800000;
        int e=be+12;
        if((long)m>((long)4095<<12))e++;
        return e;
    }
}
