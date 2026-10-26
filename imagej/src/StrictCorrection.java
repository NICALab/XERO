package org.fiarc.strictjava;
import java.nio.*;

/** Normative strict_v2 correction: unsigned integer multiply, RN-even, no exp(). */
public final class StrictCorrection {
    final int[] mantissas=new int[65535],exponents=new int[65535];
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
    static int activationExponent(int bits){
        // Smallest exponent e for which max(abs(x), 2^-40) <= 4095*2^e.
        int a=Math.max(bits&0x7fffffff,0x2b800000),be=(a>>>23)-150;
        int m=(a&0x7fffff)|0x800000;
        int e=be+12;
        if((long)m>((long)4095<<12))e++;
        return e;
    }
    StrictCorrection(byte[] bytes){
        FiARCCodec.require(bytes.length==240,"Correction coefficients size");
        ByteBuffer b=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int[] cm=new int[30],ce=new int[30];
        for(int i=0;i<30;i++){cm[i]=b.getInt();ce[i]=b.getInt();FiARCCodec.require(cm[i]<0&&Math.abs((long)ce[i])<=2048,"Invalid correction coefficient");}
        for(int q=-32767;q<=32767;q++){
            int a=Math.abs(q),offset=q<0?15:0,e=0;long m=0x80000000L;
            for(int i=0;i<15;i++)if((a&(1<<i))!=0){
                long p=m*Integer.toUnsignedLong(cm[offset+i]);int k=p<0?32:31;
                m=round(p,k);e+=ce[offset+i]+k-31;
                if(m==(1L<<32)){m>>>=1;e++;}
            }
            mantissas[q+32767]=(int)m;exponents[q+32767]=e;
        }
    }
    int restore(int prediction,int q){
        int a=prediction&0x7fffffff;
        FiARCCodec.require(a<0x7f800000&&q>=-32767&&q<=32767,"Invalid correction input");
        if(q==0)return prediction;
        int sign=a==0?0:prediction&0x80000000;if(a==0)a=0x3f800000;
        int ex=a>>>23,base=ex==0?-149:ex-150,m=(a&0x7fffff)|(ex==0?0:0x800000),ix=q+32767;
        int out=pack(sign,(long)m*Integer.toUnsignedLong(mantissas[ix]),base+exponents[ix]-31);
        FiARCCodec.require((out&0x7fffffff)<0x7f800000,"Correction overflow");return out;
    }
}
