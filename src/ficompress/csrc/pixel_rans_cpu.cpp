#include <cstdint>
#include <vector>
static constexpr uint32_t LOW=1u<<23, SCALE=12, MASK=4095;
// CDFs are known before decoding a spatial group. No within-group dependency.
extern "C" int64_t encode_pixel_rans(const int16_t* q,int64_t n,const uint16_t* cdf,uint8_t* out,int64_t capacity){
    uint32_t state=LOW;std::vector<uint8_t> bytes;bytes.reserve(n/4+16);
    for(int64_t i=n-1;i>=0;--i){
        int s=q[i]>=-3&&q[i]<=3?q[i]+3:7;
        uint32_t start=cdf[i*9+s],freq=cdf[i*9+s+1]-start;
        if(!freq)return -1;
        while(state>=((LOW>>SCALE)<<8)*freq){bytes.push_back(state&255);state>>=8;}
        state=((state/freq)<<SCALE)+(state%freq)+start;
    }
    if(int64_t(bytes.size())+4>capacity)return -2;
    for(int j=0;j<4;++j)out[j]=(state>>(8*j))&255;
    for(size_t j=0;j<bytes.size();++j)out[j+4]=bytes[bytes.size()-j-1];
    return bytes.size()+4;
}
extern "C" int decode_pixel_rans(const uint8_t* in,int64_t size,int64_t n,const uint16_t* cdf,const int16_t* escapes,int64_t ne,int16_t* q){
    if(size<4)return -1;
    uint32_t state=0;for(int j=0;j<4;++j)state|=uint32_t(in[j])<<(8*j);
    int64_t at=4,ex=0;
    for(int64_t i=0;i<n;++i){
        uint32_t slot=state&MASK;int s=0;while(s<7&&slot>=cdf[i*9+s+1])++s;
        uint32_t start=cdf[i*9+s],freq=cdf[i*9+s+1]-start;
        if(!freq)return -2;
        if(s==7){if(ex>=ne)return -3;q[i]=escapes[ex++];}else q[i]=s-3;
        state=freq*(state>>SCALE)+slot-start;
        while(state<LOW){if(at>=size)return -4;state=(state<<8)|in[at++];}
    }
    return at==size&&ex==ne&&state==LOW?0:-5;
}

extern "C" int64_t encode_categorical17(const int16_t* q,int64_t n,const uint16_t* cdf,uint8_t* out,int64_t capacity){
    uint32_t state=LOW;std::vector<uint8_t> bytes;bytes.reserve(n+16);
    for(int64_t i=n-1;i>=0;--i){
        int s=q[i];if(s<0||s>16)return -1;
        uint32_t start=cdf[i*18+s],freq=cdf[i*18+s+1]-start;if(!freq)return -2;
        while(state>=((LOW>>12)<<8)*freq){bytes.push_back(state&255);state>>=8;}
        state=((state/freq)<<12)+state%freq+start;
    }
    if(int64_t(bytes.size())+4>capacity)return -3;
    for(int j=0;j<4;++j)out[j]=(state>>(8*j))&255;
    for(size_t j=0;j<bytes.size();++j)out[j+4]=bytes[bytes.size()-1-j];return bytes.size()+4;
}
extern "C" int decode_categorical17(const uint8_t* data,int64_t size,int64_t n,const uint16_t* cdf,int16_t* out){
    if(size<4)return -1;uint32_t state=0;int64_t at=4;
    for(int j=0;j<4;++j)state|=uint32_t(data[j])<<(8*j);
    for(int64_t i=0;i<n;++i){
        uint32_t slot=state&4095;int s=0;while(s<16&&slot>=cdf[i*18+s+1])++s;
        uint32_t start=cdf[i*18+s],freq=cdf[i*18+s+1]-start;if(!freq||slot<start||slot>=start+freq)return -2;
        out[i]=s;state=freq*(state>>12)+slot-start;
        while(state<LOW){if(at>=size)return -3;state=(state<<8)|data[at++];}
    }
    return at==size&&state==LOW?0:-4;
}
