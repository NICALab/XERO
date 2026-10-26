#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <cmath>

static constexpr uint32_t LOW=1u<<23, SCALE=12, MASK=(1u<<SCALE)-1;
static inline int cat(int x){return std::max(-2,std::min(2,x))+2;}
static inline int sym(int x){return x>=-3&&x<=3?x+3:7;}
static inline int ctx(const int16_t* q,int64_t i,int H,int W,int mode){
    int64_t area=int64_t(H)*W;int col=i%W;int row=(i/W)%H;
    int l=col?q[i-1]:0,u=row?q[i-W]:0,t=i>=area?q[i-area]:0;
    if(mode==8||mode==9){
        // Ternary sign contexts cover more neighbors without a 5^N table.
        // All referenced symbols have already been decoded in raster order.
        int values[9]={l,u,t,row&&col?q[i-W-1]:0,
            row&&col+1<W?q[i-W+1]:0,
            i>=area&&col+1<W?q[i-area+1]:0,
            i>=area&&row+1<H?q[i-area+W]:0,
            col>=2?q[i-2]:0,row>=2?q[i-2*W]:0};
        int c=0;
        for(int k=0;k<(mode==8?7:9);++k)c=c*3+(values[k]>0?2:values[k]<0?0:1);
        return c;
    }
    int c=cat(l)*5+cat(u);
    if(mode>=3)c=c*5+cat(t);
    if(mode>=4)c=c*5+cat(row&&col?q[i-W-1]:0);
    if(mode>=5)c=c*5+cat(row&&col+1<W?q[i-W+1]:0);
    if(mode>=6)c=c*5+cat(i>=area&&col+1<W?q[i-area+1]:0);
    if(mode>=7)c=c*5+cat(i>=area&&row+1<H?q[i-area+W]:0);
    return c;
}

extern "C" void fiarc_context_counts(const int16_t* q,int64_t n,int H,int W,int mode,uint64_t* counts){
    for(int64_t i=0;i<n;++i)counts[ctx(q,i,H,W,mode)*8+sym(q[i])]++;
}

extern "C" int64_t fiarc_context_encode(const int16_t* q,int64_t n,int H,int W,int mode,const uint16_t* cdf,uint8_t* output,int64_t capacity){
    uint32_t state=LOW;
    std::vector<uint8_t> renorm;renorm.reserve(n/4+16);
    for(int64_t i=n-1;i>=0;--i){
        int c=ctx(q,i,H,W,mode),s=sym(q[i]);
        uint32_t start=cdf[c*9+s],freq=cdf[c*9+s+1]-start;
        if(!freq)return -1;
        uint32_t limit=((LOW>>SCALE)<<8)*freq;
        while(state>=limit){renorm.push_back(state&255);state>>=8;}
        state=((state/freq)<<SCALE)+(state%freq)+start;
    }
    if(int64_t(renorm.size())+4>capacity)return -2;
    for(int j=0;j<4;++j)output[j]=(state>>(8*j))&255;
    for(size_t j=0;j<renorm.size();++j)output[j+4]=renorm[renorm.size()-j-1];
    return renorm.size()+4;
}

extern "C" int fiarc_context_decode(const uint8_t* input,int64_t size,int64_t n,int H,int W,int mode,const uint16_t* cdf,const int16_t* escapes,int64_t nesc,int16_t* q){
    if(size<4)return -1;
    uint32_t state=0;for(int j=0;j<4;++j)state|=uint32_t(input[j])<<(8*j);
    int64_t at=4,ex=0;
    for(int64_t i=0;i<n;++i){
        int c=ctx(q,i,H,W,mode);uint32_t slot=state&MASK;
        int s=0;while(s<7&&slot>=cdf[c*9+s+1])++s;
        uint32_t start=cdf[c*9+s],freq=cdf[c*9+s+1]-start;
        if(!freq)return -2;
        if(s==7){if(ex>=nesc)return -3;q[i]=escapes[ex++];}else q[i]=s-3;
        state=freq*(state>>SCALE)+slot-start;
        while(state<LOW){if(at>=size)return -4;state=(state<<8)|input[at++];}
    }
    return at==size&&ex==nesc&&state==LOW?0:-5;
}

static inline float corrected_prediction(const float* neural,const float* rec,int i,int H,int W,double gain,const double* weights=nullptr,bool map=false){
    int col=i%W,row=i/W;
    if(weights){
        static const int dy[12]={0,0,0,-1,-1,-1,-1,-1,-2,-2,-2,-3};
        static const int dx[12]={-1,-2,-3,-2,-1,0,1,2,-1,0,1,0};
        double value=weights[map?12*H*W+i:12];
        for(int k=0;k<12;++k){int y=row+dy[k],x=col+dx[k];
            if(y>=0&&x>=0&&x<W){int at=y*W+x;value+=weights[map?k*H*W+i:k]*(double(rec[at])-neural[at]);}}
        return float(double(neural[i])+gain*value);
    }
    double l=col?double(rec[i-1])-neural[i-1]:0;
    double u=row?double(rec[i-W])-neural[i-W]:0;
    double ul=row&&col?double(rec[i-W-1])-neural[i-W-1]:0;
    double correction=0;
    if(row&&col){
        double a=l,b=u,c=l+u-ul;
        correction=a+b+c-std::min(a,std::min(b,c))-std::max(a,std::max(b,c));
    }else correction=row?u:l;
    return float(double(neural[i])+gain*correction);
}

static inline float apply_symbol(float p,int q,double step){
    if(q==0)return p;
    double anchor=p==0?1.0:std::abs(double(p));
    return float((p<0?-anchor:anchor)*std::exp(q*step));
}

static void spatial_quantize_impl(const float* source,const float* neural,int H,int W,double gain,double step,double eps,int16_t* q,float* rec,const double* weights,bool min_magnitude=false,bool map=false){
    for(int i=0;i<H*W;++i){
        float p=corrected_prediction(neural,rec,i,H,W,gain,weights,map);double s=source[i];
        int symbol=0;
        if(std::abs(s-p)>eps*std::abs(s)){
            double anchor=p==0?1.0:std::abs(double(p));
            double v=std::nearbyint(std::log(std::abs(s)/anchor)/step);
            if(!std::isfinite(v)||v<=-32768||v>32767)symbol=-32768;
            else symbol=int(v);
            if(min_magnitude){
                // Overlapping allowed intervals: prefer a smaller symbol, not
                // necessarily the reconstruction nearest to the source value.
                // Final float32 reconstruction must still pass the exact bound.
                double lo=std::ceil(std::log(std::abs(s)*(1-eps)/anchor)/step);
                double hi=std::floor(std::log(std::abs(s)*(1+eps)/anchor)/step);
                double nearest_zero=lo>0?lo:(hi<0?hi:0.);
                symbol=-32768;
                double candidates[8]={nearest_zero-2,nearest_zero-1,nearest_zero,nearest_zero+1,nearest_zero+2,v,-1,1};
                for(double candidate:candidates){
                    if(!std::isfinite(candidate)||candidate<=-32768||candidate>32767)continue;
                    int k=int(candidate);float trial=apply_symbol(p,k,step);
                    if(std::isfinite(trial)&&std::abs(s-double(trial))<=eps*std::abs(s)
                       &&(symbol==-32768||std::abs(k)<std::abs(symbol)))symbol=k;
                }
            }
        }
        float r=symbol==-32768?float(s):apply_symbol(p,symbol,step);
        if(!std::isfinite(r)||std::abs(s-r)>eps*std::abs(s)){symbol=-32768;r=float(s);}
        q[i]=int16_t(symbol);rec[i]=r;
    }
}

static int spatial_restore_impl(const float* neural,const int16_t* q,int H,int W,double gain,double step,const float* escapes,int64_t nesc,float* rec,const double* weights,bool map=false){
    int64_t ex=0;
    for(int i=0;i<H*W;++i){
        float p=corrected_prediction(neural,rec,i,H,W,gain,weights,map);
        if(q[i]==-32768){if(ex>=nesc)return -1;rec[i]=escapes[ex++];}
        else rec[i]=apply_symbol(p,q[i],step);
    }
    return ex==nesc?0:-2;
}

extern "C" void fiarc_spatial_quantize(const float* s,const float* p,int h,int w,double g,double step,double eps,int16_t* q,float* r){spatial_quantize_impl(s,p,h,w,g,step,eps,q,r,nullptr);}
extern "C" int fiarc_spatial_restore(const float* p,const int16_t* q,int h,int w,double g,double step,const float* ex,int64_t n,float* r){return spatial_restore_impl(p,q,h,w,g,step,ex,n,r,nullptr);}
extern "C" void fiarc_spatial_quantize_linear(const float* s,const float* p,int h,int w,double g,double step,double eps,int16_t* q,float* r,const double* weights){spatial_quantize_impl(s,p,h,w,g,step,eps,q,r,weights);}
extern "C" void fiarc_spatial_quantize_min_magnitude(const float* s,const float* p,int h,int w,double g,double step,double eps,int16_t* q,float* r,const double* weights){spatial_quantize_impl(s,p,h,w,g,step,eps,q,r,weights,true);}
extern "C" int fiarc_spatial_restore_linear(const float* p,const int16_t* q,int h,int w,double g,double step,const float* ex,int64_t n,float* r,const double* weights){return spatial_restore_impl(p,q,h,w,g,step,ex,n,r,weights);}
extern "C" void fiarc_spatial_quantize_map(const float* s,const float* p,int h,int w,double g,double step,double eps,int16_t* q,float* r,const double* weights){spatial_quantize_impl(s,p,h,w,g,step,eps,q,r,weights,false,true);}
extern "C" int fiarc_spatial_restore_map(const float* p,const int16_t* q,int h,int w,double g,double step,const float* ex,int64_t n,float* r,const double* weights){return spatial_restore_impl(p,q,h,w,g,step,ex,n,r,weights,true);}
extern "C" void fiarc_spatial_predictions_linear(const float* neural,const float* rec,int h,int w,const double* weights,float* predictions){
    for(int i=0;i<h*w;++i)predictions[i]=corrected_prediction(neural,rec,i,h,w,1.,weights);
}
