package org.fiarc.strictjava;
import com.google.gson.*;
import io.airlift.compress.zstd.ZstdDecompressor;
import java.io.*;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import java.util.function.*;
import java.util.regex.*;
import java.util.zip.*;

/** A13/W8 integer decoder, no browser, ONNX, Python, or torch runtime. */
public strictfp class FiARCCodec implements AutoCloseable {
    static final int ESC=-32768,LOW=1<<23,MAX_BLOB=512*1024*1024;
    public interface Conv extends AutoCloseable {int[] dot(int[] x,Layer l,int h,int w);void close();}
    public interface Graph extends Conv {
        void begin();void resetHistory();void advanceHistory(float[] rec,int history);
        Tensor historyFeatures(int history,int h,int w,float mean,float inv);
        Tensor convolution(Tensor x,Layer l,int act,int[] lut);
        Tensor pool(Tensor x);Tensor up(Tensor x);Tensor resize(Tensor x,int h,int w);Tensor cat(Tensor a,Tensor b);
        float[] download(Tensor x);
    }
    public interface Progress {void update(int done,int total,int frame,String stage);}
    Progress progress=(a,b,c,d)->{};
    BooleanSupplier cancellation=()->false;
    public void setProgress(Progress p){progress=p==null?(a,b,c,d)->{}:p;}
    void checkCancelled(){if(cancellation.getAsBoolean()||Thread.currentThread().isInterrupted())throw new java.util.concurrent.CancellationException("Decoding cancelled");}
    public static class CPU implements Conv {
        final java.util.concurrent.ExecutorService workers=java.util.concurrent.Executors.newFixedThreadPool(Math.max(1,Math.min(8,Integer.getInteger("fiarc.cpu.threads",Runtime.getRuntime().availableProcessors()))));
        public int[] dot(int[] x,Layer l,int h,int w) {
            int plane=h*w;int[] out=new int[l.co*plane];
            List<java.util.concurrent.Future<?>> tasks=new ArrayList<>();
            for(int c=0;c<l.co;c++){final int channel=c;tasks.add(workers.submit(()->{
                // Contiguous integer SAXPY: lets the JVM vectorize, without
                // changing any floating operation or dot-product semantics.
                for(int ci=0;ci<l.ci;ci++)for(int ky=0;ky<l.kh;ky++)for(int kx=0;kx<l.kw;kx++){
                    if(Thread.currentThread().isInterrupted())throw new java.util.concurrent.CancellationException("Decoding cancelled");
                    int dy=ky-l.kh/2,dx=kx-l.kw/2,weight=l.weights[((channel*l.ci+ci)*l.kh+ky)*l.kw+kx];if(weight==0)continue;
                    int y0=Math.max(0,-dy),y1=Math.min(h,h-dy),x0=Math.max(0,-dx),x1=Math.min(w,w-dx);
                    for(int y=y0;y<y1;y++){int dst=channel*plane+y*w,src=ci*plane+(y+dy)*w+dx;for(int z=x0;z<x1;z++)out[dst+z]+=x[src+z]*weight;}
                }
            }));}
            try{for(java.util.concurrent.Future<?> t:tasks)t.get();}
            catch(InterruptedException e){for(java.util.concurrent.Future<?> t:tasks)t.cancel(true);Thread.currentThread().interrupt();throw new java.util.concurrent.CancellationException("Decoding cancelled");}
            catch(java.util.concurrent.ExecutionException e){for(java.util.concurrent.Future<?> t:tasks)t.cancel(true);throw new IllegalStateException("CPU convolution failed",e.getCause());}
            return out;
        }
        public void close(){workers.shutdownNow();}
    }
    static void require(boolean b,String why){if(!b)throw new IllegalArgumentException(why);}
    static ByteBuffer le(byte[] b){return ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN);}
    static JsonObject json(byte[] b){return JsonParser.parseString(new String(b,StandardCharsets.UTF_8)).getAsJsonObject();}
    static String hash(byte[] b) throws Exception {byte[] d=MessageDigest.getInstance("SHA-256").digest(b);StringBuilder s=new StringBuilder();for(byte v:d)s.append(String.format("%02x",v&255));return s.toString();}
    static byte[] checked(Path folder,String file,String sha) throws Exception {
        require(file.matches("[a-zA-Z0-9_.-]+")&&!file.contains(".."),"Invalid filename");Path p=folder.resolve(file).toRealPath();require(p.startsWith(folder.toRealPath()),"Outside archive");
        require(Files.size(p)<=MAX_BLOB,"File too large");byte[] b=Files.readAllBytes(p);require(hash(b).equals(sha),"SHA-256 mismatch: "+file);return b;
    }
    static byte[] inflate(byte[] b,int limit) {
        long n=ZstdDecompressor.getDecompressedSize(b,0,b.length);require(n>=0&&n<=limit,"Oversized or unknown zstd content");byte[] out=new byte[(int)n];int got=new ZstdDecompressor().decompress(b,0,b.length,out,0,out.length);require(got==n,"zstd length");return out;
    }
    static class Npy {
        final int[] shape;final String type;final byte[] raw;
        Npy(byte[] b){
            require(b.length>=10&&b[0]==(byte)0x93&&new String(b,1,5,StandardCharsets.US_ASCII).equals("NUMPY"),"Not NPY");
            int major=b[6]&255;require(major==1||major==2,"NPY version");int headerAt=major==1?10:12,hs=major==1?(le(b).getShort(8)&65535):le(b).getInt(8);
            require(hs>=0&&hs<=b.length-headerAt,"NPY header");String s=new String(b,headerAt,hs,StandardCharsets.US_ASCII);
            require(s.contains("'fortran_order': False"),"NPY order");Matcher d=Pattern.compile("'descr':\\s*'([^']+)'").matcher(s);require(d.find(),"NPY dtype");type=d.group(1);
            Matcher m=Pattern.compile("'shape':\\s*\\(([^)]*)\\)").matcher(s);require(m.find(),"NPY shape");String[] dims=m.group(1).trim().split(",");List<Integer> sh=new ArrayList<>();for(String v:dims)if(!v.trim().isEmpty())sh.add(Integer.parseInt(v.trim()));shape=sh.stream().mapToInt(Integer::intValue).toArray();
            int size=1;for(int v:shape){require(v>=0,"Negative shape");size=Math.multiplyExact(size,v);}int elem=type.endsWith("1")?1:type.endsWith("2")?2:type.endsWith("4")?4:0;
            require(elem>0&&!type.startsWith(">"),"Unsupported NPY dtype");raw=Arrays.copyOfRange(b,headerAt+hs,b.length);require(raw.length==Math.multiplyExact(size,elem),"NPY length");
        }
        int[] ints(){ByteBuffer b=le(raw);int size=type.endsWith("4")?raw.length/4:type.endsWith("2")?raw.length/2:raw.length;int[] a=new int[size];for(int i=0;i<size;i++)a[i]=type.endsWith("4")?b.getInt():type.endsWith("2")?b.getShort():b.get();return a;}
        float[] floats(){require(type.equals("<f4"),"Expected float32");float[] a=new float[raw.length/4];le(raw).asFloatBuffer().get(a);for(float v:a)require(Float.isFinite(v),"Nonfinite weights");return a;}
    }
    static Map<String,Npy> npz(byte[] data) throws Exception {
        Map<String,Npy> a=new HashMap<>();int total=0;
        try(ZipInputStream z=new ZipInputStream(new ByteArrayInputStream(data))){ZipEntry e;while((e=z.getNextEntry())!=null){require(e.getName().endsWith(".npy")&&!e.getName().contains("/"),"Invalid model entry");ByteArrayOutputStream b=new ByteArrayOutputStream();byte[] buf=new byte[65536];int k;while((k=z.read(buf))!=-1){total=Math.addExact(total,k);require(total<=64*1024*1024,"Oversized model");b.write(buf,0,k);}String key=e.getName().substring(0,e.getName().length()-4);require(!a.containsKey(key),"Duplicate model entry");a.put(key,new Npy(b.toByteArray()));}}
        return a;
    }
    public static class Layer {
        public final int ci,co,kh,kw;public final int[] weights,exponents;public final float[] bias;
        Layer(String name,Map<String,Npy> a){Npy v=a.get(name+".q");require(v!=null&&v.shape.length==4,"Bad conv weights");co=v.shape[0];ci=v.shape[1];kh=v.shape[2];kw=v.shape[3];weights=v.ints();exponents=a.get(name+".exponent").ints();bias=a.get(name+".bias").floats();require(exponents.length==co&&bias.length==co,"Conv size");require(kh%2==1&&kw%2==1&&ci>0&&co>0,"Conv kernel");
            for(int c=0;c<co;c++){long sum=0;for(int j=c*ci*kh*kw;j<(c+1)*ci*kh*kw;j++){require(Math.abs(weights[j])<=127,"Not W8");sum+=Math.abs(weights[j]);}require(sum*4095<Integer.MAX_VALUE,"Integer dot overflow");}
        }
    }
    static class Tensor {final float[] a;final Object device;final int c,h,w;Tensor(float[] a,int c,int h,int w){this(a,null,c,h,w);}Tensor(float[] a,Object device,int c,int h,int w){this.a=a;this.device=device;this.c=c;this.h=h;this.w=w;require(a==null||a.length==c*h*w,"Tensor shape");}}
    final Path folder;final JsonObject manifest,meta,codec;final Map<String,Layer> layers=new LinkedHashMap<>();final Conv engine;
    public final int frames,h,w,restart,history;final int[] lut,exp;final float[] seedOffsets,finalOffsets,evenWeights;final float mean,std,inv,gain;final double seedStep,finalStep;final boolean lattice;final int stride;
    final float conditioningTolerance;final double[] seedFactors,finalFactors;
    float[] hist;int age=0;public int packetsRead=0,prefixFrames=0;
    public long coreNanos=0;public int processedFrames=0;
    final Map<String,byte[]> packetCache=new LinkedHashMap<>();
    public void preloadPackets()throws Exception {packetCache.clear();long bytes=0;for(JsonElement e:manifest.getAsJsonArray("groups")){JsonObject g=e.getAsJsonObject();if(bytes+g.get("bytes").getAsLong()>64L*1024*1024)break;String name=g.get("file").getAsString();byte[] b=checked(folder,name,g.get("sha256").getAsString());require(b.length==g.get("bytes").getAsInt(),"Packet bytes");packetCache.put(name,b);bytes+=b.length;}}
    public FiARCCodec(Path archive,Conv engine) throws Exception {
        this.engine=engine;folder=archive.toRealPath();manifest=json(Files.readAllBytes(folder.resolve("manifest.json")));
        String profile=manifest.get("profile").getAsString();
        require(profile.equals("fiarc_integer_experimental_v1")||profile.equals("fiarc_integer_boundscale_v2"),"Unsupported XERO archive profile");
        JsonArray shape=manifest.getAsJsonArray("shape");frames=shape.get(0).getAsInt();h=shape.get(1).getAsInt();w=shape.get(2).getAsInt();restart=manifest.get("restart").getAsInt();
        require(frames>0&&h>=4&&w>=4&&h<=4096&&w<=4096&&restart>=128&&(restart&(restart-1))==0,"Unsupported geometry/restart");
        Map<String,Npy> a=npz(checked(folder,manifest.get("model").getAsString(),manifest.get("model_sha256").getAsString()));meta=json(a.get("metadata").raw);codec=meta.getAsJsonObject("codec");JsonObject base=meta.getAsJsonObject("base");
        require(meta.get("profile").getAsString().equals(profile),"Unsupported model profile");
        require(meta.get("bits").getAsInt()==13&&meta.get("weight_bits").getAsInt()==8,"Only A13/W8 supported");
        require(base.get("input_mode").getAsString().equals("relative_latest")&&codec.get("conditioning").getAsString().equals("relative"),"Unsupported conditioning");
        require(!base.has("interpolate")||!base.get("interpolate").getAsBoolean(),"Interpolation model not supported");
        require(codec.get("conditional_seed_entropy").getAsBoolean()&&codec.get("conditional_final_entropy").getAsBoolean(),"Unsupported entropy topology");
        history=base.get("history").getAsInt();require(history>0&&history<=256,"History limit");mean=base.get("mean").getAsFloat();std=base.get("std").getAsFloat();require(std>0&&Float.isFinite(std)&&Float.isFinite(mean),"Normalization");inv=1.f/std;
        require(codec.get("tolerance").getAsDouble()>0&&codec.get("tolerance").getAsDouble()<=.5,"Invalid error bound");
        require(manifest.get("history").getAsString().equals("repeat_first_decoded"),"Unsupported initial history");
        conditioningTolerance=profile.equals("fiarc_integer_boundscale_v2")?codec.get("tolerance").getAsFloat():.01f;
        stride=codec.get("grid_stride").getAsInt();require(stride==2,"Seed stride must be 2");lattice=codec.get("lattice_context").getAsBoolean();gain=codec.get("even_refinement_gain").getAsFloat();
        seedStep=2*StrictMath.log1p(codec.get("seed_tolerance").getAsDouble()*.9999);finalStep=2*StrictMath.log1p(codec.get("tolerance").getAsDouble()*.9999);
        seedFactors=factors(seedStep);finalFactors=seedStep==finalStep?seedFactors:factors(finalStep);
        require(codec.get("tolerance").getAsDouble()==manifest.get("tolerance").getAsDouble(),"Tolerance/model mismatch");
        lut=a.get("silu_table").ints();exp=a.get("exp_table").ints();seedOffsets=a.get("seed_entropy").floats();finalOffsets=a.get("final_entropy").floats();evenWeights=a.containsKey("even_filter")?a.get("even_filter").floats():new float[]{.25f,.25f,.25f,.25f};
        require(lut.length==131073&&exp.length==257&&seedOffsets.length==368&&finalOffsets.length==368&&evenWeights.length==4,"Table sizes");
        for(Map.Entry<String,JsonElement> e:meta.getAsJsonObject("nodes").entrySet()){JsonObject node=e.getValue().getAsJsonObject();require(node.get("groups").getAsInt()==1,"Grouped conv unsupported");for(JsonElement v:node.getAsJsonArray("stride"))require(v.getAsInt()==1,"Stride unsupported");Layer l=new Layer(e.getKey(),a);require(node.getAsJsonArray("padding").get(0).getAsInt()==l.kh/2&&node.getAsJsonArray("padding").get(1).getAsInt()==l.kw/2,"Padding unsupported");layers.put(e.getKey(),l);}
        int end=0;for(JsonElement e:manifest.getAsJsonArray("groups")){JsonObject g=e.getAsJsonObject();int start=g.get("start").getAsInt(),nf=g.get("frames").getAsInt();require(start==end&&nf>0&&nf<=64&&start%restart+nf<=restart,"Packet/restart index");end+=nf;}require(end==frames,"Incomplete index");reset();
    }
    void reset(){int n=Math.multiplyExact(engine instanceof Graph?1:history,h*w);if(hist==null||hist.length!=n)hist=new float[n];Arrays.fill(hist,mean);age=0;if(engine instanceof Graph)((Graph)engine).resetHistory();}
    float[] host(Tensor t){return t.a!=null?t.a:((Graph)engine).download(t);}
    Tensor conv(String name,Tensor x,int act){
        checkCancelled();
        if(engine instanceof Graph){Layer l=layers.get(name);require(l!=null&&x.c==l.ci,"Layer mismatch");return ((Graph)engine).convolution(x,l,act,lut);}
        Layer l=layers.get(name);require(l!=null&&x.c==l.ci,"Unknown or mismatched layer "+name);float max=Math.scalb(1.f,-40);for(float v:x.a){require(Float.isFinite(v),"Nonfinite feature");max=Math.max(max,Math.abs(v));}
        double scale=Math.scalb(1.,ExactArithmetic.exponent(Float.floatToRawIntBits(max)));
        int[] q=new int[x.a.length];for(int i=0;i<q.length;i++){q[i]=(int)Math.rint(x.a[i]/scale);require(Math.abs(q[i])<=4095,"A13 overflow");}
        int[] dot=engine.dot(q,l,x.h,x.w);float[] out=new float[dot.length];int n=x.h*x.w;
        for(int c=0;c<l.co;c++){double cs=Math.scalb(scale,l.exponents[c]);for(int i=c*n;i<(c+1)*n;i++){float v=(float)(dot[i]*cs+(double)l.bias[c]);if(act==1){if(v<=-16.f)v=0;else if(v<16.f)v=lut[(int)Math.rint(v*4096.f)+65536]/4096.f;}else if(act==2&&v<=0)v=v*.1f;if(act!=0)v=(float)Math.rint(v*4096.f)/4096.f;out[i]=v;}}
        return new Tensor(out,l.co,x.h,x.w);
    }
    Tensor block(String name,Tensor x,int act){return conv(name+".2",conv(name+".0",x,act),act);}
    Tensor pool(Tensor x){if(engine instanceof Graph)return ((Graph)engine).pool(x);int h=x.h/2,w=x.w/2;float[] a=new float[x.c*h*w];for(int c=0;c<x.c;c++)for(int y=0;y<h;y++)for(int z=0;z<w;z++){int i=c*x.h*x.w+2*y*x.w+2*z;a[c*h*w+y*w+z]=((x.a[i]+x.a[i+1])+x.a[i+x.w]+x.a[i+x.w+1])*.25f;}return new Tensor(a,x.c,h,w);}
    Tensor up(Tensor x,Tensor target){if(engine instanceof Graph)return ((Graph)engine).resize(x,target.h,target.w);return new Tensor(PortableResize.resize(x.a,x.c,x.h,x.w,target.h,target.w),x.c,target.h,target.w);}
    Tensor cat(Tensor a,Tensor b){if(engine instanceof Graph)return ((Graph)engine).cat(a,b);require(a.h==b.h&&a.w==b.w,"Concat geometry");float[] out=Arrays.copyOf(a.a,a.a.length+b.a.length);System.arraycopy(b.a,0,out,a.a.length,b.a.length);return new Tensor(out,a.c+b.c,a.h,a.w);}
    float[] normalize(float[] x){float[] a=new float[x.length];for(int i=0;i<a.length;i++)a[i]=(x[i]-mean)*inv;return a;}
    float[] forecast(){int n=h*w;if(age==0){float[] a=new float[n];Arrays.fill(a,mean);return a;}Tensor input;float[] last;
        if(engine instanceof Graph){Graph g=(Graph)engine;g.begin();input=g.historyFeatures(history,h,w,mean,inv);last=normalize(hist);}
        else{float[] a=normalize(hist);last=Arrays.copyOfRange(a,a.length-n,a.length);for(int c=0;c<history-1;c++)for(int i=0;i<n;i++)a[c*n+i]-=last[i];input=new Tensor(a,history,h,w);}
        Tensor fine=block("base.fine",input,2),mid=block("base.middle",block("base.down",pool(fine),2),2);float[] out=host(conv("base.out",block("base.up",cat(fine,up(mid,fine)),2),0));for(int i=0;i<n;i++)out[i]=(out[i]+last[i])*std+mean;return out;}
    Tensor spatial(float[] x){if(engine instanceof Graph)((Graph)engine).begin();Tensor fine=block("refiner.fine",new Tensor(x,x.length/(h*w),h,w),1),mid=block("refiner.mid",pool(fine),1),low=block("refiner.low",pool(mid),1),u=block("refiner.upmid",cat(mid,up(low,mid)),1);return conv("refiner.out",block("refiner.upfine",cat(fine,up(u,fine)),1),0);}
    void advance(float[] r){int n=h*w;if(engine instanceof Graph){((Graph)engine).advanceHistory(r,history);hist=r.clone();}else if(age==0)for(int j=0;j<history;j++)System.arraycopy(r,0,hist,j*n,n);else{System.arraycopy(hist,n,hist,0,hist.length-n);System.arraycopy(r,0,hist,hist.length-n,n);}age++;}
    static int[] sites(int h,int w,int odd){int[] a=new int[(h*w+1)/2];int n=0;for(int y=0;y<h;y++)for(int x=0;x<w;x++)if(((y+x)&1)==odd)a[n++]=y*w+x;return Arrays.copyOf(a,n);}
    static int label(int q){return q>=-3&&q<=3?q+3:7;}
    static int ctx(short[] q,int h,int w,int i){int y=i/w,x=i%w,n=0,v=0;int[] k={y>0?label(q[i-w]):3,y+1<h?label(q[i+w]):3,x>0?label(q[i-1]):3,x+1<w?label(q[i+1]):3};for(int a:k){if(a!=3)n++;if(a>=4&&a<=6)v++;if(a<=2)v--;}return n*9+v+4;}
    int[] cdf(float[] logits,float[] offsets,int[] ix,short[] q,int h,int w,int odd,short[] mask){int[] out=new int[ix.length*9];int[] v=new int[8];long[] weight=new long[8];for(int j=0;j<ix.length;j++){int i=ix[j],at=j*9;if(mask!=null&&mask[i]!=0){for(int k=4;k<9;k++)out[at+k]=4096;continue;}int context=odd!=0?ctx(q,h,w,i):45,mx=Integer.MIN_VALUE,best=0;for(int k=0;k<8;k++){v[k]=(int)Math.rint((logits[8*i+k]+offsets[context*8+k])*16.f);if(v[k]>mx){mx=v[k];best=k;}}long sum=0;for(int k=0;k<8;k++){weight[k]=exp[Math.min(256,mx-v[k])];sum+=weight[k];}int total=0;int[] freq=new int[8];for(int k=0;k<8;k++){freq[k]=(int)(weight[k]*4088/sum)+1;total+=freq[k];}freq[best]+=4096-total;for(int k=0;k<8;k++)out[at+k+1]=out[at+k]+freq[k];}return out;}
    static double[] factors(double step){double[] a=new double[65535];for(int q=-32767;q<=32767;q++)a[q+32767]=Math.exp(q*step);return a;}
    float[] restore(float[] p,short[] q,float[] esc,double step){float[] out=new float[p.length];int j=0;double[] factors=step==seedStep?seedFactors:finalFactors;for(int i=0;i<p.length;i++){if(q[i]==ESC){require(j<esc.length,"Escape exhaustion");out[i]=esc[j++];}else out[i]=q[i]==0?p[i]:(float)((p[i]==0?1.:(double)p[i])*factors[q[i]+32767]);require(Float.isFinite(out[i]),"Nonfinite reconstruction");}require(j==esc.length,"Unused escapes");return out;}
    float[] decode(Packet group,float[] pred){int n=h*w,lh=(h+1)/2,lw=(w+1)/2;if(engine instanceof Graph)((Graph)engine).begin();float[] prior=host(conv("prior.2",conv("prior.0",new Tensor(normalize(pred),1,h,w),1),0));float[] ll=new float[lh*lw*8];for(int y=0;y<lh;y++)for(int x=0;x<lw;x++)for(int k=0;k<8;k++)ll[(y*lw+x)*8+k]=prior[(k+1)*n+y*2*w+x*2];short[] low=new short[lh*lw],sq=new short[n];for(int odd=0;odd<2;odd++){int[] ix=sites(lh,lw,odd);short[] v=group.phases[odd].decode(cdf(ll,seedOffsets,ix,low,lh,lw,odd,null));for(int j=0;j<ix.length;j++)low[ix[j]]=v[j];}
        float[] small=new float[low.length];for(int y=0;y<lh;y++)for(int x=0;x<lw;x++){int i=y*2*w+x*2;sq[i]=low[y*lw+x];small[y*lw+x]=pred[i];}
        float[] restored=restore(small,low,group.take(low,0),seedStep),seed=pred.clone();boolean[] mask=new boolean[n];for(int i=0;i<n;i++)mask[i]=sq[i]!=0;for(int y=0;y<lh;y++)for(int x=0;x<lw;x++)seed[y*2*w+x*2]=restored[y*lw+x];
        float[] feat=new float[n*(lattice?4:3)];for(int i=0;i<n;i++){float scale=Math.max(conditioningTolerance*Math.abs(pred[i]),1e-6f);feat[i]=(pred[i]-mean)*inv;feat[n+i]=mask[i]?1:0;feat[2*n+i]=mask[i]?(seed[i]-pred[i])/scale:0;if(lattice)feat[3*n+i]=((i/w)%2==0&&(i%w)%2==0)?1:0;}
        float[] neural=host(spatial(feat));float[] post=new float[n],logits=new float[n*8];for(int i=0;i<n;i++){post[i]=mask[i]?seed[i]:pred[i]+neural[i]*Math.max(conditioningTolerance*Math.abs(pred[i]),1e-6f);for(int k=0;k<8;k++)logits[i*8+k]=neural[(k+1)*n+i];}
        short[] q=new short[n];for(int odd=0;odd<2;odd++){short[] context=new short[n];for(int i=0;i<n;i++)context[i]=sq[i]!=0?sq[i]:q[i];int[] ix=sites(h,w,odd);short[] v=group.phases[2+odd].decode(cdf(logits,finalOffsets,ix,context,h,w,odd,odd==0?sq:null));for(int j=0;j<ix.length;j++)q[ix[j]]=v[j];}
        float[] escapes=group.take(q,1),rec=restore(post,q,escapes,finalStep);
        if(gain!=0){float[] delta=new float[n],after=post.clone();for(int i=0;i<n;i++)delta[i]=((i/w+i%w)&1)==0?rec[i]-post[i]:0;
            for(int y=0;y<h;y++)for(int x=0;x<w;x++){int i=y*w+x;if(((x+y)&1)==0||mask[i])continue;float sum=(y>0?delta[i-w]:0)*evenWeights[0];sum+=(x>0?delta[i-1]:0)*evenWeights[1];sum+=(x+1<w?delta[i+1]:0)*evenWeights[2];sum+=(y+1<h?delta[i+w]:0)*evenWeights[3];after[i]=post[i]+gain*sum;}rec=restore(after,q,escapes,finalStep);
        }return rec;
    }
    public void range(int first,int last,BiConsumer<Integer,float[]> sink,BooleanSupplier cancel) throws Exception {
        require(first>=1&&last>=first&&last<=frames,"Invalid requested range");cancellation=cancel;int start=(first-1)/restart*restart;reset();packetsRead=0;prefixFrames=0;coreNanos=0;processedFrames=0;
        int end=last;for(JsonElement e:manifest.getAsJsonArray("groups")){JsonObject g=e.getAsJsonObject();int at=g.get("start").getAsInt();if(at<last&&at+g.get("frames").getAsInt()>=last){end=at+g.get("frames").getAsInt();break;}}int total=end-start;progress.update(0,total,start+1,"Reading compressed packet");
        for(JsonElement e:manifest.getAsJsonArray("groups")){JsonObject g=e.getAsJsonObject();int pos=g.get("start").getAsInt();if(pos<start||pos>=last)continue;String filename=g.get("file").getAsString();byte[] bytes=packetCache.get(filename);if(bytes==null)bytes=checked(folder,filename,g.get("sha256").getAsString());require(bytes.length==g.get("bytes").getAsInt(),"Packet byte count");Packet p=new Packet(bytes);require(p.h==h&&p.w==w&&p.n==g.get("frames").getAsInt(),"Packet dimensions mismatch");packetsRead++;
            // Finish the last packet for terminal entropy / escape integrity checks.
            for(int j=0;j<p.n;j++){checkCancelled();int frame=pos+j;progress.update(frame-start,total,frame+1,frame<first-1?"Restoring restart context":frame>=last?"Verifying packet tail":"Decoding requested frames");if(frame%restart==0)reset();long t=System.nanoTime();float[] rec=engine instanceof FiARCStrictGPU?((FiARCStrictGPU)engine).decodeFrame(this,p):decode(p,forecast());coreNanos+=System.nanoTime()-t;processedFrames++;if(engine instanceof FiARCStrictGPU)age++;else advance(rec);if(frame<first-1)prefixFrames++;else if(frame<last)sink.accept(frame+1,rec);progress.update(frame-start+1,total,frame+1,"Decoded");}
            if(!(engine instanceof FiARCStrictGPU))for(int k=0;k<2;k++)require(p.escapeAt[k]==p.escapes[k].length,"Unused float escapes");
        }
    }
    /** Incremental viewer cursor. A packet stays resident until its last frame;
     * terminal rANS/escape checks are still performed at the packet boundary.
     * The entire compressed packet is SHA-256 checked before any frame is used.
     * Backward seeks restore only the enclosing restart's causal prefix. */
    public final class Cursor {
        int next=-1,groupIndex,within;Packet packet;
        final JsonArray groups=manifest.getAsJsonArray("groups");
        public synchronized float[] frame(int target,BiConsumer<Integer,float[]> sink,BooleanSupplier cancel)throws Exception{
            require(target>=1&&target<=frames,"Frame outside archive");cancellation=cancel;
            int wanted=target-1;
            try {
                if(next<0||wanted<next){next=wanted/restart*restart;reset();packet=null;within=0;groupIndex=0;while(groups.get(groupIndex).getAsJsonObject().get("start").getAsInt()<next)groupIndex++;}
                int start=next;float[] result=null;
                while(next<=wanted){
                    checkCancelled();progress.update(next-start,wanted-start+1,next+1,"Decoding virtual frame");
                    if(packet==null){JsonObject g=groups.get(groupIndex).getAsJsonObject();require(g.get("start").getAsInt()==next,"Cursor packet position");String file=g.get("file").getAsString();byte[] bytes=packetCache.get(file);if(bytes==null)bytes=checked(folder,file,g.get("sha256").getAsString());require(bytes.length==g.get("bytes").getAsInt(),"Packet byte count");packet=new Packet(bytes);require(packet.h==h&&packet.w==w&&packet.n==g.get("frames").getAsInt(),"Packet dimensions mismatch");within=0;packetsRead++;}
                    if(next%restart==0)reset();long t=System.nanoTime();result=engine instanceof FiARCStrictGPU?((FiARCStrictGPU)engine).decodeFrame(FiARCCodec.this,packet):decode(packet,forecast());coreNanos+=System.nanoTime()-t;processedFrames++;
                    if(engine instanceof FiARCStrictGPU)age++;else advance(result);next++;within++;sink.accept(next,result);
                    if(within==packet.n){if(!(engine instanceof FiARCStrictGPU))for(int k=0;k<2;k++)require(packet.escapeAt[k]==packet.escapes[k].length,"Unused float escapes");packet=null;groupIndex++;}
                }
                progress.update(wanted-start+1,wanted-start+1,target,"Ready");return result;
            }catch(Exception e){next=-1;packet=null;throw e;}
        }
    }
    public Cursor cursor(){return new Cursor();}
    public void close(){packetCache.clear();hist=null;engine.close();}
    static class Bits {final byte[] a;int p;Bits(byte[] a){this.a=a;}int bit(){require(p<a.length*8,"Truncated Golomb");return (a[p>>3]>>(7-(p++%8)))&1;}}
    static class Rans {
        int[] counts,cursor,state;byte[][] streams;short[] rare;int frame,rareAt;
        Rans(byte[] a){require(a.length>=32,"Short rANS");ByteBuffer b=le(a);String magic=new String(a,0,8,StandardCharsets.US_ASCII);require(magic.equals("SRANSG01")||magic.equals("SRANSG02"),"rANS profile");int nf=b.getInt(8),ns=b.getInt(12),n=b.getInt(16),total=b.getInt(20),side=b.getInt(24),nb=b.getInt(28);require(nf>0&&nf<=64&&ns>0&&ns<=65536&&n>=0&&nb>=0&&nb<=n&&side>=0&&total>=0,"rANS counts");int off=32+4*(nf+ns);require((long)off+total+side==a.length,"rANS length");counts=new int[nf];long sum=0;for(int i=0;i<nf;i++){counts[i]=b.getInt(32+4*i);require(counts[i]>=0,"Negative count");sum+=counts[i];}require(sum==n,"rANS count sum");streams=new byte[ns][];cursor=new int[ns];state=new int[ns];int at=off;
            for(int i=0;i<ns;i++){int len=b.getInt(32+4*(nf+i));require(len>=4&&(long)at+len<=off+total,"rANS stream length");streams[i]=Arrays.copyOfRange(a,at,at+len);state[i]=le(streams[i]).getInt();require(state[i]>=LOW,"rANS state");cursor[i]=4;at+=len;}require(at==off+total&&(side>0)==(nb>0),"rANS side framing");rare=new short[nb];
            if(nb>0){byte[] body=Arrays.copyOfRange(a,at,a.length);if(magic.equals("SRANSG01")){byte[] un=inflate(body,Math.multiplyExact(nb,2));require(un.length==nb*2,"Bypass size");le(un).asShortBuffer().get(rare);}else{Bits bits=new Bits(body);for(int i=0;i<nb;i++){int k=0;while(bits.bit()==0)require(++k<=15,"Golomb overflow");int code=1;for(int j=0;j<k;j++)code=code*2+bits.bit();int u=code-1,mag=u/2+4;require(mag<=32768&&(mag!=32768||(u&1)!=0),"Bad symbol");rare[i]=(short)((u&1)!=0?-mag:mag);}require(body.length*8-bits.p<=7,"Extra Golomb bytes");while(bits.p<body.length*8)require(bits.bit()==0,"Golomb padding");}}
        }
        short[] decode(int[] cdf){int n=cdf.length/9,ns=streams.length;require(frame<counts.length&&n==counts[frame],"CDF count");short[] out=new short[n];for(int s=0;s<ns;s++)for(int i=(int)((long)n*s/ns);i<(int)((long)n*(s+1)/ns);i++){int slot=state[s]&4095,k=0,at=i*9;while(k<8&&slot>=cdf[at+k+1])k++;require(k<8,"CDF mismatch");int lo=cdf[at+k],freq=cdf[at+k+1]-lo;require(freq>0&&slot>=lo,"Bad frequency");out[i]=(short)(k==7?ESC:k-3);state[s]=freq*(state[s]>>>12)+slot-lo;while(state[s]<LOW){require(cursor[s]<streams[s].length,"Entropy exhausted (model mismatch)");state[s]=(state[s]<<8)|(streams[s][cursor[s]++]&255);}}
            for(int i=0;i<n;i++)if(out[i]==ESC){require(rareAt<rare.length,"Bypass exhausted");out[i]=rare[rareAt++];}if(++frame==counts.length){for(int s=0;s<ns;s++)require(cursor[s]==streams[s].length&&state[s]==LOW,"Nonterminal rANS state");require(rareAt==rare.length,"Unused bypass");}return out;
        }
    }
    static class Packet {
        final int n,h,w;final Rans[] phases=new Rans[4];final float[][] escapes=new float[2][];final int[] escapeAt={0,0};
        Packet(byte[] a){require(a.length>=68&&new String(a,0,8,StandardCharsets.US_ASCII).equals("FSEEDSG1"),"Packet magic");ByteBuffer b=le(a);n=b.getInt(8);h=b.getInt(12);w=b.getInt(16);require(n>0&&n<=64&&h>0&&w>0&&h<=4096&&w<=4096,"Packet shape");int at=68;for(int i=0;i<6;i++){long len=b.getLong(20+i*8);require(len>=0&&len<=a.length-at,"Packet section length");byte[] body=Arrays.copyOfRange(a,at,at+(int)len);at+=(int)len;if(i<4){phases[i]=new Rans(body);require(phases[i].counts.length==n,"Phase length");}else{byte[] un=inflate(body,(int)Math.min(MAX_BLOB,(long)n*h*w*4));require(un.length%4==0,"Escape bytes");float[] v=new float[un.length/4];le(un).asFloatBuffer().get(v);escapes[i-4]=v;}}require(at==a.length,"Trailing packet bytes");}
        float[] take(short[] q,int stage){int n=0;for(short v:q)if(v==ESC)n++;int a=escapeAt[stage];require(a+n<=escapes[stage].length,"Float escapes exhausted");escapeAt[stage]+=n;return Arrays.copyOfRange(escapes[stage],a,a+n);}
    }
}
