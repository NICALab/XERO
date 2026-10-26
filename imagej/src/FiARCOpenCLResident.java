package org.fiarc.strictjava;
import org.jocl.*;
import static org.jocl.CL.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/** Whole conv pyramids stay on GPU. The archive and arithmetic are unchanged. */
public class FiARCOpenCLResident extends FiARCOpenCL implements FiARCCodec.Graph {
    cl_program graphProgram;Map<String,cl_kernel> kernels=new HashMap<>();
    static class Buffer {cl_mem mem;long bytes;boolean used;Buffer(cl_mem m,long b){mem=m;bytes=b;}}
    static class Pair {FiARCCodec.Tensor a,b;Pair(FiARCCodec.Tensor a,FiARCCodec.Tensor b){this.a=a;this.b=b;}}
    final List<Buffer> pool=new ArrayList<>();final Map<FiARCCodec.Layer,cl_mem[]> layerBuffers=new IdentityHashMap<>();
    cl_mem table,histA,histB,guard;int histPlane,histCount;boolean historyValid=false;public boolean dp4a;
    public FiARCOpenCLResident(int gpu){super(gpu);try{
        byte[] source;try(InputStream in=getClass().getResourceAsStream("/fiarc/resident.cl")){ByteArrayOutputStream b=new ByteArrayOutputStream();byte[] a=new byte[8192];int n;while((n=in.read(a))!=-1)b.write(a,0,n);source=b.toByteArray();}
        int[] error={0};graphProgram=clCreateProgramWithSource(context,1,new String[]{resource("scalar.cl")+resource("portable.cl")+resource("entropy.cl")+new String(source,StandardCharsets.UTF_8)},null,error);ok(error[0]);
        long[] len={0};ok(clGetDeviceInfo(device,CL_DEVICE_VENDOR,0,null,len));byte[] name=new byte[(int)len[0]];ok(clGetDeviceInfo(device,CL_DEVICE_VENDOR,name.length,Pointer.to(name),null));dp4a=new String(name).contains("NVIDIA")&&!Boolean.getBoolean("fiarc.disable.dp4a");
        int code=clBuildProgram(graphProgram,0,null,"-cl-std=CL1.2"+(dp4a?" -D NVIDIA_DP4A=1":""),null,null);
        if(code!=CL_SUCCESS&&dp4a){dp4a=false;code=clBuildProgram(graphProgram,0,null,"-cl-std=CL1.2",null,null);}
        if(code!=CL_SUCCESS){clGetProgramBuildInfo(graphProgram,device,CL_PROGRAM_BUILD_LOG,0,null,len);byte[] log=new byte[(int)len[0]];clGetProgramBuildInfo(graphProgram,device,CL_PROGRAM_BUILD_LOG,log.length,Pointer.to(log),null);throw new IllegalStateException(new String(log));}
        for(String n:new String[]{"maxparts","scalevalue","pack","conv","pool","up","portable_up","history_shift","history_features","pixels","cdf","rans","prefix","symbols","escape_count","escape_scatter"}){cl_kernel k=clCreateKernel(graphProgram,n,error);ok(error[0]);kernels.put(n,k);}
        if(dp4a&&!Boolean.getBoolean("fiarc.disable.mma")){cl_kernel k=clCreateKernel(graphProgram,"conv_mma",error);ok(error[0]);kernels.put("conv_mma",k);}
        guard=create(4,Pointer.to(new int[1]));
    }catch(Throwable e){close();throw new IllegalStateException("Resident GPU setup failed",e);}}
    static String resource(String name)throws IOException {try(InputStream in=FiARCOpenCLResident.class.getResourceAsStream("/fiarc/"+name)){ByteArrayOutputStream b=new ByteArrayOutputStream();byte[] a=new byte[8192];int n;while((n=in.read(a))!=-1)b.write(a,0,n);return new String(b.toByteArray(),StandardCharsets.UTF_8);}}
    cl_mem create(long bytes,Pointer pointer){int[] err={0};cl_mem m=clCreateBuffer(context,CL_MEM_READ_WRITE|(pointer!=null?CL_MEM_COPY_HOST_PTR:0),Math.max(4,bytes),pointer,err);ok(err[0]);return m;}
    cl_mem alloc(long bytes){Buffer best=null;for(Buffer b:pool)if(!b.used&&b.bytes>=bytes&&(best==null||b.bytes<best.bytes))best=b;if(best!=null){best.used=true;return best.mem;}Buffer b=new Buffer(create(bytes,null),bytes);b.used=true;pool.add(b);return b.mem;}
    void release(cl_mem m){for(Buffer b:pool)if(b.mem==m){b.used=false;return;}}
    void release(FiARCCodec.Tensor t){if(t.device instanceof Pair){Pair p=(Pair)t.device;release(p.a);release(p.b);}else if(t.device!=null)release((cl_mem)t.device);}
    public long scratchBytes(){long n=0;for(Buffer b:pool)n+=b.bytes;return n;}
    public long historyBytes(){return (long)histPlane*histCount*4*(histB==null?1:2);}
    public void begin(){for(Buffer b:pool)b.used=false;}
    void args(cl_kernel k,Object...args){for(int i=0;i<args.length;i++){Object a=args[i];if(a instanceof cl_mem)ok(clSetKernelArg(k,i,Sizeof.cl_mem,Pointer.to((cl_mem)a)));else if(a instanceof Float)ok(clSetKernelArg(k,i,4,Pointer.to(new float[]{(Float)a})));else ok(clSetKernelArg(k,i,4,Pointer.to(new int[]{(Integer)a})));}}
    void launch(String name,int n,Object...args){cl_kernel k=kernels.get(name);args(k,args);ok(clEnqueueNDRangeKernel(queue,k,1,null,new long[]{((n+255L)/256)*256},new long[]{256},0,null,null));}
    cl_mem device(FiARCCodec.Tensor x){if(x.device!=null)return (cl_mem)x.device;cl_mem m=alloc((long)x.a.length*4);ok(clEnqueueWriteBuffer(queue,m,CL_TRUE,0,(long)x.a.length*4,Pointer.to(x.a),0,null,null));return m;}
    public float[] download(FiARCCodec.Tensor x){if(x.a!=null)return x.a;float[] a=new float[x.c*x.h*x.w];ok(clEnqueueReadBuffer(queue,(cl_mem)x.device,CL_TRUE,0,(long)a.length*4,Pointer.to(a),0,null,null));return a;}
    public FiARCCodec.Tensor convolution(FiARCCodec.Tensor x,FiARCCodec.Layer l,int act,int[] lut){
        return convolution(x,l,act,lut,false);
    }
    FiARCCodec.Tensor convolution(FiARCCodec.Tensor x,FiARCCodec.Layer l,int act,int[] lut,boolean consume){
        int plane=x.h*x.w,n=plane*x.c,ci4=(x.c+3)/4;Pair pair=x.device instanceof Pair?(Pair)x.device:null;
        cl_mem input=pair==null?device(x):device(pair.a),inputB=pair==null?input:device(pair.b);int ca=pair==null?x.c:pair.a.c;
        cl_mem parts=alloc(4096),scale=alloc(4),packed=alloc((long)plane*ci4*8);int groups=Math.min(1024,(n+255)/256);
        launch("maxparts",groups*256,input,inputB,parts,n,ca*plane,guard);launch("scalevalue",256,parts,scale,groups);launch("pack",plane*ci4,input,inputB,scale,packed,plane,x.c,ca,guard);
        // In-order queue: packing has consumed input before convolution writes
        // its output, so an owned input buffer can immediately be recycled.
        if(consume)release(x);cl_mem out=alloc((long)plane*l.co*4);
        cl_mem[] ls=layerBuffers.get(l);if(ls==null){int[] weights=new int[l.co*l.kh*l.kw*ci4];for(int o=0;o<l.co;o++)for(int c=0;c<l.ci;c++)for(int ky=0;ky<l.kh;ky++)for(int kx=0;kx<l.kw;kx++){int v=l.weights[((o*l.ci+c)*l.kh+ky)*l.kw+kx]&255;weights[((o*l.kh+ky)*l.kw+kx)*ci4+c/4]|=v<<(8*(c%4));}
            ls=new cl_mem[]{create((long)weights.length*4,Pointer.to(weights)),create((long)l.exponents.length*4,Pointer.to(l.exponents)),create((long)l.bias.length*4,Pointer.to(l.bias))};layerBuffers.put(l,ls);}
        if(table==null)table=create((long)lut.length*4,Pointer.to(lut));
        boolean mma=kernels.containsKey("conv_mma");cl_kernel k=kernels.get(mma?"conv_mma":"conv");args(k,packed,ls[0],out,scale,ls[1],ls[2],table,x.h,x.w,ci4,l.co,l.kh,l.kw,act);
        ok(clEnqueueNDRangeKernel(queue,k,2,null,mma?new long[]{((plane+15L)/16)*32,((l.co+7L)/8)}:new long[]{((plane+31L)/32)*8,((l.co+15L)/16)*8},mma?new long[]{32,1}:new long[]{8,8},0,null,null));release(parts);release(scale);release(packed);return new FiARCCodec.Tensor(null,out,l.co,x.h,x.w);
    }
    public FiARCCodec.Tensor pool(FiARCCodec.Tensor x){int n=x.c*(x.h/2)*(x.w/2);cl_mem out=alloc((long)n*4);launch("pool",n,device(x),out,x.c,x.h,x.w);return new FiARCCodec.Tensor(null,out,x.c,x.h/2,x.w/2);}
    public FiARCCodec.Tensor up(FiARCCodec.Tensor x){int n=x.c*x.h*x.w*4;cl_mem out=alloc((long)n*4);launch("up",n,device(x),out,x.c,x.h,x.w);return new FiARCCodec.Tensor(null,out,x.c,x.h*2,x.w*2);}
    final Map<String,cl_mem> axes=new HashMap<>();
    public FiARCCodec.Tensor resize(FiARCCodec.Tensor x,int h,int w){
        String key=x.h+":"+x.w+":"+h+":"+w;cl_mem axis=axes.get(key);
        if(axis==null){int[] a=Arrays.copyOf(PortableResize.axis(x.h,h),(h+w)*4);System.arraycopy(PortableResize.axis(x.w,w),0,a,h*4,w*4);axis=create((long)a.length*4,Pointer.to(a));axes.put(key,axis);}
        cl_mem out=alloc((long)x.c*h*w*4);launch("portable_up",x.c*h*w,device(x),out,axis,x.c,x.h,x.w,h,w,h==2*x.h&&w==2*x.w?1:0);return new FiARCCodec.Tensor(null,out,x.c,h,w);
    }
    public FiARCCodec.Tensor cat(FiARCCodec.Tensor a,FiARCCodec.Tensor b){FiARCCodec.require(a.h==b.h&&a.w==b.w,"Concat geometry");long an=(long)a.c*a.h*a.w*4,bn=(long)b.c*b.h*b.w*4;cl_mem out=alloc(an+bn);ok(clEnqueueCopyBuffer(queue,device(a),out,0,0,an,0,null,null));ok(clEnqueueCopyBuffer(queue,device(b),out,0,an,bn,0,null,null));return new FiARCCodec.Tensor(null,out,a.c+b.c,a.h,a.w);}
    public void resetHistory(){historyValid=false;}
    public void advanceHistory(float[] rec,int history){
        int plane=rec.length;if(histA==null||histPlane!=plane||histCount!=history){if(histA!=null)clReleaseMemObject(histA);if(histB!=null)clReleaseMemObject(histB);histA=create((long)plane*history*4,null);histB=create((long)plane*history*4,null);histPlane=plane;histCount=history;historyValid=false;}
        long bytes=(long)plane*4;
        if(!historyValid){ok(clEnqueueWriteBuffer(queue,histA,CL_TRUE,0,bytes,Pointer.to(rec),0,null,null));for(int i=0;i<history;i++)ok(clEnqueueCopyBuffer(queue,histA,histB,0,bytes*i,bytes,0,null,null));}
        else {if(history>1)ok(clEnqueueCopyBuffer(queue,histA,histB,bytes,0,bytes*(history-1),0,null,null));ok(clEnqueueWriteBuffer(queue,histB,CL_TRUE,bytes*(history-1),bytes,Pointer.to(rec),0,null,null));}
        cl_mem temp=histA;histA=histB;histB=temp;historyValid=true;
    }
    public FiARCCodec.Tensor historyFeatures(int history,int h,int w,float mean,float inv){FiARCCodec.require(historyValid&&histCount==history&&histPlane==h*w,"GPU history missing");cl_mem out=alloc((long)history*h*w*4);launch("history_features",history*h*w,histA,out,h*w,history,mean,inv);return new FiARCCodec.Tensor(null,out,history,h,w);}
    public void close(){
        if(queue!=null)clFinish(queue);
        if(axes!=null){for(cl_mem a:axes.values())clReleaseMemObject(a);axes.clear();}
        if(kernels!=null){for(cl_kernel k:kernels.values())clReleaseKernel(k);kernels.clear();}
        if(pool!=null){for(Buffer b:pool)clReleaseMemObject(b.mem);pool.clear();}
        if(layerBuffers!=null){for(cl_mem[] a:layerBuffers.values())for(cl_mem m:a)clReleaseMemObject(m);layerBuffers.clear();}
        if(table!=null){clReleaseMemObject(table);table=null;}if(histA!=null){clReleaseMemObject(histA);histA=null;}if(histB!=null){clReleaseMemObject(histB);histB=null;}
        if(guard!=null){clReleaseMemObject(guard);guard=null;}
        if(graphProgram!=null){clReleaseProgram(graphProgram);graphProgram=null;}super.close();
    }
}
