package org.fiarc.strictjava;
import org.jocl.*;
import static org.jocl.CL.*;
import java.util.*;
import java.util.concurrent.*;
import java.nio.*;

/** Same bitstream and global arithmetic, host-backed layers and tiled GPU work.
 * Tiles are computation units, NOT independently decoded spatial substreams. */
public strictfp final class FiARCTiledGPU extends FiARCStrictGPU {
    final int tile,threads;final ExecutorService workers;
    float[][] history;int oldest;
    cl_mem tileInput,tilePacked,tileOutput,tileScale,tileAxes;
    long inputBytes,packedBytes,outputBytes,axesBytes;
    ByteBuffer inputHost,outputHost;
    public long tilesDecoded,maxPassNanos,tileNanos;
    private static int checkedIndex(int index,int tile){FiARCCodec.require(tile>=32&&tile<=1024,"GPU tile must be 32..1024 pixels");return index;}
    public FiARCTiledGPU(int index,int tile){
        super(checkedIndex(index,tile));this.tile=tile;
        threads=Math.max(1,Math.min(8,Integer.getInteger("fiarc.tile.threads",Runtime.getRuntime().availableProcessors())));
        workers=Executors.newFixedThreadPool(threads);
    }
    abstract static class Host {
        final int c,h,w;Host(int c,int h,int w){this.c=c;this.h=h;this.w=w;}
        abstract void row(int c,int y,int x,int n,float[] target,int at);
        void putRow(int c,int y,int x,int n,FloatBuffer target,float[] scratch){row(c,y,x,n,scratch,0);target.put(scratch,0,n);}
        float max(int c0,int c1){float max=0x1p-40f;float[] row=new float[w];for(int c=c0;c<c1;c++)for(int y=0;y<h;y++){row(c,y,0,w,row,0);for(float v:row){FiARCCodec.require(Float.isFinite(v),"Nonfinite feature");max=Math.max(max,Math.abs(v));}}return max;}
    }
    static final class Dense extends Host {
        final float[] data;Dense(float[] a,int c,int h,int w){super(c,h,w);data=a;}
        void row(int c,int y,int x,int n,float[] target,int at){System.arraycopy(data,(c*h+y)*w+x,target,at,n);}
        void putRow(int c,int y,int x,int n,FloatBuffer target,float[] scratch){target.put(data,(c*h+y)*w+x,n);}
        float max(int c0,int c1){float max=0x1p-40f;for(int i=c0*h*w,end=c1*h*w;i<end;i++){float v=data[i];FiARCCodec.require(Float.isFinite(v),"Nonfinite feature");max=Math.max(max,Math.abs(v));}return max;}
    }
    static final class Joined extends Host {
        final Host a,b;Joined(Host a,Host b){super(a.c+b.c,a.h,a.w);FiARCCodec.require(a.h==b.h&&a.w==b.w,"Tile concat geometry");this.a=a;this.b=b;}
        void row(int c,int y,int x,int n,float[] target,int at){if(c<a.c)a.row(c,y,x,n,target,at);else b.row(c-a.c,y,x,n,target,at);}
        void putRow(int c,int y,int x,int n,FloatBuffer target,float[] scratch){if(c<a.c)a.putRow(c,y,x,n,target,scratch);else b.putRow(c-a.c,y,x,n,target,scratch);}
        float max(int c0,int c1){float v=0x1p-40f;if(c0<a.c)v=Math.max(v,a.max(c0,Math.min(c1,a.c)));if(c1>a.c)v=Math.max(v,b.max(Math.max(0,c0-a.c),c1-a.c));return v;}
    }
    final class History extends Host {
        final float[] last;
        History(){super(histCount,codec.h,codec.w);last=history[(oldest+histCount-1)%histCount];}
        void row(int c,int y,int x,int n,float[] target,int at){float[] src=history[(oldest+c)%histCount];int p=y*w+x;if(c==histCount-1)System.arraycopy(src,p,target,at,n);else for(int i=0;i<n;i++)target[at+i]=src[p+i]-last[p+i];}
        float max(int c0,int c1){float m=0x1p-40f;for(int c=c0;c<c1;c++){float[] src=history[(oldest+c)%histCount];for(int i=0;i<src.length;i++){float v=c==histCount-1?src[i]:src[i]-last[i];FiARCCodec.require(Float.isFinite(v),"Nonfinite history");m=Math.max(m,Math.abs(v));}}return m;}
    }
    interface Work {void run(int from,int to);}
    void parallel(int n,Work work){
        if(n<2||threads==1){work.run(0,n);return;}List<Future<?>> jobs=new ArrayList<>();int tasks=Math.min(threads,n);
        for(int j=0;j<tasks;j++){final int from=j*n/tasks,to=(j+1)*n/tasks;jobs.add(workers.submit(()->work.run(from,to)));}
        try{for(Future<?> f:jobs)f.get();}catch(InterruptedException e){for(Future<?> f:jobs)f.cancel(true);Thread.currentThread().interrupt();throw new CancellationException();}catch(ExecutionException e){for(Future<?> f:jobs)f.cancel(true);throw new IllegalStateException("Host tile preparation failed",e.getCause());}
    }
    float globalScale(Host a){long t=System.nanoTime();float[] max=new float[a.c];parallel(a.c,(from,to)->{for(int c=from;c<to;c++)max[c]=a.max(c,c+1);});float m=0x1p-40f;for(float v:max)m=Math.max(m,v);maxPassNanos+=System.nanoTime()-t;return Math.scalb(1.f,StrictCorrection.activationExponent(Float.floatToRawIntBits(m)));}
    @Override void allocateHistory(){
        // Host-backed layers deliberately trade system memory for VRAM. Reject
        // obviously undersized heaps before allocating multi-gigabyte tensors.
        long recommended=Math.addExact((long)histPlane*(histCount+320)*4,64L<<20);
        FiARCCodec.require(Runtime.getRuntime().maxMemory()>=recommended,
            "Low VRAM mode needs approximately "+((recommended+(1L<<20)-1)>>20)+
            " MiB of Java heap for this image. Increase ImageJ Memory & Threads, restart ImageJ, or use Fast mode. Low VRAM does not mean low system RAM.");
        history=new float[histCount][];oldest=0;
    }
    @Override public void resetHistory(){super.resetHistory();if(history!=null)Arrays.fill(history,null);oldest=0;}
    @Override void updateHistory(cl_mem normalized){float[] a=new float[histPlane];ok(clEnqueueReadBuffer(queue,normalized,CL_TRUE,0,(long)histPlane*4,Pointer.to(a),0,null,null));if(!historyValid)Arrays.fill(history,a);else{history[oldest]=a;oldest=(oldest+1)%histCount;}historyValid=true;}
    @Override public long historyBytes(){return 0;}
    public long hostHistoryBytes(){if(history==null)return 0;Set<float[]> seen=Collections.newSetFromMap(new IdentityHashMap<float[],Boolean>());long n=0;for(float[] a:history)if(a!=null&&seen.add(a))n+=(long)a.length*4;return n;}
    @Override public long scratchBytes(){return super.scratchBytes()+inputBytes+packedBytes+outputBytes+axesBytes+(tileScale==null?0:4);}
    void inputCapacity(int n){long bytes=(long)n*4;if(bytes>inputBytes){if(tileInput!=null)clReleaseMemObject(tileInput);tileInput=create(bytes,null);inputBytes=bytes;}if(inputHost==null||inputHost.capacity()<bytes)inputHost=ByteBuffer.allocateDirect(Math.toIntExact(bytes)).order(ByteOrder.nativeOrder());}
    void outputCapacity(int n){long bytes=(long)n*4;if(bytes>outputBytes){if(tileOutput!=null)clReleaseMemObject(tileOutput);tileOutput=create(bytes,null);outputBytes=bytes;}if(outputHost==null||outputHost.capacity()<bytes)outputHost=ByteBuffer.allocateDirect(Math.toIntExact(bytes)).order(ByteOrder.nativeOrder());}
    void packedCapacity(long bytes){if(bytes>packedBytes){if(tilePacked!=null)clReleaseMemObject(tilePacked);tilePacked=create(bytes,null);packedBytes=bytes;}}
    void gather(Host a,int y,int x,int hh,int ww){
        int n=a.c*hh*ww;inputCapacity(n);int x0=Math.max(0,x),x1=Math.min(a.w,x+ww),left=x0-x,right=x+ww-x1;
        parallel(a.c,(from,to)->{FloatBuffer view=inputHost.asFloatBuffer();float[] zero=new float[ww],scratch=new float[ww];for(int c=from;c<to;c++)for(int row=0;row<hh;row++){view.position((c*hh+row)*ww);int yy=y+row;if(yy<0||yy>=a.h)view.put(zero);else{if(left>0)view.put(zero,0,left);a.putRow(c,yy,x0,x1-x0,view,scratch);if(right>0)view.put(zero,0,right);}}});
        ok(clEnqueueWriteBuffer(queue,tileInput,CL_TRUE,0,(long)n*4,Pointer.to(inputHost),0,null,null));
    }
    cl_mem[] weights(FiARCCodec.Layer l){cl_mem[] ls=layerBuffers.get(l);if(ls!=null)return ls;int ci4=(l.ci+3)/4;int[] weight=new int[l.co*l.kh*l.kw*ci4];for(int o=0;o<l.co;o++)for(int c=0;c<l.ci;c++)for(int ky=0;ky<l.kh;ky++)for(int kx=0;kx<l.kw;kx++){int v=l.weights[((o*l.ci+c)*l.kh+ky)*l.kw+kx]&255;weight[((o*l.kh+ky)*l.kw+kx)*ci4+c/4]|=v<<(8*(c%4));}
        ls=new cl_mem[]{create((long)weight.length*4,Pointer.to(weight)),create((long)l.exponents.length*4,Pointer.to(l.exponents)),create((long)l.bias.length*4,Pointer.to(l.bias))};layerBuffers.put(l,ls);return ls;
    }
    Dense convolution(String name,Host a,int act){
        codec.checkCancelled();FiARCCodec.Layer l=codec.layers.get(name);FiARCCodec.require(l!=null&&l.ci==a.c,"Tiled conv channels");float scale=globalScale(a);
        if(tileScale==null)tileScale=create(4,null);ok(clEnqueueWriteBuffer(queue,tileScale,CL_TRUE,0,4,Pointer.to(new float[]{scale}),0,null,null));
        if(table==null)table=create((long)codec.lut.length*4,Pointer.to(codec.lut));cl_mem[] ls=weights(l);int ci4=(a.c+3)/4,ry=l.kh/2,rx=l.kw/2;float[] result=new float[l.co*a.h*a.w];
        for(int y=0;y<a.h;y+=tile)for(int x=0;x<a.w;x+=tile){
            codec.checkCancelled();long t=System.nanoTime();int oh=Math.min(tile,a.h-y),ow=Math.min(tile,a.w-x),hh=oh+2*ry,ww=ow+2*rx,plane=hh*ww;
            gather(a,y-ry,x-rx,hh,ww);packedCapacity((long)plane*ci4*8);outputCapacity(plane*l.co);
            launch("pack",plane*ci4,tileInput,tileInput,tileScale,tilePacked,plane,a.c,a.c,guard);
            boolean mma=kernels.containsKey("conv_mma");cl_kernel k=kernels.get(mma?"conv_mma":"conv");args(k,tilePacked,ls[0],tileOutput,tileScale,ls[1],ls[2],table,hh,ww,ci4,l.co,l.kh,l.kw,act);
            ok(clEnqueueNDRangeKernel(queue,k,2,null,mma?new long[]{((plane+15L)/16)*32,(l.co+7L)/8}:new long[]{((plane+31L)/32)*8,((l.co+15L)/16)*8},mma?new long[]{32,1}:new long[]{8,8},0,null,null));
            ok(clEnqueueReadBuffer(queue,tileOutput,CL_TRUE,0,(long)plane*l.co*4,Pointer.to(outputHost),0,null,null));
            final int yy=y,xx=x;parallel(l.co,(from,to)->{FloatBuffer view=outputHost.asFloatBuffer();for(int c=from;c<to;c++)for(int row=0;row<oh;row++){view.position((c*hh+row+ry)*ww+rx);view.get(result,(c*a.h+yy+row)*a.w+xx,ow);}});tilesDecoded++;tileNanos+=System.nanoTime()-t;
        }
        return new Dense(result,l.co,a.h,a.w);
    }
    Dense block(String name,Host a,int act){return convolution(name+".2",convolution(name+".0",a,act),act);}
    Dense pool(Dense a){int hh=a.h/2,ww=a.w/2;float[] out=new float[a.c*hh*ww];parallel(a.c,(from,to)->{for(int c=from;c<to;c++)for(int y=0;y<hh;y++)for(int x=0;x<ww;x++){int i=(c*a.h+2*y)*a.w+2*x;out[(c*hh+y)*ww+x]=(((a.data[i]+a.data[i+1])+a.data[i+a.w])+a.data[i+a.w+1])*.25f;}});return new Dense(out,a.c,hh,ww);}
    Dense up(Host a,Host target){
        int hh=target.h,ww=target.w;int[] ay=PortableResize.axis(a.h,hh),ax=PortableResize.axis(a.w,ww);float[] result=new float[a.c*hh*ww];
        for(int y=0;y<hh;y+=tile)for(int x=0;x<ww;x+=tile){
            codec.checkCancelled();int oh=Math.min(tile,hh-y),ow=Math.min(tile,ww-x),y0=ay[y*4],y1=ay[(y+oh-1)*4+1],x0=ax[x*4],x1=ax[(x+ow-1)*4+1],ih=y1-y0+1,iw=x1-x0+1;
            gather(a,y0,x0,ih,iw);outputCapacity(a.c*oh*ow);int[] axes=new int[(oh+ow)*4];for(int j=0;j<oh;j++){System.arraycopy(ay,(y+j)*4,axes,j*4,4);axes[j*4]-=y0;axes[j*4+1]-=y0;}for(int j=0;j<ow;j++){System.arraycopy(ax,(x+j)*4,axes,(oh+j)*4,4);axes[(oh+j)*4]-=x0;axes[(oh+j)*4+1]-=x0;}
            long bytes=(long)axes.length*4;if(bytes>axesBytes){if(tileAxes!=null)clReleaseMemObject(tileAxes);tileAxes=create(bytes,null);axesBytes=bytes;}ok(clEnqueueWriteBuffer(queue,tileAxes,CL_TRUE,0,bytes,Pointer.to(axes),0,null,null));
            launch("portable_up",a.c*oh*ow,tileInput,tileOutput,tileAxes,a.c,ih,iw,oh,ow,!codec.cli?-1:hh==2*a.h&&ww==2*a.w?1:0);
            ok(clEnqueueReadBuffer(queue,tileOutput,CL_TRUE,0,(long)a.c*oh*ow*4,Pointer.to(outputHost),0,null,null));final int yy=y,xx=x;parallel(a.c,(from,to)->{FloatBuffer view=outputHost.asFloatBuffer();for(int c=from;c<to;c++)for(int row=0;row<oh;row++){view.position((c*oh+row)*ow);view.get(result,(c*hh+yy+row)*ww+xx,ow);}});
        }return new Dense(result,a.c,hh,ww);
    }
    Dense read(cl_mem m,int channels){float[] a=new float[channels*histPlane];ok(clEnqueueReadBuffer(queue,m,CL_TRUE,0,(long)a.length*4,Pointer.to(a),0,null,null));return new Dense(a,channels,codec.h,codec.w);}
    cl_mem upload(Dense a){cl_mem m=alloc((long)a.data.length*4);ok(clEnqueueWriteBuffer(queue,m,CL_TRUE,0,(long)a.data.length*4,Pointer.to(a.data),0,null,null));return m;}
    @Override cl_mem forecast(){
        if(!historyValid)return pixel(0,null,null,null,null,null);
        History input=new History();Dense fine=block("base.fine",input,2),mid=block("base.middle",block("base.down",pool(fine),2),2);
        Dense out=convolution("base.out",block("base.up",new Joined(fine,up(mid,fine)),2),0);float[] last=input.last;
        parallel(out.h,(from,to)->{for(int i=from*out.w,end=to*out.w;i<end;i++)out.data[i]=(out.data[i]+last[i])*codec.std+codec.mean;});return upload(out);
    }
    @Override cl_mem prior(cl_mem pred){cl_mem normalized=pixel(2,pred,null,null,null,null);Dense input=read(normalized,1);release(normalized);return upload(convolution("prior.2",convolution("prior.0",input,1),0));}
    @Override cl_mem spatial(cl_mem features){
        Dense input=read(features,codec.lattice?4:3);release(features);Dense fine=block("refiner.fine",input,1),mid=block("refiner.mid",pool(fine),1),low=block("refiner.low",pool(mid),1);
        Dense u=block("refiner.upmid",new Joined(mid,up(low,mid)),1);return upload(convolution("refiner.out",block("refiner.upfine",new Joined(fine,up(u,fine)),1),0));
    }
    @Override public void close(){
        if(queue!=null)clFinish(queue);if(workers!=null)workers.shutdownNow();
        if(tileInput!=null){clReleaseMemObject(tileInput);tileInput=null;}if(tilePacked!=null){clReleaseMemObject(tilePacked);tilePacked=null;}if(tileOutput!=null){clReleaseMemObject(tileOutput);tileOutput=null;}if(tileScale!=null){clReleaseMemObject(tileScale);tileScale=null;}if(tileAxes!=null){clReleaseMemObject(tileAxes);tileAxes=null;}
        inputBytes=packedBytes=outputBytes=axesBytes=0;inputHost=outputHost=null;history=null;super.close();
    }
}
