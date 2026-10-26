package org.fiarc.strictjava;
import org.jocl.*;
import static org.jocl.CL.*;
import java.util.*;

/** All prediction, rANS, correction and recurrent history stay on GPU. */
public class FiARCStrictGPU extends FiARCOpenCLResident {
    FiARCCodec codec;cl_mem dummy,error,expTable,seedOffsets,finalOffsets,factors,finalFactors,evenWeights;
    final List<cl_mem> persistent=new ArrayList<>(),groupMemory=new ArrayList<>();
    FiARCCodec.Packet packet;Phase[] phases;cl_mem[] escapes,escapeCounters;int frameInPacket;
    static class Phase {int ns,bytes,rareCount;cl_mem body,offsets,state,rare,counter;}
    public FiARCStrictGPU(int gpu){super(gpu);}
    cl_mem keep(int[] a,List<cl_mem> owner){cl_mem b=create((long)a.length*4,Pointer.to(a.length==0?new int[]{0}:a));owner.add(b);return b;}
    cl_mem keep(float[] a,List<cl_mem> owner){cl_mem b=create((long)a.length*4,Pointer.to(a.length==0?new float[]{0}:a));owner.add(b);return b;}
    cl_mem zero(int n){cl_mem m=alloc((long)Math.max(1,n)*4);ok(clEnqueueFillBuffer(queue,m,Pointer.to(new int[]{0}),4,0,(long)Math.max(1,n)*4,0,null,null));return m;}
    void initialize(FiARCCodec c){
        if(codec!=null){FiARCCodec.require(codec==c,"GPU reused for another codec");return;}codec=c;
        dummy=keep(new int[1],persistent);error=guard;expTable=keep(c.exp,persistent);
        seedOffsets=keep(c.seedOffsets,persistent);finalOffsets=keep(c.finalOffsets,persistent);evenWeights=keep(c.evenWeights,persistent);
        int[] cf=new int[65535*2];for(int i=0;i<65535;i++){long bits=Double.doubleToRawLongBits(c.seedFactors[i]);cf[2*i]=(int)bits;cf[2*i+1]=(int)(bits>>>32);}factors=keep(cf,persistent);finalFactors=factors;
        if(c.seedStep!=c.finalStep){for(int i=0;i<65535;i++){long bits=Double.doubleToRawLongBits(c.finalFactors[i]);cf[2*i]=(int)bits;cf[2*i+1]=(int)(bits>>>32);}finalFactors=keep(cf,persistent);}
        histPlane=c.h*c.w;histCount=c.history;allocateHistory();
    }
    void allocateHistory(){histA=create((long)histPlane*histCount*4,null);}
    void updateHistory(cl_mem normalized){launch("history_shift",histPlane,histA,normalized,histPlane,histCount,historyValid?1:0);historyValid=true;}
    void prepare(FiARCCodec.Packet p){
        if(packet==p)return;for(cl_mem m:groupMemory)clReleaseMemObject(m);groupMemory.clear();packet=p;frameInPacket=0;phases=new Phase[4];
        for(int k=0;k<4;k++){
            FiARCCodec.Rans r=p.phases[k];Phase s=new Phase();s.ns=r.streams.length;s.rareCount=r.rare.length;
            int[] offsets=new int[s.ns+1],state=new int[s.ns*2],rare=new int[r.rare.length];
            for(int j=0;j<s.ns;j++){offsets[j+1]=Math.addExact(offsets[j],r.streams[j].length);state[2*j]=r.state[j];state[2*j+1]=r.cursor[j];}
            s.bytes=offsets[s.ns];byte[] body=new byte[Math.max(4,s.bytes)];for(int j=0;j<s.ns;j++)System.arraycopy(r.streams[j],0,body,offsets[j],r.streams[j].length);
            s.body=create(body.length,Pointer.to(body));groupMemory.add(s.body);for(int j=0;j<rare.length;j++)rare[j]=r.rare[j];
            s.offsets=keep(offsets,groupMemory);s.state=keep(state,groupMemory);s.rare=keep(rare,groupMemory);s.counter=keep(new int[1],groupMemory);phases[k]=s;
        }
        escapes=new cl_mem[2];escapeCounters=new cl_mem[2];for(int k=0;k<2;k++){escapes[k]=keep(p.escapes[k],groupMemory);escapeCounters[k]=keep(new int[1],groupMemory);}
    }
    cl_mem pixel(int mode,cl_mem a,cl_mem b,cl_mem c,cl_mem d,cl_mem second){
        int n=histPlane,count=mode==8?n*histCount:n,size=mode==3?n*(codec.lattice?4:3):count;
        cl_mem out=alloc((long)size*4);
        launch("pixels",count,a==null?dummy:a,b==null?dummy:b,c==null?dummy:c,d==null?dummy:d,out,second==null?dummy:second,error,
            mode,count,codec.h,codec.w,histCount,Float.floatToRawIntBits(codec.mean),Float.floatToRawIntBits(codec.std),Float.floatToRawIntBits(codec.inv),Float.floatToRawIntBits(codec.gain),codec.stride,codec.lattice?1:0,Float.floatToRawIntBits(codec.conditioningTolerance));
        return out;
    }
    FiARCCodec.Tensor tensor(cl_mem m,int c){return new FiARCCodec.Tensor(null,m,c,codec.h,codec.w);}
    FiARCCodec.Tensor conv(String name,FiARCCodec.Tensor x,int kind){return convolution(x,codec.layers.get(name),kind,codec.lut,true);}
    FiARCCodec.Tensor targetUp(FiARCCodec.Tensor x,FiARCCodec.Tensor target){FiARCCodec.Tensor out=resize(x,target.h,target.w);release(x);return out;}
    FiARCCodec.Tensor merge(FiARCCodec.Tensor a,FiARCCodec.Tensor b){FiARCCodec.require(a.h==b.h&&a.w==b.w,"Concat geometry");return new FiARCCodec.Tensor(null,new Pair(a,b),a.c+b.c,a.h,a.w);}
    FiARCCodec.Tensor block(String name,FiARCCodec.Tensor x,int kind){return conv(name+".2",conv(name+".0",x,kind),kind);}
    cl_mem forecast(){
        if(!historyValid)return pixel(0,null,null,null,null,null);
        FiARCCodec.Tensor fine=block("base.fine",tensor(pixel(8,histA,null,null,null,null),histCount),2);
        FiARCCodec.Tensor mid=block("base.middle",block("base.down",pool(fine),2),2);
        FiARCCodec.Tensor out=conv("base.out",block("base.up",merge(fine,targetUp(mid,fine)),2),0);
        cl_mem result=pixel(1,(cl_mem)out.device,histA,null,null,null);release(out);return result;
    }
    cl_mem spatial(cl_mem features){
        FiARCCodec.Tensor fine=block("refiner.fine",tensor(features,codec.lattice?4:3),1),mid=block("refiner.mid",pool(fine),1),low=block("refiner.low",pool(mid),1);
        FiARCCodec.Tensor u=block("refiner.upmid",merge(mid,targetUp(low,mid)),1);
        return (cl_mem)conv("refiner.out",block("refiner.upfine",merge(fine,targetUp(u,fine)),1),0).device;
    }
    cl_mem prior(cl_mem pred){return (cl_mem)conv("prior.2",conv("prior.0",tensor(pixel(2,pred,null,null,null,null),1),1),0).device;}
    cl_mem prefix(cl_mem counts,cl_mem counter,int count,int available){cl_mem pos=alloc((long)count*4);launch("prefix",1,counts,counter,pos,error,count,available,frameInPacket+1==packet.n?1:0);return pos;}
    void phase(int stage,cl_mem logits,cl_mem q,cl_mem seed,int h,int w){
        Phase s=phases[stage];int odd=stage&1,n=(h*w+1-odd)/2;FiARCCodec.require(packet.phases[stage].counts[frameInPacket]==n,"GPU phase counts");
        cl_mem cdf=alloc((long)n*9*4),labels=alloc((long)n*4),counts=alloc((long)s.ns*4);
        launch("cdf",n,logits,q,seed,expTable,stage<2?seedOffsets:finalOffsets,cdf,error,h,w,histPlane,stage<2?codec.stride:1,odd,stage<2?1:0,codec.w);
        launch("rans",s.ns,s.body,s.offsets,s.state,cdf,labels,counts,error,n,s.ns,s.bytes,frameInPacket+1==packet.n?1:0);
        cl_mem positions=prefix(counts,s.counter,s.ns,s.rareCount);
        launch("symbols",s.ns,labels,s.rare,positions,q,error,n,s.ns,w,odd,s.rareCount);
        release(cdf);release(labels);release(counts);release(positions);
    }
    cl_mem escaped(int stage,cl_mem q,int n){
        int blocks=(n+255)/256;cl_mem counts=alloc((long)blocks*4);launch("escape_count",n,q,counts,n);
        cl_mem positions=prefix(counts,escapeCounters[stage],blocks,packet.escapes[stage].length),out=alloc((long)n*4);
        launch("escape_scatter",n,q,escapes[stage],positions,out,error,n,packet.escapes[stage].length);release(counts);release(positions);return out;
    }
    float[] decodeFrame(FiARCCodec c,FiARCCodec.Packet p){
        initialize(c);prepare(p);begin();ok(clEnqueueFillBuffer(queue,error,Pointer.to(new int[]{0}),4,0,4,0,null,null));
        cl_mem pred=forecast(),prior=prior(pred);
        int h=c.h,w=c.w,n=h*w,lh=(h+1)/2,lw=(w+1)/2;cl_mem low=zero(lh*lw);
        phase(0,prior,low,dummy,lh,lw);phase(1,prior,low,dummy,lh,lw);
        cl_mem lowEsc=escaped(0,low,lh*lw),seedq=alloc((long)n*4),seed=pixel(5,pred,low,lowEsc,factors,seedq);
        release(prior);release(low);release(lowEsc);
        cl_mem net=spatial(pixel(3,pred,seedq,seed,null,null)),post=pixel(4,pred,net,seedq,seed,null),q=zero(n);
        release(pred);release(seed);
        phase(2,net,q,seedq,h,w);phase(3,net,q,seedq,h,w);release(net);cl_mem esc=escaped(1,q,n),rec=pixel(6,post,q,esc,finalFactors,null);
        if(c.gain!=0)rec=pixel(6,pixel(7,post,rec,seedq,evenWeights,null),q,esc,finalFactors,null);
        cl_mem normalized=pixel(2,rec,null,null,null,null);long bytes=(long)n*4;
        updateHistory(normalized);
        float[] out=new float[n];ok(clEnqueueReadBuffer(queue,rec,CL_TRUE,0,bytes,Pointer.to(out),0,null,null));int[] invalid={0};ok(clEnqueueReadBuffer(queue,error,CL_TRUE,0,4,Pointer.to(invalid),0,null,null));
        FiARCCodec.require(invalid[0]==0,"strict GPU rejected frame "+frameInPacket+": "+invalid[0]);frameInPacket++;return out;
    }
    public void close(){if(queue!=null)clFinish(queue);if(persistent!=null){for(cl_mem b:persistent)clReleaseMemObject(b);persistent.clear();}if(groupMemory!=null){for(cl_mem b:groupMemory)clReleaseMemObject(b);groupMemory.clear();}super.close();}
}
