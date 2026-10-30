package org.fiarc.strictjava;

import com.google.gson.*;
import java.io.*;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Arrays;

/** Persistent, length-framed Python bridge. Only compressed inputs enter stdin;
 * selected pixels leave stdout. No decoded image is ever written to disk. */
public final class XeroZarrBridge {
    static JsonObject readJson(DataInputStream in)throws Exception {
        int n=in.readInt();FiARCCodec.require(n>0&&n<=32*1024*1024,"Invalid request header");byte[] b=new byte[n];in.readFully(b);return FiARCCodec.json(b);
    }
    static void writeJson(DataOutputStream out,JsonObject j)throws Exception {
        byte[] b=j.toString().getBytes(StandardCharsets.UTF_8);out.writeInt(b.length);out.write(b);out.flush();
    }
    public static void main(String[] args)throws Exception {
        DataInputStream input=new DataInputStream(new BufferedInputStream(System.in,1<<20));
        DataOutputStream output=new DataOutputStream(new BufferedOutputStream(System.out,1<<20));
        JsonObject cfg=readJson(input);int h=cfg.get("height").getAsInt(),w=cfg.get("width").getAsInt(),restart=cfg.get("restart").getAsInt();
        FiARCCodec.require(h>=4&&w>=4&&h<=4096&&w<=4096&&restart>=128&&restart<=1048576&&(restart&(restart-1))==0,"Invalid decoder shape");
        JsonObject manifest=new JsonObject();for(String k:new String[]{"profile","restart","tolerance","model_sha256"})manifest.add(k,cfg.get(k));
        JsonArray shape=new JsonArray();shape.add(restart);shape.add(h);shape.add(w);manifest.add("shape",shape);
        manifest.addProperty("history","repeat_first_decoded");manifest.addProperty("model","model.npz");
        JsonArray groups=new JsonArray();for(int i=0;i<restart;i+=64){JsonObject g=new JsonObject();g.addProperty("start",i);g.addProperty("frames",Math.min(64,restart-i));groups.add(g);}manifest.add("groups",groups);
        XeroSource.Memory source=new XeroSource.Memory(manifest,XeroSource.unpackModel(cfg));
        String mode=args.length==0?"cpu":args[0];
        FiARCCodec.Conv engine=mode.equals("cpu")?new FiARCCodec.CPU():new FiARCStrictGPU(Integer.parseInt(mode.substring(4)));
        try(FiARCCodec codec=new FiARCCodec(source,engine)) {
            JsonObject ready=new JsonObject();ready.addProperty("ready",true);writeJson(output,ready);
            for(;;) {
                JsonObject request;try{request=readJson(input);}catch(EOFException end){break;}
                int n=request.get("bytes").getAsInt();FiARCCodec.require(n>=44&&n<=FiARCCodec.MAX_BLOB,"Oversized chunk");
                byte[] chunk=new byte[n];input.readFully(chunk);
                FiARCCodec.require(new String(chunk,0,8,StandardCharsets.US_ASCII).equals("XEROZ001"),"Invalid chunk magic");
                MessageDigest digest=MessageDigest.getInstance("SHA-256");digest.update(chunk,40,n-40);
                FiARCCodec.require(Arrays.equals(digest.digest(),Arrays.copyOfRange(chunk,8,40)),"Chunk integrity failure");
                int hs=FiARCCodec.le(chunk).getInt(40);FiARCCodec.require(hs>0&&hs<=1024*1024&&hs<=n-44,"Invalid chunk header");
                JsonObject header=FiARCCodec.json(Arrays.copyOfRange(chunk,44,44+hs));int count=header.get("frames").getAsInt();
                FiARCCodec.require(count>0&&count<=restart&&header.get("origin").getAsInt()%restart==0,"Invalid restart chunk");
                source.set(chunk,header,44+hs);
                JsonArray slices=request.getAsJsonArray("selection");int[][] ranges=new int[3][3];int[] dims={count,h,w};long values=1;
                for(int a=0;a<3;a++){JsonArray r=slices.get(a).getAsJsonArray();for(int j=0;j<3;j++)ranges[a][j]=r.get(j).getAsInt();int lo=ranges[a][0],hi=ranges[a][1],step=ranges[a][2];FiARCCodec.require(lo>=0&&hi>lo&&hi<=dims[a]&&step>0,"Invalid selection");values=Math.multiplyExact(values,(long)((hi-lo+step-1)/step));}
                FiARCCodec.require(values*4<=256L*1024*1024,"Requested output exceeds 256 MiB; use smaller slices");
                final int[] t=ranges[0],y=ranges[1],x=ranges[2];int row=(x[1]-x[0]+x[2]-1)/x[2];
                byte[] pixels=new byte[row*((y[1]-y[0]+y[2]-1)/y[2])*4];ByteBuffer floats=ByteBuffer.wrap(pixels).order(ByteOrder.LITTLE_ENDIAN);
                JsonObject response=new JsonObject();response.addProperty("bytes",values*4);writeJson(output,response);long started=System.nanoTime();
                codec.range(t[0]+1,t[1],(frame,data)->{
                    if((frame-1-t[0])%t[2]!=0)return;floats.clear();
                    if(x[0]==0&&x[1]==w&&x[2]==1&&y[2]==1)floats.asFloatBuffer().put(data,y[0]*w,(y[1]-y[0])*w);
                    else if(x[2]==1){FloatBuffer fb=floats.asFloatBuffer();for(int yy=y[0];yy<y[1];yy+=y[2])fb.put(data,yy*w+x[0],x[1]-x[0]);}
                    else for(int yy=y[0];yy<y[1];yy+=y[2])for(int xx=x[0];xx<x[1];xx+=x[2])floats.putFloat(data[yy*w+xx]);
                    try{output.write(pixels);}catch(IOException e){throw new UncheckedIOException(e);}
                },()->false);
                JsonObject receipt=new JsonObject();receipt.addProperty("processed_frames",codec.processedFrames);receipt.addProperty("packets",codec.packetsRead);
                receipt.addProperty("core_seconds",codec.coreNanos/1e9);receipt.addProperty("java_seconds",(System.nanoTime()-started)/1e9);writeJson(output,receipt);
                source.chunk=null;
            }
        }
    }
}
