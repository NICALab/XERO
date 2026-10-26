package org.fiarc.strictjava;
import java.nio.file.*;
import java.nio.*;
import java.util.*;
import com.google.gson.*;
import ij.ImageStack;
import ij.process.FloatProcessor;

public class FiARCStandaloneBenchmark {
    public static void main(String[] args) throws Exception {
        Path archive=Paths.get(args[0]),reference=Paths.get(args[1]),out=Paths.get(args[3]);int frames=Integer.parseInt(args[2]),gpu=Integer.parseInt(args[4]);
        byte[] expected=Files.readAllBytes(reference);JsonArray rows=new JsonArray();boolean allExact=true;
        try(FiARCCodec codec=new FiARCCodec(archive,new FiARCStrictGPU(gpu))){
            codec.preloadPackets();
            if(frames%64!=0)throw new IllegalArgumentException("Benchmark on complete entropy packets");
            for(int repeat=0;repeat<4;repeat++){
                final boolean[] exact={true};final ImageStack stack=new ImageStack(codec.w,codec.h);long started=System.nanoTime();
                codec.range(1,frames,(frame,pixels)->stack.addSlice("Frame "+frame,new FloatProcessor(codec.w,codec.h,pixels)),()->false);
                double sec=(System.nanoTime()-started)/1e9;
                ByteBuffer target=ByteBuffer.wrap(expected).order(ByteOrder.LITTLE_ENDIAN);for(int i=1;i<=stack.size();i++)for(float v:(float[])stack.getPixels(i))if(Float.floatToRawIntBits(v)!=target.getInt())exact[0]=false;
                JsonObject r=new JsonObject();r.addProperty("repeat",repeat);r.addProperty("warmup",repeat==0);r.addProperty("bit_exact",exact[0]);r.addProperty("frames",stack.size());r.addProperty("seconds",sec);r.addProperty("ram_decode_fps",frames/sec);r.addProperty("core_ms_per_frame",codec.coreNanos/1e6/codec.processedFrames);rows.add(r);allExact&=exact[0];System.out.println(r);
            }
            JsonObject report=new JsonObject();report.addProperty("bit_exact",allExact);report.addProperty("gpu",gpu);report.addProperty("device",((FiARCOpenCLResident)codec.engine).deviceName);report.addProperty("dp4a",((FiARCOpenCLResident)codec.engine).dp4a);report.addProperty("width",codec.w);report.addProperty("height",codec.h);report.add("runs",rows);Files.createDirectories(out.getParent());Files.write(out,new GsonBuilder().setPrettyPrinting().create().toJson(report).getBytes("UTF-8"));if(!allExact)throw new IllegalStateException("Cross-device mismatch");
        }
    }
}
