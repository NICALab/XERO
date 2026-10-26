package org.fiarc.strictjava;
import java.nio.*;
import java.nio.channels.*;
import java.nio.file.*;
import java.lang.management.*;
import com.google.gson.*;

/** Bounded output sink: reference comparison is separate from decode timing. */
public final class FiARCStreamBenchmark {
 public static void main(String[] args)throws Exception{
  Path archive=Paths.get(args[0]),reference=Paths.get(args[1]),out=Paths.get(args[2]);boolean gpu=args[3].equals("gpu");int device=Integer.parseInt(args[4]),last=Integer.parseInt(args[5]),repeats=Integer.parseInt(args[6]);
  FiARCCodec.Conv engine=gpu?new FiARCStrictGPU(device):new FiARCCodec.CPU();JsonArray runs=new JsonArray();
  try(FiARCCodec c=new FiARCCodec(archive,engine);FileChannel f=FileChannel.open(reference)){
   c.preloadPackets();final long[] different={0};final double[] max={0};
   c.range(1,last,(frame,pixels)->{try{
    ByteBuffer b=ByteBuffer.allocate(pixels.length*4).order(ByteOrder.LITTLE_ENDIAN);long pos=(long)(frame-1)*pixels.length*4;while(b.hasRemaining()){int n=f.read(b,pos+b.position());if(n<0)throw new IllegalStateException("Short reference");}b.flip();
    for(float v:pixels){float ref=b.getFloat();if(Float.floatToRawIntBits(v)!=Float.floatToRawIntBits(ref))different[0]++;max[0]=Math.max(max[0],Math.abs((double)v-ref));}
   }catch(Exception e){throw new RuntimeException(e);}},()->false);
   if(different[0]!=0)throw new IllegalStateException("Native mismatch "+different[0]+" max="+max[0]);
   for(int r=0;r<repeats;r++){
    long t=System.nanoTime();c.range(1,last,(frame,pixels)->{},()->false);double seconds=(System.nanoTime()-t)/1e9;JsonObject row=new JsonObject();
    row.addProperty("repeat",r);row.addProperty("seconds",seconds);row.addProperty("fps",last/seconds);row.addProperty("core_fps",c.processedFrames/(c.coreNanos/1e9));
    row.addProperty("heap_used_bytes",ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed());
    if(gpu){row.addProperty("gpu_scratch_bytes",((FiARCStrictGPU)engine).scratchBytes());row.addProperty("gpu_history_bytes",((FiARCStrictGPU)engine).historyBytes());}
    runs.add(row);System.out.println(row);
   }
   JsonObject report=new JsonObject();report.addProperty("bit_exact",true);report.addProperty("backend",gpu?((FiARCStrictGPU)engine).deviceName:"Java CPU");report.addProperty("frames",last);report.addProperty("height",c.h);report.addProperty("width",c.w);report.addProperty("output_accumulated",false);report.add("runs",runs);
   if(gpu){JsonArray sizes=new JsonArray();for(FiARCOpenCLResident.Buffer b:((FiARCStrictGPU)engine).pool)sizes.add(b.bytes);report.add("scratch_buffer_bytes",sizes);}
   Files.createDirectories(out.getParent());Files.write(out,new GsonBuilder().setPrettyPrinting().create().toJson(report).getBytes("UTF-8"));
  }
 }
}
