package org.fiarc.strictjava;
import java.nio.*;
import java.nio.channels.*;
import java.nio.file.*;
import java.lang.management.*;
import com.google.gson.*;

/** Prefix/packet comparison against unchanged native reference, bounded sink. */
public final class FiARCTileBenchmark {
 public static void main(String[] a)throws Exception{
  Path archive=Paths.get(a[0]),ref=Paths.get(a[1]),out=Paths.get(a[2]);int tile=Integer.parseInt(a[3]),last=Integer.parseInt(a[4]),repeats=Integer.parseInt(a[5]);
  FiARCStrictGPU engine=tile==0?new FiARCStrictGPU(0):new FiARCTiledGPU(0,tile);JsonObject result=new JsonObject();JsonArray rows=new JsonArray();
  try(FiARCCodec c=new FiARCCodec(archive,engine);FileChannel f=FileChannel.open(ref)){
   c.preloadPackets();ByteBuffer buffer=ByteBuffer.allocate(c.h*c.w*4).order(ByteOrder.LITTLE_ENDIAN);long started=System.nanoTime();
   c.cursor().frame(last,(frame,pixels)->{try{buffer.clear();long pos=(long)(frame-1)*pixels.length*4;while(buffer.hasRemaining())if(f.read(buffer,pos+buffer.position())<0)throw new IllegalStateException("Short reference");buffer.flip();long wrong=0;double max=0;for(float v:pixels){float x=buffer.getFloat();if(Float.floatToRawIntBits(v)!=Float.floatToRawIntBits(x))wrong++;max=Math.max(max,Math.abs((double)v-x));}if(wrong!=0)throw new IllegalStateException("Mismatch frame="+frame+" pixels="+wrong+" max="+max);if(frame==1||frame%8==0)System.out.println("exact frame="+frame+" elapsed="+(System.nanoTime()-started)/1e9);}catch(Exception e){throw new RuntimeException(e);}},()->false);
   result.addProperty("bit_exact",true);result.addProperty("tile",tile);result.addProperty("width",c.w);result.addProperty("height",c.h);result.addProperty("frames",last);result.addProperty("archive_changed",false);result.addProperty("verification_seconds",(System.nanoTime()-started)/1e9);
   result.addProperty("verification_gpu_scratch_bytes",engine.scratchBytes());result.addProperty("verification_gpu_history_bytes",engine.historyBytes());
   if(tile>0)result.addProperty("verification_host_history_bytes",((FiARCTiledGPU)engine).hostHistoryBytes());
   for(int r=0;r<repeats;r++){System.gc();long t=System.nanoTime();c.cursor().frame(last,(frame,pixels)->{},()->false);double sec=(System.nanoTime()-t)/1e9;JsonObject row=new JsonObject();row.addProperty("seconds",sec);row.addProperty("fps",last/sec);row.addProperty("gpu_scratch_bytes",engine.scratchBytes());row.addProperty("gpu_history_bytes",engine.historyBytes());row.addProperty("java_heap_used_bytes",ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed());if(tile>0){row.addProperty("host_history_bytes",((FiARCTiledGPU)engine).hostHistoryBytes());row.addProperty("tiles_processed",((FiARCTiledGPU)engine).tilesDecoded);row.addProperty("global_max_seconds",((FiARCTiledGPU)engine).maxPassNanos/1e9);}rows.add(row);System.out.println(row);}
   result.add("runs",rows);Files.createDirectories(out.getParent());Files.write(out,new GsonBuilder().setPrettyPrinting().create().toJson(result).getBytes("UTF-8"));
  }
 }
}
