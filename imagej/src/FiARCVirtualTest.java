package org.fiarc.strictjava;
import java.nio.*;
import java.nio.channels.*;
import java.nio.file.*;
import java.util.*;
public final class FiARCVirtualTest {
 public static void main(String[] a)throws Exception{
  FiARCCodec.Conv engine=a[2].equals("tiled")?new FiARCTiledGPU(0,32):a[2].equals("gpu")?new FiARCStrictGPU(0):new FiARCCodec.CPU();
  try(FiARCCodec c=new FiARCCodec(Paths.get(a[0]),engine);FileChannel f=FileChannel.open(Paths.get(a[1]))){
   FiARCVirtualStack s=new FiARCVirtualStack(c,1,c.frames,(long)c.w*c.h*4*3);
   int[] targets=a.length>3?new int[]{1,2,4,2,8}:new int[]{1,2,64,65,130,131,129,7,192,192};
   for(int t:targets){float[] v=(float[])s.getProcessor(t).getPixels();ByteBuffer b=ByteBuffer.allocate(v.length*4).order(ByteOrder.LITTLE_ENDIAN);long offset=(long)(t-1)*v.length*4;while(b.hasRemaining())if(f.read(b,offset+b.position())<0)throw new IllegalStateException("short reference");b.flip();for(float x:v)if(Float.floatToRawIntBits(x)!=b.getInt())throw new IllegalStateException("Mismatch at frame "+t);if(s.cache.size()>3)throw new IllegalStateException("Cache leak");System.out.println("PASS frame="+t+" cache_bytes="+s.cacheBytes);}
   // An interrupted cursor must be safe to restart rather than reuse partial entropy state.
   FiARCCodec.Cursor cursor=c.cursor();try{cursor.frame(20,(n,p)->{},()->true);throw new IllegalStateException("Cancellation ignored");}catch(java.util.concurrent.CancellationException expected){}
   cursor.frame(1,(n,p)->{},()->false);s.close();if(s.cacheBytes!=0)throw new IllegalStateException("Close leak");System.out.println("PASS seek/cache/cancellation/close");
  }
 }
}
