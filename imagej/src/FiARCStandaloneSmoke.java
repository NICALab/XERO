package org.fiarc.strictjava;
import ij.*;
import ij.process.FloatProcessor;
import java.nio.*;
import java.nio.file.*;
import java.io.*;
import java.security.MessageDigest;
import com.google.gson.*;

/** Real byte decoding and ImageJ stack construction, without a GUI or browser. */
public class FiARCStandaloneSmoke {
    public static void main(String[] args) throws Exception {
        Path folder=Paths.get(args[0]),out=Paths.get(args[4]);int first=Integer.parseInt(args[1]),last=Integer.parseInt(args[2]);String mode=args[3];
        FiARCCodec.Conv engine=mode.equals("cpu")?new FiARCCodec.CPU():new FiARCStrictGPU(Integer.parseInt(args[5]));
        try(FiARCCodec codec=new FiARCCodec(folder,engine)){
            ImageStack stack=new ImageStack(codec.w,codec.h);long t=System.nanoTime();
            codec.range(first,last,(frame,pixels)->stack.addSlice("Frame "+frame,new FloatProcessor(codec.w,codec.h,pixels)),()->false);
            ImagePlus image=new ImagePlus("XERO smoke",stack);Files.createDirectories(out);MessageDigest hash=MessageDigest.getInstance("SHA-256");
            try(OutputStream stream=Files.newOutputStream(out.resolve("imagej.f32"))){for(int i=1;i<=stack.size();i++){float[] a=(float[])stack.getPixels(i);byte[] bytes=new byte[a.length*4];ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(a);stream.write(bytes);hash.update(bytes);}}
            JsonObject receipt=new JsonObject();receipt.addProperty("mode",mode);receipt.addProperty("seconds",(System.nanoTime()-t)/1e9);receipt.addProperty("width",image.getWidth());receipt.addProperty("height",image.getHeight());receipt.addProperty("frames",image.getStackSize());receipt.addProperty("first",first);receipt.addProperty("last",last);receipt.addProperty("packets",codec.packetsRead);receipt.addProperty("prefix",codec.prefixFrames);
            StringBuilder hex=new StringBuilder();for(byte b:hash.digest())hex.append(String.format("%02x",b&255));receipt.addProperty("output_sha256",hex.toString());receipt.addProperty("browser_used",false);receipt.addProperty("python_used",false);if(engine instanceof FiARCOpenCL)receipt.addProperty("device",((FiARCOpenCL)engine).deviceName);
            String text=new GsonBuilder().setPrettyPrinting().create().toJson(receipt);Files.write(out.resolve("receipt.json"),text.getBytes("UTF-8"));System.out.println(text);
        }
    }
}
