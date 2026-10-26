package org.fiarc.strictjava;
import ij.*;
import ij.gui.GenericDialog;
import ij.plugin.PlugIn;
import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.security.*;
import java.util.*;
import javax.swing.SwingUtilities;

/** Check a small XERO archive against native-reference pixels. */
public final class FiARC_Java_Check implements PlugIn {
    static byte[] resource(String file)throws Exception {
        try(InputStream in=FiARC_Java_Check.class.getResourceAsStream("/fiarc/check_cli/"+file)){
            if(in==null)throw new IOException("Missing self-test fixture");
            ByteArrayOutputStream out=new ByteArrayOutputStream();byte[] b=new byte[8192];int n;
            while((n=in.read(b))!=-1)out.write(b,0,n);return out.toByteArray();
        }
    }
    static FiARCCodec.Conv engine(boolean gpu,int index,int tile){return !gpu?new FiARCCodec.CPU():tile>0?new FiARCTiledGPU(index,tile):new FiARCStrictGPU(index);}
    static String check(boolean gpu,int index,FiARCProgress progress)throws Exception {return check(gpu,index,progress,0);}
    static String check(boolean gpu,int index,FiARCProgress progress,int tile)throws Exception {
        Path root=Files.createTempDirectory("xero-check-");List<Path> files=new ArrayList<>();long start=System.nanoTime();
        try{
            String packet=FiARCCodec.json(resource("manifest.json")).getAsJsonArray("groups").get(0).getAsJsonObject().get("file").getAsString();
            for(String n:new String[]{"manifest.json","model.npz",packet}){Path p=root.resolve(n);Files.write(p,resource(n));files.add(p);}
            MessageDigest digest=MessageDigest.getInstance("SHA-256");String backend;
            FiARCCodec.Conv compute=engine(gpu,index,tile);
            // The codec owns the engine after construction succeeds.
            FiARCCodec c;
            try{c=new FiARCCodec(root,compute);}catch(Throwable error){compute.close();throw error;}
            try(FiARCCodec owned=c){
                backend=gpu?((FiARCStrictGPU)compute).deviceName:"Java CPU";
                owned.setProgress(progress);
                owned.cursor().frame(4,(frame,pixels)->{byte[] b=new byte[pixels.length*4];ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(pixels);digest.update(b);},()->progress!=null&&progress.cancelled.get());
            }
            StringBuilder h=new StringBuilder();for(byte b:digest.digest())h.append(String.format("%02x",b&255));
            String want=new String(resource("reference.sha256"),"UTF-8").trim();
            if(!h.toString().equals(want))throw new IllegalStateException("Decoded pixels differ from the native reference: "+h);
            return "PASS — XERO 49×61 / 1% synthetic reference check\nOS: "+System.getProperty("os.name")+" / "+System.getProperty("os.arch")+"\nJava: "+System.getProperty("java.version")+"\nBackend: "+backend+"\nTile size: "+tile+"\nSHA-256: "+h+"\nElapsed (not an FPS benchmark): "+String.format("%.2f s",(System.nanoTime()-start)/1e9);
        }finally{for(Path p:files)Files.deleteIfExists(p);Files.deleteIfExists(root);}
    }
    public void run(String arg){
        GenericDialog d=new GenericDialog("XERO — Compatibility check");
        d.addChoice("Backend",new String[]{"CPU","GPU / OpenCL","GPU / Low VRAM"},"CPU");
        d.addNumericField("OpenCL GPU index",0,0);
        d.addMessage("Decodes a bundled example and checks its pixels.\nNo image download or upload.");
        d.showDialog();if(d.wasCanceled())return;
        int backend=d.getNextChoiceIndex();boolean gpu=backend!=0;int tile=backend==2?32:0;
        double value=d.getNextNumber();int index=(int)value;
        if(value!=index||index<0){IJ.error("Invalid GPU index");return;}
        SwingUtilities.invokeLater(()->{
            FiARCProgress progress=new FiARCProgress(1,4,gpu?"GPU compatibility":"CPU compatibility");
            Thread t=new Thread(()->{try{String result=check(gpu,index,progress,tile);IJ.log(result);SwingUtilities.invokeLater(()->IJ.showMessage("XERO compatibility",result));}catch(Throwable e){SwingUtilities.invokeLater(()->IJ.handleException(e));}finally{progress.close();}},"XERO-check");
            progress.worker=t;t.start();
        });
    }
    public static void main(String[] a)throws Exception {System.out.println(check(a.length>0&&a[0].equals("gpu"),a.length>1?Integer.parseInt(a[1]):0,null,a.length>2?Integer.parseInt(a[2]):0));}
}
