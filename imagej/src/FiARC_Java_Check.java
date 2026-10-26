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

/** A small native-reference fixture, no downloads or user image required. */
public final class FiARC_Java_Check implements PlugIn {
    static byte[] resource(String file)throws Exception {return resource("check",file);}
    static byte[] resource(String group,String file)throws Exception {try(InputStream in=FiARC_Java_Check.class.getResourceAsStream("/fiarc/"+group+"/"+file)){if(in==null)throw new IOException("Missing self-test fixture");ByteArrayOutputStream out=new ByteArrayOutputStream();byte[] b=new byte[8192];int n;while((n=in.read(b))!=-1)out.write(b,0,n);return out.toByteArray();}}
    static FiARCCodec.Conv engine(boolean gpu,int index,int tile){return !gpu?new FiARCCodec.CPU():tile>0?new FiARCTiledGPU(index,tile):new FiARCStrictGPU(index);}
    static String currentCheck(boolean gpu,int index,FiARCProgress progress,int tile)throws Exception {
        Path root=Files.createTempDirectory("fiarc-cli-check-");List<Path> files=new ArrayList<>();
        try{
            byte[] manifest=resource("check_cli","manifest.json");String packet=FiARCCodec.json(manifest).getAsJsonArray("groups").get(0).getAsJsonObject().get("file").getAsString();
            for(String n:new String[]{"manifest.json","model.npz",packet}){Path p=root.resolve(n);Files.write(p,resource("check_cli",n));files.add(p);}
            MessageDigest digest=MessageDigest.getInstance("SHA-256");
            try(FiARCCodec c=new FiARCCodec(root,engine(gpu,index,tile))){
                c.setProgress(progress);c.cursor().frame(4,(frame,pixels)->{byte[] b=new byte[pixels.length*4];ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(pixels);digest.update(b);},()->progress!=null&&progress.cancelled.get());
            }
            StringBuilder h=new StringBuilder();for(byte b:digest.digest())h.append(String.format("%02x",b&255));String want=new String(resource("check_cli","reference.sha256"),"UTF-8").trim();if(!h.toString().equals(want))throw new IllegalStateException("Current CLI odd-XY fixture mismatch: "+h);
            return "\nPASS — current CLI 49×61 / 1% synthetic native-reference check\nCLI SHA-256: "+h;
        }finally{for(Path p:files)Files.deleteIfExists(p);Files.deleteIfExists(root);}
    }
    static String check(boolean gpu,int index,FiARCProgress progress)throws Exception {
        return check(gpu,index,progress,0);
    }
    static String check(boolean gpu,int index,FiARCProgress progress,int tile)throws Exception {
        Path root=Files.createTempDirectory("fiarc-java-check-");List<Path> files=new ArrayList<>();FiARCCodec.Conv engine=null;
        try {
            for(String n:new String[]{"manifest.json","model.npz","correction.bin","group_0000.bin","group_0001.bin","group_0002.bin"}){Path f=root.resolve(n);Files.write(f,resource(n));files.add(f);}
            engine=engine(gpu,index,tile);MessageDigest digest=MessageDigest.getInstance("SHA-256");long start=System.nanoTime();
            try(FiARCCodec c=new FiARCCodec(root,engine)){
                c.setProgress(progress);c.range(130,131,(frame,data)->{byte[] b=new byte[data.length*4];ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(data);digest.update(b);},()->progress!=null&&progress.cancelled.get());
            }
            StringBuilder hash=new StringBuilder();for(byte b:digest.digest())hash.append(String.format("%02x",b&255));String want=new String(resource("reference.sha256"),"UTF-8").trim();
            if(!hash.toString().equals(want))throw new IllegalStateException("FAIL: decoded output differs from native strict_v2 reference: "+hash);
            return "PASS — bit-exact strict v2 restart/seek test\nOS: "+System.getProperty("os.name")+" / "+System.getProperty("os.arch")+"\nJava: "+System.getProperty("java.version")+"\nBackend: "+(gpu?((FiARCStrictGPU)engine).deviceName:"Java CPU")+"\nTile size: "+tile+"\nSHA-256: "+hash+currentCheck(gpu,index,progress,tile)+"\nElapsed (not an FPS benchmark): "+String.format("%.2f s",(System.nanoTime()-start)/1e9);
        } finally {if(engine!=null)engine.close();for(Path f:files)Files.deleteIfExists(f);Files.deleteIfExists(root);}
    }
    public void run(String arg){GenericDialog d=new GenericDialog("XERO — Java compatibility check");d.addChoice("Backend",new String[]{"CPU","GPU / OpenCL","GPU / Low VRAM"},"CPU");d.addNumericField("OpenCL GPU index",0,0);d.addMessage("Uses bundled tiny archives; no image download or upload.\nTests legacy strict v2 and current CLI odd-sized images\nagainst native decoder reference pixels.");d.showDialog();if(d.wasCanceled())return;int backend=d.getNextChoiceIndex();boolean gpu=backend!=0;int tile=backend==2?32:0;double value=d.getNextNumber();int index=(int)value;if(value!=index||index<0){IJ.error("Invalid GPU index");return;}
        SwingUtilities.invokeLater(()->{FiARCProgress progress=new FiARCProgress(130,131,gpu?"GPU compatibility":"CPU compatibility");Thread t=new Thread(()->{try{String result=check(gpu,index,progress,tile);IJ.log(result);SwingUtilities.invokeLater(()->IJ.showMessage("XERO compatibility",result));}catch(Throwable e){SwingUtilities.invokeLater(()->IJ.handleException(e));}finally{progress.close();}},"XERO-check");progress.worker=t;t.start();});
    }
    public static void main(String[] a)throws Exception {System.out.println(check(a.length>0&&a[0].equals("gpu"),a.length>1?Integer.parseInt(a[1]):0,null,a.length>2?Integer.parseInt(a[2]):0));}
}
