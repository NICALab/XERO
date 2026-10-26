package org.fiarc.strictjava;
import ij.*;
import ij.gui.GenericDialog;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import java.nio.file.*;
import javax.swing.SwingUtilities;

/** One JAR, CPU or optional OpenCL GPU, all decoding inside ImageJ's process. */
public class FiARC_Standalone_Open implements PlugIn {
    public void run(String arg) {
        try {
            GenericDialog d=new GenericDialog("XERO — Open compressed image");
            d.addDirectoryField("Compressed archive folder", "",60);
            d.addNumericField("First frame (1-based)",1,0);d.addNumericField("Last frame (inclusive)",8,0);
            d.addChoice("Compute",new String[]{"Auto (GPU, otherwise CPU)","GPU / OpenCL","CPU"},"Auto (GPU, otherwise CPU)");d.addNumericField("OpenCL GPU index",0,0);
            d.addChoice("GPU workspace",new String[]{"Fast (full frame)","Low VRAM (512-pixel tiles)","Low VRAM (256-pixel tiles)"},"Fast (full frame)");
            d.addCheckbox("Virtual stack (bounded RAM)",true);
            d.addMessage("Open the folder containing manifest.json and model.npz.\nCurrent CLI/browser integer archives and legacy strict v2 are supported.\nVirtual mode retains at most 128 MiB of recent decoded frames.\nLow VRAM uses system RAM for features/history and is much slower.\nIt uses the same archive; no recompression is needed.\nNo browser or Python required. Escape cancels decoding.");
            d.showDialog();if(d.wasCanceled())return;
            Path path=Paths.get(d.getNextString());double a=d.getNextNumber(),b=d.getNextNumber();int first=(int)a,last=(int)b;int compute=d.getNextChoiceIndex();int workspace=d.getNextChoiceIndex();double g=d.getNextNumber();int index=(int)g;boolean virtual=d.getNextBoolean();
            if(!Double.isFinite(a)||!Double.isFinite(b)||!Double.isFinite(g)||a!=first||b!=last||g!=index||index<0||first<1||last<first)throw new IllegalArgumentException("Invalid integer range or GPU index");
            SwingUtilities.invokeLater(()->{
                final FiARCProgress progress=new FiARCProgress(first,last,compute==2?"CPU":"GPU / OpenCL");
                Thread task=new Thread(()->{
                FiARCCodec.Conv engine=null;FiARCCodec codec=null;boolean transferred=false;
                try {
                    if(compute==2)engine=new FiARCCodec.CPU();else try{engine=workspace==0?new FiARCStrictGPU(index):new FiARCTiledGPU(index,workspace==1?512:256);}catch(Throwable e){if(compute==1)throw e;IJ.log("XERO: GPU unavailable; using Java CPU. "+e.getMessage());engine=new FiARCCodec.CPU();}
                    codec=new FiARCCodec(path,engine);final FiARCCodec c=codec;boolean gpu=engine instanceof FiARCStrictGPU;
                    FiARCCodec.require(last<=codec.frames,"Last frame exceeds archive length ("+codec.frames+")");
                    codec.setProgress(progress);
                    long output=Math.multiplyExact(Math.multiplyExact((long)(last-first+1),codec.h),codec.w)*4;
                    if(virtual){
                        FiARCVirtualStack stack=new FiARCVirtualStack(codec,first,last,Math.min(128L<<20,Runtime.getRuntime().maxMemory()/8));
                        codec.setProgress(progress);stack.getProcessor(1);if(progress.cancelled.get())throw new java.util.concurrent.CancellationException();
                        codec.setProgress((done,total,frame,stage)->{IJ.showStatus("XERO · "+stage+" · frame "+frame);IJ.showProgress(done,total);});
                        ImagePlus image=new ImagePlus("XERO "+first+"–"+last+" (virtual, "+(gpu?"GPU":"CPU")+")",stack);stack.attach(image);transferred=true;SwingUtilities.invokeLater(()->image.show());
                    }else{
                    if(output>Runtime.getRuntime().maxMemory()/3)throw new IllegalArgumentException("Selected output exceeds one third of ImageJ heap. Enable Virtual stack or request a shorter range.");
                    ImageStack stack=new ImageStack(codec.w,codec.h);IJ.resetEscape();long start=System.nanoTime();
                    codec.range(first,last,(frame,pixels)->{stack.addSlice("Frame "+frame,new FloatProcessor(c.w,c.h,pixels));IJ.showProgress(frame-first+1,last-first+1);},()->progress.cancelled.get()||IJ.escapePressed());
                    if(progress.cancelled.get())throw new java.util.concurrent.CancellationException();
                    ImagePlus image=new ImagePlus("XERO "+first+"–"+last+(gpu?" (OpenCL)":" (CPU)"),stack);SwingUtilities.invokeLater(()->image.show());
                    IJ.log(String.format("XERO standalone: %d frames, %.3f s, %d packets, %d prefix frames; %s",stack.size(),(System.nanoTime()-start)/1e9,codec.packetsRead,codec.prefixFrames,gpu?((FiARCOpenCL)engine).deviceName:"CPU"));
                    }
                } catch(java.util.concurrent.CancellationException e){IJ.showStatus("XERO cancelled; partial output discarded");}
                catch(Throwable e){SwingUtilities.invokeLater(()->IJ.handleException(e));}
                finally{if(!transferred){if(codec!=null)codec.close();else if(engine!=null)engine.close();}progress.close();}
                },"XERO-decoder");task.setPriority(Thread.NORM_PRIORITY-1);progress.worker=task;task.start();
            });
        } catch(Throwable e){IJ.handleException(e);}
    }
}
