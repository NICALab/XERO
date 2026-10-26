package org.fiarc.strictjava;

import ij.ImageJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.GenericDialog;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.nio.*;
import java.nio.file.*;
import com.google.gson.JsonParser;
import com.google.gson.JsonArray;

/** Run under a real display (or Xvfb), including Java 8. Exercises the menu
 * opener, not only the codec self-check. Arguments: archive, reference.f32,
 * cpu|gpu. Compile with --release 8 against the shipped JAR.
 */
public final class ImageJOpenRegression {
    public static void main(String[] args) throws Exception {
        Thread.setDefaultUncaughtExceptionHandler((t,e)->{e.printStackTrace();System.exit(1);});
        if (FiARC_Standalone_Open.decodedBytes(160,64,64)!=2621440L)
            throw new AssertionError("Output byte count");
        if (FiARC_Standalone_Open.decodedBytes(4096,2048,2048)!=68719476736L)
            throw new AssertionError("Long output byte count");
        try {
            FiARC_Standalone_Open.decodedBytes(Integer.MAX_VALUE,Integer.MAX_VALUE,Integer.MAX_VALUE);
            throw new AssertionError("Overflow not rejected");
        } catch (ArithmeticException expected) { }
        final boolean gpu=args[2].equals("gpu");
        JsonArray shape=JsonParser.parseString(new String(Files.readAllBytes(Paths.get(args[0],"manifest.json")),java.nio.charset.StandardCharsets.UTF_8)).getAsJsonObject().getAsJsonArray("shape");
        final int frames=shape.get(0).getAsInt(),height=shape.get(1).getAsInt(),width=shape.get(2).getAsInt(),area=height*width;
        ByteBuffer reference=ByteBuffer.wrap(Files.readAllBytes(Paths.get(args[1]))).order(ByteOrder.LITTLE_ENDIAN);
        if(reference.capacity()!=FiARC_Standalone_Open.decodedBytes(frames,height,width))throw new AssertionError("Reference length");
        new ImageJ();
        new Thread(()->new FiARC_Standalone_Open().run(""),"Open-test").start();
        GenericDialog dialog=null;
        long deadline=System.currentTimeMillis()+15000;
        while(dialog==null && System.currentTimeMillis()<deadline) {
            for(Window w:Window.getWindows())
                if(w instanceof GenericDialog && w.isShowing() && ((Dialog)w).getTitle().equals("XERO — Open compressed image"))
                    dialog=(GenericDialog)w;
            Thread.sleep(30);
        }
        if(dialog==null)throw new AssertionError("Open dialog absent");
        final GenericDialog d=dialog;
        EventQueue.invokeAndWait(()->{
            ((TextField)d.getStringFields().get(0)).setText(args[0]);
            ((TextField)d.getNumericFields().get(0)).setText("1");
            ((TextField)d.getNumericFields().get(1)).setText(String.valueOf(frames));
            ((Choice)d.getChoices().get(0)).select(gpu?1:2);
            if(d.getCheckboxes()!=null && !d.getCheckboxes().isEmpty())throw new AssertionError("Unexpected checkbox");
            Button ok=d.getButtons()[0];
            ok.dispatchEvent(new ActionEvent(ok,ActionEvent.ACTION_PERFORMED,"OK"));
        });
        deadline=System.currentTimeMillis()+120000;
        while(WindowManager.getCurrentImage()==null && System.currentTimeMillis()<deadline)Thread.sleep(30);
        final ImagePlus im=WindowManager.getCurrentImage();
        if(im==null)throw new AssertionError("No image opened; inspect ImageJ exception/log windows");
        if(im.getWidth()!=width || im.getHeight()!=height || im.getStackSize()!=frames || im.getBitDepth()!=32 || im.getStack().isVirtual())
            throw new AssertionError("Unexpected image dimensions/type");
        for(int frame:new int[]{1,Math.min(2,frames),Math.min(64,frames),Math.min(128,frames),Math.min(129,frames),frames,1}) {
            final int f=frame;
            EventQueue.invokeAndWait(()->im.setSlice(f));
            float[] pixels=(float[])im.getProcessor().getPixels();
            for(int i=0;i<pixels.length;i++)
                if(Float.floatToRawIntBits(pixels[i])!=reference.getInt(((frame-1)*area+i)*4))
                    throw new AssertionError("Pixel mismatch at "+frame+":"+i);
        }
        // A resident stack is already decoded; verify every stored pixel too.
        for(int frame=1;frame<=frames;frame++) {
            float[] pixels=(float[])im.getStack().getPixels(frame);
            for(int i=0;i<pixels.length;i++)
                if(Float.floatToRawIntBits(pixels[i])!=reference.getInt(((frame-1)*area+i)*4))
                    throw new AssertionError("Resident pixel mismatch at "+frame+":"+i);
        }
        EventQueue.invokeAndWait(()->{im.changes=false;im.close();});
        System.out.println("PASS actual menu open, "+args[2]+", "+frames+" resident frames; all pixels bit-exact; Java "+System.getProperty("java.version"));
        System.exit(0);
    }
}
