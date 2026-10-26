package org.fiarc.strictjava;
import ij.*;
import ij.gui.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.imageio.ImageIO;

/** Actual ImageJ GUI, not a mock: cancel then open a verified float stack. */
public final class FiARCJavaGuiSmoke {
    static Window waitFor(String title,long ms)throws Exception {long end=System.currentTimeMillis()+ms;while(System.currentTimeMillis()<end){for(Window w:Window.getWindows())if(w.isShowing()){String t=w instanceof Dialog?((Dialog)w).getTitle():w instanceof Frame?((Frame)w).getTitle():"";if(t.equals(title))return w;}Thread.sleep(50);}throw new Exception("Missing "+title);}
    static Component button(Container c,String label){for(Component v:c.getComponents()){if(v instanceof Button&&((Button)v).getLabel().trim().equals(label)||v instanceof JButton&&((JButton)v).getText().trim().equals(label))return v;if(v instanceof Container){Component b=button((Container)v,label);if(b!=null)return b;}}return null;}
    static void shot(Window w,File f)throws Exception {EventQueue.invokeAndWait(()->{w.setLocation(60,60);w.toFront();});Thread.sleep(150);ImageIO.write(new Robot().createScreenCapture(w.getBounds()),"png",f);}
    static void start(String archive,int first,int last,boolean virtual)throws Exception {new Thread(()->new FiARC_Standalone_Open().run("")).start();GenericDialog d=(GenericDialog)waitFor("XERO — Open compressed image",15000);EventQueue.invokeAndWait(()->{((TextField)d.getStringFields().get(0)).setText(archive);((TextField)d.getNumericFields().get(0)).setText(""+first);((TextField)d.getNumericFields().get(1)).setText(""+last);((Choice)d.getChoices().get(0)).select(virtual?1:2);if(virtual&&Boolean.getBoolean("fiarc.gui.test.tiled"))((Choice)d.getChoices().get(1)).select(1);((Checkbox)d.getCheckboxes().get(0)).setState(virtual);Button b=(Button)button(d,"OK");b.dispatchEvent(new ActionEvent(b,ActionEvent.ACTION_PERFORMED,"OK"));});}
    public static void main(String[] a)throws Exception {
        Thread.setDefaultUncaughtExceptionHandler((t,e)->{e.printStackTrace();System.exit(1);});new ImageJ();File out=new File(a[1]);out.mkdirs();
        start(a[0],1,192,false);Window progress=waitFor("XERO decoding",15000);Thread.sleep(1000);shot(progress,new File(out,"progress.png"));
        EventQueue.invokeAndWait(()->((JButton)button((Container)progress,"Cancel")).doClick());long until=System.currentTimeMillis()+15000;while(progress.isShowing()&&System.currentTimeMillis()<until)Thread.sleep(50);if(progress.isShowing()||WindowManager.getImageCount()!=0)throw new Exception("Cancel did not discard output");
        start(a[0],1,192,true);until=System.currentTimeMillis()+20000;while(WindowManager.getCurrentImage()==null&&System.currentTimeMillis()<until)Thread.sleep(50);ImagePlus im=WindowManager.getCurrentImage();if(im==null)throw new Exception("No virtual image opened");Window image=im.getWindow();if(im.getStackSize()!=192||im.getBitDepth()!=32||!im.getStack().isVirtual())throw new Exception("Wrong virtual stack");
        EventQueue.invokeAndWait(()->{im.setSlice(65);im.resetDisplayRange();});shot(image,new File(out,"decoded.png"));FiARCVirtualStack stack=(FiARCVirtualStack)im.getStack();EventQueue.invokeAndWait(()->im.close());until=System.currentTimeMillis()+5000;while(!stack.released()&&System.currentTimeMillis()<until)Thread.sleep(50);if(!stack.released())throw new Exception("Virtual close did not release resources");System.out.println("PASS: real progress, Cancel, no partial image, virtual stack seek and close");System.exit(0);
    }
}
