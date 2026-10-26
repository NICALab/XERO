package org.fiarc.strictjava;
import ij.*;
import ij.process.*;
import java.util.*;

/** Bounded decoded-frame LRU; no uncompressed temporary files. */
public final class FiARCVirtualStack extends VirtualStack implements AutoCloseable {
    final FiARCCodec codec;final FiARCCodec.Cursor cursor;final int first,last,capacity;
    final LinkedHashMap<Integer,float[]> cache=new LinkedHashMap<>(16,.75f,true);
    boolean closed=false;long cacheBytes;
    public FiARCVirtualStack(FiARCCodec c,int first,int last,long budget){
        super(c.w,c.h,null,null);FiARCCodec.require(first>=1&&last>=first&&last<=c.frames,"Invalid virtual range");
        codec=c;cursor=c.cursor();this.first=first;this.last=last;long size=(long)c.h*c.w*4;capacity=(int)Math.max(1,Math.min(128,budget/size));
        codec.setProgress((done,total,frame,stage)->{IJ.showStatus("XERO · "+stage+" · frame "+frame);IJ.showProgress(done,total);});
    }
    @Override public int getSize(){return last-first+1;}
    @Override public String getSliceLabel(int n){return "Frame "+(first+n-1);}
    @Override public int getBitDepth(){return 32;}
    @Override public Object getPixels(int n){return getProcessor(n).getPixels();}
    @Override public synchronized ImageProcessor getProcessor(int n){
        if(closed)throw new IllegalStateException("XERO image is closed");if(n<1||n>getSize())throw new IllegalArgumentException("Slice outside stack");int target=first+n-1;
        try{float[] p=cache.get(target);if(p==null){IJ.resetEscape();p=cursor.frame(target,(frame,pixels)->{if(frame<first||frame>last)return;cache.put(frame,pixels);while(cache.size()>capacity){Iterator<Integer> it=cache.keySet().iterator();it.next();it.remove();}cacheBytes=(long)cache.size()*codec.w*codec.h*4;},()->IJ.escapePressed());}
            return new FloatProcessor(codec.w,codec.h,p.clone());
        }catch(java.util.concurrent.CancellationException e){throw e;}catch(Exception e){throw new IllegalStateException("XERO decode failed at frame "+target,e);}
    }
    public void attach(ImagePlus image){ImagePlus.addImageListener(new ImageListener(){public void imageOpened(ImagePlus x){}public void imageUpdated(ImagePlus x){}public void imageClosed(ImagePlus x){if(x==image){close();ImagePlus.removeImageListener(this);}}});}
    public synchronized boolean released(){return closed&&cacheBytes==0;}
    @Override public synchronized void close(){if(!closed){closed=true;cache.clear();cacheBytes=0;codec.close();}}
}
