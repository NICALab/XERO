package org.fiarc.strictjava;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.concurrent.atomic.AtomicBoolean;

/** Non-modal progress, including restart prefix and integrity-check tail. */
public class FiARCProgress implements FiARCCodec.Progress {
    final AtomicBoolean cancelled=new AtomicBoolean();
    final JDialog dialog=new JDialog((Frame)null,"XERO decoding",false);
    final JProgressBar bar=new JProgressBar(0,1000);
    final JLabel status=new JLabel("Loading model…"),detail=new JLabel(" ");
    final JButton cancel=new JButton("Cancel");
    final long started=System.nanoTime();long lastUpdate=0;volatile Thread worker;
    public FiARCProgress(int first,int last,String mode){
        dialog.setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);
        JPanel panel=new JPanel();panel.setLayout(new BoxLayout(panel,BoxLayout.Y_AXIS));panel.setBorder(BorderFactory.createEmptyBorder(18,20,18,20));
        panel.add(new JLabel("Frames "+first+"–"+last+" · "+mode));panel.add(Box.createVerticalStrut(12));bar.setStringPainted(true);bar.setIndeterminate(true);panel.add(bar);panel.add(Box.createVerticalStrut(10));panel.add(status);panel.add(detail);panel.add(Box.createVerticalStrut(12));panel.add(cancel);
        cancel.addActionListener(e->cancel());dialog.addWindowListener(new WindowAdapter(){public void windowClosing(WindowEvent e){cancel();}});
        dialog.setContentPane(panel);dialog.setSize(520,205);dialog.setLocationRelativeTo(ij.IJ.getInstance());dialog.setVisible(true);
    }
    void cancel(){cancelled.set(true);cancel.setEnabled(false);status.setText("Cancelling; no partial image will be opened…");Thread t=worker;if(t!=null)t.interrupt();}
    public void update(int done,int total,int frame,String stage){
        long now=System.nanoTime();if(done<total&&now-lastUpdate<100_000_000L)return;lastUpdate=now;
        double seconds=(now-started)/1e9,eta=done>0?seconds*(total-done)/done:Double.NaN;
        SwingUtilities.invokeLater(()->{if(cancelled.get())return;bar.setIndeterminate(false);bar.setValue(total==0?0:(int)(1000L*done/total));bar.setString(String.format("%.1f%%",100.*done/Math.max(1,total)));status.setText(stage+" · frame "+frame);detail.setText(String.format("%d / %d processed · %.1f s elapsed%s",done,total,seconds,Double.isFinite(eta)?String.format(" · ~%.1f s remaining",eta):""));});
    }
    public void close(){SwingUtilities.invokeLater(()->dialog.dispose());}
}
