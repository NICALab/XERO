package org.fiarc.strictjava;
import org.jocl.*;
import static org.jocl.CL.*;
import java.util.*;

/** Optional OpenCL integer convolution. No float GPU math or CUDA dependency. */
public class FiARCOpenCL implements FiARCCodec.Conv {
    cl_context context;cl_command_queue queue;cl_program program;cl_kernel kernel;
    final Map<FiARCCodec.Layer,cl_mem> weights=new IdentityHashMap<>();
    public String deviceName;
    cl_device_id device;
    static final String SOURCE=
        "__kernel void dot(__global const int *x,__global const int *weights,__global int *out,int ci,int co,int h,int w,int kh,int kw){"
        +"int i=get_global_id(0);if(i>=co*h*w)return;int plane=h*w,c=i/plane,y=(i%plane)/w,z=i%w;int sum=0;"
        +"for(int ch=0;ch<ci;ch++)for(int ky=0;ky<kh;ky++){int yy=y+ky-kh/2;if(yy<0||yy>=h)continue;"
        +"for(int kx=0;kx<kw;kx++){int xx=z+kx-kw/2;if(xx>=0&&xx<w)sum+=x[ch*plane+yy*w+xx]*weights[((c*ci+ch)*kh+ky)*kw+kx];}}out[i]=sum;}";
    static void ok(int code){if(code!=CL_SUCCESS)throw new IllegalStateException("OpenCL error "+code);}
    public FiARCOpenCL(int index) {
        try {
            int[] count={0};ok(clGetPlatformIDs(0,null,count));cl_platform_id[] platforms=new cl_platform_id[count[0]];ok(clGetPlatformIDs(platforms.length,platforms,null));
            List<cl_device_id> devices=new ArrayList<>();List<cl_platform_id> owners=new ArrayList<>();
            for(cl_platform_id p:platforms){int code=clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,0,null,count);if(code==CL_DEVICE_NOT_FOUND)continue;ok(code);cl_device_id[] d=new cl_device_id[count[0]];ok(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,d.length,d,null));for(cl_device_id v:d){devices.add(v);owners.add(p);}}
            if(index<0||index>=devices.size())throw new IllegalArgumentException("No OpenCL GPU at index "+index+"; use CPU or install the GPU vendor driver");
            cl_device_id dev=devices.get(index);device=dev;long[] bytes={0};ok(clGetDeviceInfo(dev,CL_DEVICE_NAME,0,null,bytes));byte[] text=new byte[(int)bytes[0]];ok(clGetDeviceInfo(dev,CL_DEVICE_NAME,text.length,Pointer.to(text),null));deviceName=new String(text).trim();
            int[] error={0};cl_context_properties props=new cl_context_properties();props.addProperty(CL_CONTEXT_PLATFORM,owners.get(index));context=clCreateContext(props,1,new cl_device_id[]{dev},null,null,error);ok(error[0]);
            queue=clCreateCommandQueue(context,dev,0,error);ok(error[0]);program=clCreateProgramWithSource(context,1,new String[]{SOURCE},null,error);ok(error[0]);
            ok(clBuildProgram(program,0,null,"-cl-std=CL1.2",null,null));kernel=clCreateKernel(program,"dot",error);ok(error[0]);
        } catch(Throwable e){close();throw new IllegalStateException("OpenCL GPU initialization failed; CPU remains available",e);}
    }
    public int[] dot(int[] x,FiARCCodec.Layer l,int h,int w){
        int[] error={0},out=new int[l.co*h*w];cl_mem input=null,output=null;
        try {
            cl_mem weight=weights.get(l);if(weight==null){weight=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(long)l.weights.length*4,Pointer.to(l.weights),error);ok(error[0]);weights.put(l,weight);}
            input=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(long)x.length*4,Pointer.to(x),error);ok(error[0]);output=clCreateBuffer(context,CL_MEM_WRITE_ONLY,(long)out.length*4,null,error);ok(error[0]);
            ok(clSetKernelArg(kernel,0,Sizeof.cl_mem,Pointer.to(input)));ok(clSetKernelArg(kernel,1,Sizeof.cl_mem,Pointer.to(weight)));ok(clSetKernelArg(kernel,2,Sizeof.cl_mem,Pointer.to(output)));
            int[] params={l.ci,l.co,h,w,l.kh,l.kw};for(int k=0;k<params.length;k++)ok(clSetKernelArg(kernel,k+3,Sizeof.cl_int,Pointer.to(new int[]{params[k]})));
            ok(clEnqueueNDRangeKernel(queue,kernel,1,null,new long[]{out.length},null,0,null,null));ok(clEnqueueReadBuffer(queue,output,CL_TRUE,0,(long)out.length*4,Pointer.to(out),0,null,null));return out;
        } finally{if(output!=null)clReleaseMemObject(output);if(input!=null)clReleaseMemObject(input);}
    }
    public void close(){for(cl_mem m:weights.values())clReleaseMemObject(m);weights.clear();if(kernel!=null){clReleaseKernel(kernel);kernel=null;}if(program!=null){clReleaseProgram(program);program=null;}if(queue!=null){clReleaseCommandQueue(queue);queue=null;}if(context!=null){clReleaseContext(context);context=null;}}
}
