package org.fiarc.strictjava;

import com.google.gson.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.zip.InflaterInputStream;

/** Container adapter only: numerical/entropy decoding remains in FiARCCodec. */
public abstract class XeroSource implements AutoCloseable {
    final JsonObject manifest;
    XeroSource(JsonObject manifest){this.manifest=manifest;}
    abstract byte[] model() throws Exception;
    abstract byte[] packet(JsonObject group) throws Exception;
    public void close(){}
    static byte[] verified(byte[] data,String sha)throws Exception {
        FiARCCodec.require(FiARCCodec.hash(data).equals(sha),"XERO SHA-256 mismatch");return data;
    }
    public static XeroSource open(Path path)throws Exception {
        Path root=path.toRealPath();
        if(Files.exists(root.resolve("zarr.json")))return new Ome(root);
        return new Native(root);
    }
    static final class Native extends XeroSource {
        final Path root;
        Native(Path root)throws Exception {super(FiARCCodec.json(Files.readAllBytes(root.resolve("manifest.json"))));this.root=root;
            FiARCCodec.require(!Files.exists(root.resolve("INCOMPLETE")),"Incomplete XERO archive");}
        byte[] model()throws Exception{return FiARCCodec.checked(root,manifest.get("model").getAsString(),manifest.get("model_sha256").getAsString());}
        byte[] packet(JsonObject g)throws Exception{return FiARCCodec.checked(root,g.get("file").getAsString(),g.get("sha256").getAsString());}
    }
    static byte[] unpackModel(JsonObject config)throws Exception {
        String encoded=config.get("model_zlib_base64").getAsString();
        FiARCCodec.require(encoded.length()<=32*1024*1024,"Oversized model metadata");
        ByteArrayOutputStream out=new ByteArrayOutputStream();
        try(InputStream in=new InflaterInputStream(new ByteArrayInputStream(Base64.getDecoder().decode(encoded)))) {
            byte[] b=new byte[65536];int n;while((n=in.read(b))!=-1){FiARCCodec.require(out.size()+n<=16*1024*1024,"Oversized model");out.write(b,0,n);}
        }
        return verified(out.toByteArray(),config.get("model_sha256").getAsString());
    }
    static JsonObject omeManifest(Path root)throws Exception {
        JsonObject metadata=FiARCCodec.json(Files.readAllBytes(root.resolve("zarr.json")));
        FiARCCodec.require(metadata.get("zarr_format").getAsInt()==3,"Expected Zarr v3");
        JsonObject x=metadata.getAsJsonObject("attributes").getAsJsonObject("xero");
        FiARCCodec.require(x.get("format").getAsString().equals("xero.ome_zarr.v1")&&x.get("complete").getAsBoolean(),"Unsupported/incomplete XERO OME-Zarr");
        return x.getAsJsonObject("archive");
    }
    static final class Ome extends XeroSource {
        final Path root;final JsonObject config;
        Ome(Path root)throws Exception {
            super(omeManifest(root));this.root=root;
            JsonObject a=FiARCCodec.json(Files.readAllBytes(root.resolve("0/zarr.json")));
            JsonArray codecs=a.getAsJsonArray("codecs");
            FiARCCodec.require(codecs.size()==1&&codecs.get(0).getAsJsonObject().get("name").getAsString().equals("xero.restart_v1"),"Unsupported Zarr codecs");
            config=codecs.get(0).getAsJsonObject().getAsJsonObject("configuration");
            FiARCCodec.require(a.getAsJsonArray("shape").equals(manifest.getAsJsonArray("shape")),"Zarr shape/index mismatch");
            for(String key:new String[]{"profile","restart","tolerance","model_sha256"})FiARCCodec.require(config.get(key).equals(manifest.get(key)),"Zarr configuration mismatch: "+key);
        }
        byte[] model()throws Exception{return unpackModel(config);}
        byte[] packet(JsonObject g)throws Exception {
            String key=g.get("xero_chunk").getAsString();
            FiARCCodec.require(key.matches("0/c/[0-9]+/0/0"),"Invalid chunk key");
            Path path=root.resolve(key).toRealPath();FiARCCodec.require(path.startsWith(root),"Outside OME-Zarr");
            long offset=g.get("xero_offset").getAsLong(),n=g.get("bytes").getAsLong();
            FiARCCodec.require(offset>=44&&n>0&&n<=FiARCCodec.MAX_BLOB,"Invalid packet extent");
            byte[] bytes=new byte[(int)n];try(RandomAccessFile f=new RandomAccessFile(path.toFile(),"r")){
                FiARCCodec.require(offset<=f.length()-n,"Truncated XERO chunk");f.seek(offset);f.readFully(bytes);
            }
            return verified(bytes,g.get("sha256").getAsString());
        }
    }
    /** One independent restart, replaced between requests while weights stay resident. */
    static final class Memory extends XeroSource {
        final byte[] weights;byte[] chunk;final Map<String,int[]> extents=new HashMap<>();
        Memory(JsonObject manifest,byte[] weights){super(manifest);this.weights=weights;}
        byte[] model()throws Exception{return verified(weights,manifest.get("model_sha256").getAsString());}
        void set(byte[] chunk,JsonObject header,int offset)throws Exception {
            this.chunk=chunk;extents.clear();JsonArray groups=new JsonArray();int end=0,i=0;
            for(JsonElement e:header.getAsJsonArray("groups")){
                JsonObject g=e.getAsJsonObject().deepCopy();int count=g.get("frames").getAsInt(),n=g.get("bytes").getAsInt();
                FiARCCodec.require(g.get("start").getAsInt()==end&&count>0&&count<=64&&n>0&&n<=chunk.length-offset,"Invalid chunk packet table");
                String name=String.format("group_%06d.bin",i++);g.addProperty("file",name);groups.add(g);extents.put(name,new int[]{offset,n});offset+=n;end+=count;
            }
            FiARCCodec.require(end==header.get("frames").getAsInt()&&offset==chunk.length,"Incomplete restart chunk");manifest.add("groups",groups);
        }
        byte[] packet(JsonObject g)throws Exception {int[] p=extents.get(g.get("file").getAsString());FiARCCodec.require(p!=null,"Missing packet");return verified(Arrays.copyOfRange(chunk,p[0],p[0]+p[1]),g.get("sha256").getAsString());}
        public void close(){chunk=null;extents.clear();}
    }
}
