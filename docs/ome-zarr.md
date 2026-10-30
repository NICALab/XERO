# XERO OME-Zarr format

XERO uses **Zarr v3**, **OME-NGFF 0.5** and the **`xero.restart_v1`** codec. Readers need the XERO codec; OME-Zarr support alone is not sufficient.

For usage, see [ImageJ](../imagej/README.md), [napari](../napari/README.md) or [Python](python.md).

## Layout

```text
movie.ome.zarr/
  zarr.json           OME axes/scales and packet/chunk index
  0/zarr.json         array shape, XERO codec and embedded decoder model
  0/c/0/0/0           first restart chunk
  0/c/1/0/0           next restart chunk
  encode_report.json encoding statistics
```

Arrays are float32 with `[t,y,x]` or `[z,y,x]` axes and one resolution level. Channels and planes are stored as separate images. The model is stored once in array metadata. `--axis` specifies the frame axis; physical pixel sizes and time intervals are not inferred.

## Restarts and random access

Each chunk contains one independent restart segment, split into entropy packets of at most 64 frames. Prediction continues across packet boundaries until the next restart.

The default restart interval is the first power of two whose `K × H × W × 2` exceeds **100,000,000 bytes**, with a minimum of 128 frames. Override it with `--restart 128`. Changing it requires re-encoding, not retraining.

Random access decodes from the preceding restart and finishes each consumed entropy packet for validation. Complete restart chunks can be downloaded independently.

## Writing and integrity

`xero encode` verifies every reconstructed pixel before writing each chunk. It stages one restart block at a time; no intermediate archive or repacking step is needed. Existing destinations are not overwritten. Interrupted runs leave a separate `.incomplete-*` directory.

Checksums detect damaged models, packets and chunks. Missing chunks raise errors rather than becoming blank frames. Only open trusted archives: checksums verify integrity, not authenticity.

`encode_report.json` reports metadata, embedded model and chunk bytes, excluding the report itself and filesystem overhead. `compression_ratio` uses the input dtype; `uint16_equivalent_ratio` uses two bytes per pixel.

The hosted browser additionally uses a browser model export, HTTP range/CORS configuration and a catalog entry.

[OME-NGFF specification](https://ngff.openmicroscopy.org/0.5/)
