# OME-Zarr layout and on-demand decoding

```text
movie.ome.zarr/
  zarr.json           # OME axes/scales and packet/chunk index
  0/zarr.json         # array shape, XERO codec and compressed decoder model
  0/c/0/0/0          # first restart chunk
  0/c/1/0/0          # next restart chunk
```

Each chunk contains one independent restart segment, comprising entropy packets of at most 64 sequential frames. An entropy packet is **not** necessarily an I-frame segment: neural prediction continues across packet boundaries until the restart interval. By default, the interval is the first power of two whose `K × H × W × 2` exceeds **100,000,000 bytes**, with a minimum of 128 frames. `--restart 128` overrides this explicitly. Changing the restart interval requires re-encoding, not retraining.

Random access starts at the preceding restart. Partial requests finish the consumed entropy packet for validation. Python yields one float32 frame at a time for on-demand analysis.

The encoder stages one restart block at a time in host RAM, verifies every pixel with an independent decoder, then writes packet files. The `xero repack` export step wraps those unchanged packets in OME-Zarr chunks. The destination becomes visible only after export completes. Existing output directories/files are refused rather than overwritten.

The encoder's `encode_report.json` counts packets, model and manifest bytes, excluding the report and filesystem overhead. `compression_ratio` uses the input dtype; `uint16_equivalent_ratio` uses two bytes per pixel. OME-Zarr metadata and chunk headers add container overhead; use the actual exported size when reporting stored bytes.

ImageJ and the XERO napari reader open these folders directly. Adding an image to the hosted browser requires a browser model export, HTTP range/CORS configuration and a catalog entry. See [reader setup](ome-zarr.md).
