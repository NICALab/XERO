# Archive layout and on-demand decoding

```text
compressed/
  manifest.json       # shape, bound, restart schedule, packet sizes and hashes
  model.npz           # decoder-only arrays and numerical profile; no optimizer
  group_000000.bin    # entropy packet, at most 64 sequential frames
  group_000001.bin
  ...
  encode_report.json  # measurement/verification receipt, not decoding input
```

An entropy packet is **not** necessarily an I-frame segment. Neural prediction continues across packet boundaries until the restart interval. By default, the interval is the first power of two whose `K × H × W × 2` exceeds **100,000,000 bytes**, with a minimum of 128 frames. `--restart 128` overrides this explicitly. Changing the restart interval requires re-encoding, not retraining.

Random access starts at the preceding restart. Partial requests finish the consumed entropy packet for validation. Python yields one float32 frame at a time for on-demand analysis.

The encoder stages one restart block at a time in host RAM, verifies every pixel with an independent decoder, then writes packet files. The manifest is published only after success. `INCOMPLETE` means an interrupted/failed encode and must not be treated as a valid archive. Existing output directories/files are refused rather than overwritten.

`encode_report.json` counts packets, model and manifest bytes, excluding the report and filesystem overhead. `compression_ratio` uses the input dtype; `uint16_equivalent_ratio` uses two bytes per pixel.

ImageJ opens these folders directly. Adding an archive to the hosted browser requires a browser model export, HTTP range/CORS configuration and a catalog entry.
