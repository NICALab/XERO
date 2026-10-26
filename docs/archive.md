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

Random access starts at the preceding restart and restores its causal prefix. Partial requests finish the consumed entropy packet for terminal-state validation. Python yields one copied float32 frame at a time; the caller can compute an ROI signal immediately instead of writing a decompressed volume. Neural state/workspace is not the same as the output cache and can still be large.

The encoder stages one restart block at a time in host RAM, verifies every pixel with an independent decoder, then writes packet files. The manifest is published only after success. `INCOMPLETE` means an interrupted/failed encode and must not be treated as a valid archive. Existing output directories/files are refused rather than overwritten.

`encode_report.json` includes actual bytes for packets, model and manifest. It excludes the report itself and filesystem allocation overhead. `compression_ratio` uses the supplied input dtype; `uint16_equivalent_ratio` is an explicitly named alternative. No timing field in this wrapper is presented as a paper speed benchmark.

The ImageJ plugin can open these folders directly. The hosted browser currently serves its configured catalog. Hosting new archives requires the browser's provider-side model export and HTTP range/CORS configuration; copying a folder to a random public URL does not automatically add it to that catalog. No Cloudflare account, token or private deployment configuration is included in this repository.
