# Example movie

**[Download XERO_demo.zip](XERO_demo.zip?raw=true)** — approximately 0.46 MB.

Unzip it, choose **Plugins → XERO → Open compressed image** in ImageJ/Fiji, and select the extracted `XERO_demo` folder. Open frames **1–160**. No Python or training is needed.

| Property | Value |
|---|---|
| Content | Synthetic fluorescence-like movie with three fluctuating spots |
| Dimensions | 64 × 64 pixels, 160 frames |
| Relative-error bound | 1% |
| Restart interval | 128 frames |
| Reconstructed pixels | float32 |

The ZIP contains the decoder model, manifest and three entropy packets. It contains no biological or private data. The model was trained for ten demonstration steps, so this example is **not** a benchmark of XERO compression quality or speed. The software and generated example are MIT licensed.

To generate your own example with the command-line tools:

```bash
bash tutorials/02_roundtrip.sh
python tutorials/04_package_imagej_example.py \
  --archive outputs/demo/archive --output outputs/XERO_demo.zip
```

`metadata.json` records this download's size, SHA-256 and independently decoded pixel hash for validation.
