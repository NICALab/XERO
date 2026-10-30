# Training on your microscopy data

## Prepare frozen targets

Use one channel and one ordered recording/stack per input. Denoise beforehand if required by your workflow.

```bash
xero prepare --input sample_a.h5 sample_b.h5 --output outputs/bank
```

Each source is split into contiguous 80/10/10 training/calibration/test intervals and saved as float32 HDF5. Normalization is estimated from training data only. For specimen-level evaluation, use separate banks and exclude the held-out specimen from training and calibration.

Preparation creates HDF5 copies, so allow additional disk space. Context never crosses recording or split boundaries. Include representative sparse and dense regions in training; the cache balances recordings.

## Train

```bash
xero train --bank outputs/bank --out outputs/model_1pct \
  --tolerance 0.01 --steps 50000 --history 64 --rollout 64 \
  --patch 128 --batch 8 --tiles 8 --refresh 500 \
  --width 32 --refiner-width 16 --compile --skip-short-audits
```

Adjust training duration using calibration performance, and reduce batch/tile/patch sizes for limited VRAM. Supported tolerances are positive fractions up to `0.5`; select a bound appropriate for your analysis.

Outputs include `latest.pt`, numbered checkpoints, logs and previews. `best_estimated.pt` uses the surrogate training rate; select the final checkpoint using actual calibration bytes.

## Select using calibration bytes

Encode the calibration recording(s) with candidate checkpoints in separate output directories:

```bash
xero encode --input outputs/bank/record000_calibration.h5 \
  --checkpoint outputs/model_1pct/latest.pt --output outputs/calibration_latest.ome.zarr --axis t
```

Compare actual compressed sizes, including model and OME-Zarr metadata. Use `--axis z` for structural stacks. Freeze the chosen checkpoint **before** encoding `record000_test.h5`. Do not tune on the test compression ratio or downstream metric. Use the same input dtype and accounting basis in comparisons; float32 and uint16-equivalent ratios differ by 2× for identical pixels and bytes.

The checkpoint fixes the tolerance. For 1%, 5% and 20%, train/select separate checkpoints at `0.01`, `0.05` and `0.20`. Integer model export occurs during `xero encode`, which independently verifies the reconstruction before completing the archive. Do not bypass a failed reconstruction check.

## Resume

```bash
xero train --bank outputs/bank --out outputs/model_1pct \
  --tolerance 0.01 --steps 50000 --history 64 --rollout 64 \
  --patch 128 --batch 8 --tiles 8 --refresh 500 \
  --width 32 --refiner-width 16 --compile --skip-short-audits --resume
```

Keep the bank, architecture and schedule fixed. For preplanned interruptions, `--stop-at` saves a segment without redefining the total schedule. Checkpoints are trusted Python artifacts; never load a checkpoint from an unknown source.
