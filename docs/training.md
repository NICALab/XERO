# Training on your microscopy data

## Prepare frozen targets

Use one channel and one ordered recording/stack per input. Denoise first if that is your scientific workflow. XERO does not alter that decision or run a denoiser implicitly.

```bash
xero prepare --input sample_a.h5 sample_b.h5 --output outputs/bank
```

Each source is split into contiguous 80/10/10 train/calibration/test intervals. Targets are stored as float32 HDF5, with the original intensity values. Separate roles never overlap within a recording. Normalization is estimated only from the training bank. These are within-recording holdouts, **not independent-animal generalization**. For specimen-level generalization, prepare separate banks explicitly and leave the held-out specimen out of training and calibration.

The preparer writes derived HDF5 copies, so budget disk space. It does not overwrite input. Temporal/z-boundaries are respected; absent context is filled with the training mean, not frames from another role/recording. For sparse heterogeneous acquisitions, provide representative training recordings covering the signal-density distribution; the cache balances recordings but does not automatically establish biological representativeness.

## Train

```bash
xero train --bank outputs/bank --out outputs/model_1pct \
  --tolerance 0.01 --steps 50000 --history 64 --rollout 64 \
  --patch 128 --batch 8 --tiles 8 --refresh 500 \
  --width 32 --refiner-width 16 --compile --skip-short-audits
```

The 50,000-step example is a starting point, not a guarantee of convergence. Reduce batch/tile/patch sizes for limited VRAM. The demo's 10 steps and two-step rollout are **software checks only**. Architecture and tolerance are recorded in each checkpoint. Supported release bounds are positive fractions up to 0.5; large tolerances require scientific validation, not just a successful codec audit.

Output includes `latest.pt`, numbered checkpoints, surrogate loss/rate logs, TensorBoard records and previews. A completed training run does not establish a good compression ratio. `best_estimated.pt` is selected by a surrogate training rate, not held-out actual compressed size.

## Select using calibration bytes

Encode the calibration recording(s) with candidate checkpoints in separate output directories:

```bash
xero encode --input outputs/bank/record000_calibration.h5 \
  --checkpoint outputs/model_1pct/latest.pt --output outputs/calibration_latest
```

Compare actual compressed sizes, including model and manifest. Freeze the chosen checkpoint **before** encoding `record000_test.h5`. Do not tune on the test compression ratio or downstream metric. Use the same input dtype and accounting basis in comparisons; float32 and uint16-equivalent ratios differ by 2× for identical pixels and bytes.

The error tolerance belongs to the trained checkpoint. To evaluate 1%, 5% and 20%, train/select separate checkpoints at `0.01`, `0.05` and `0.20`; do not relabel one archive. Post-training integer export occurs during `xero encode`; changing to this deployment representation alone does not require inventing a different training objective.

Do not deploy an untrained, zero-output-head checkpoint. A degenerate initialized checkpoint failed the independent reconstruction audit during release stress testing; the encoder rejected it and did not publish a manifest. A successful optimizer step or model export is not a decoding qualification. If any checkpoint fails the audit, keep the incomplete output for diagnosis, select/calibrate another checkpoint and do not bypass the check or relabel the files.

## Resume

```bash
xero train --bank outputs/bank --out outputs/model_1pct \
  --tolerance 0.01 --steps 50000 --history 64 --rollout 64 \
  --patch 128 --batch 8 --tiles 8 --refresh 500 \
  --width 32 --refiner-width 16 --compile --skip-short-audits --resume
```

Keep the bank, architecture and schedule fixed. For preplanned interruptions, `--stop-at` saves a segment without redefining the total schedule. Checkpoints are trusted Python artifacts; never load a checkpoint from an unknown source.
