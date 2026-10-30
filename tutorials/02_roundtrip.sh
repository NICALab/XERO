#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root after installation. Creates only outputs/demo/.
python tutorials/01_make_demo.py
xero prepare --input outputs/demo/input.h5 --output outputs/demo/bank
# A 10-step software demo, NOT a trained model performance claim.
xero train --bank outputs/demo/bank --out outputs/demo/model \
  --steps 10 --tolerance 0.01 --history 64 --rollout 2 \
  --width 32 --refiner-width 16 --patch 32 --batch 2 --tiles 2 \
  --refresh 10 --skip-short-audits
xero encode --input outputs/demo/input.h5 --checkpoint outputs/demo/model/latest.pt \
  --output outputs/demo/archive --restart 128
xero repack --archive outputs/demo/archive --output outputs/demo/movie.ome.zarr --axis t
xero decode --archive outputs/demo/movie.ome.zarr --output outputs/demo/reconstructed.h5
xero verify --archive outputs/demo/movie.ome.zarr --input outputs/demo/input.h5
