# Numerical contract and model

XERO learns a causal predictor from previously **reconstructed** frames. An encoder-only seed selector inspects the source and prediction; transmitted seed symbols update sparse locations. The decoder uses the prediction, decoded seeds and spatial refinement network to reconstruct a full-frame proposal. Entropy-coded final correction symbols (and exact float32 escapes when needed) enforce the per-pixel bound.

```text
decoded history → temporal predictor → predicted frame
                                     ↓
source → seed selection → seed symbols → spatial refinement
                                     ↓
source → final bounded correction → entropy-coded packets
                                     ↓
                           reconstructed frame → history
```

The source is never available to the receiver. Prediction is sequential in the acquisition/stack dimension; spatial neural computation and entropy streams can be parallelized without substituting future source frames for decoded history.

## Learning

Training combines a seed/final-symbol rate objective, detached seed-conditioning surrogate, corrected-history rollout and auxiliary tolerance-scaled Huber terms:

```text
rate(seed symbols + final symbols + escapes)
    + 16 × Huber(spatial proposal error / local scale)
    + 0.4 × Huber(temporal prediction error / local scale)
```

The loss scale is clamped to avoid division by zero during optimization. **That training clamp is not a dark-pixel floor in the final fidelity contract.** Hard symbols/masks in the forward pass and surrogate gradients during training do not themselves prove a bound. Actual bytes and independently decoded values are checked when encoding.

Floating-point training is followed by decoder-model export with A13 activations, W8 weights and checked bounded integer convolution accumulation. Fixed activation/probability tables and the serialized profile govern decoding. Some preprocessing, interpolation and final reconstruction still involve defined floating-point operations; do not describe this as every operation being an integer operation or as a formal proof for every driver/device.

## Source domain

For finite supplied input `x`, `abs(decoded - x) <= epsilon * abs(x)`. Zero is exact; signed and difficult values may require exact float32 escapes. No implicit min–max normalization, added intensity baseline, display LUT or clipping changes the user's target. Input `uint8`/`uint16` is represented exactly in float32. Float64 input is rejected rather than silently rounded.

When input is denoised, this certifies **compression relative to denoised input**, not denoising relative to biological ground truth. Downstream equivalence, segmentation and behavioral decoding are separate empirical evaluations, not guaranteed by the pixelwise bound.

## Compatibility

The CLI export supports causal models. Unsupported future/contextual model families are rejected. Test a new decoding platform against independent reference pixels using the ImageJ compatibility check.

## Trust

Training checkpoints use PyTorch serialization and must be trusted. Archives contain array-based decoder models and integrity hashes; hashes detect accidental corruption, not malicious replacement. No security audit for hostile inputs is claimed. Do not expose a privileged server-side decoder to untrusted uploads.
