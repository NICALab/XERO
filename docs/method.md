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

Prediction is sequential along time or z; spatial neural computation and entropy streams run in parallel.

## Learning

Training combines a seed/final-symbol rate objective, detached seed-conditioning surrogate, corrected-history rollout and auxiliary tolerance-scaled Huber terms:

```text
rate(seed symbols + final symbols + escapes)
    + 16 × Huber(spatial proposal error / local scale)
    + 0.4 × Huber(temporal prediction error / local scale)
```

Training uses hard symbols/masks with surrogate gradients. The loss scale is clamped for numerical stability; the final error bound has no dark-pixel floor. Encoding independently checks reconstructed values against the input.

Floating-point training is followed by export with A13 activations, W8 weights and bounded integer convolution accumulation. Fixed activation/probability tables and the serialized profile govern decoding; preprocessing, interpolation and reconstruction also use specified floating-point operations.

## Source domain

For finite supplied input `x`, `abs(decoded - x) <= epsilon * abs(x)`. Zero is exact; signed and difficult values may require exact float32 escapes. No implicit min–max normalization, added intensity baseline, display LUT or clipping changes the user's target. Input `uint8`/`uint16` is represented exactly in float32. Float64 input is rejected rather than silently rounded.

When input is denoised, this certifies **compression relative to denoised input**, not denoising relative to biological ground truth. Downstream equivalence, segmentation and behavioral decoding are separate empirical evaluations, not guaranteed by the pixelwise bound.

## Compatibility

The CLI export supports causal models. Unsupported future/contextual model families are rejected. Test a new decoding platform against independent reference pixels using the ImageJ compatibility check.

## Trust

Only load trusted checkpoints and archives. Integrity hashes detect corruption, not authenticity.
