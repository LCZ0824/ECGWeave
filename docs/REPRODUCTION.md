# Usage

## Model

```python
import torch
from ecgweave.diagnosis.stats_model import build_classifier

model = build_classifier()
logits = model({
    "signals": torch.randn(1, 12, 5000),
    "reference": torch.full((1, 12, 250, 4), 0.25),
    "reference_support": torch.ones(1, 12, 250, dtype=torch.bool),
})  # [1, 71]
```

## Inference

Input NPZ: `signals`, float32 `[B,12,5000]`, 500 Hz, mV. Lead order: I, II, III, aVR, aVL, aVF, V1–V6. The example above creates a randomly initialized model; inference requires locally supplied [checkpoints](CHECKPOINTS.json).

```bash
ecgweave-predict --input input.npz --checkpoint stats_prior20.pt \
  --delineator delineator.pt --output prediction.npz
```

Outputs: 71 logits, sigmoid scores, and label names. For CUDA BF16, add `--device cuda:0 --precision bf16`.

Precomputed `reference` (float32 `[B,12,250,4]`, BG/P/QRS/T) and `reference_support` (bool `[B,12,250]`) replace `--delineator`. Raw inputs may include `sample_support` (bool `[B,12,5000]`).

The delineator uses sequential fifth-order, zero-phase 0.67-Hz high-pass and 40-Hz low-pass filtering; diagnosis uses raw mV. References average 20-sample probabilities, round through float16, and normalize in FP32.
