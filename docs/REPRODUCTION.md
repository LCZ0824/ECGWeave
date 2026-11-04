# Usage

## Model

```python
import torch
from ecgweave.diagnosis.stats_model import build_classifier

model = build_classifier()  # Random initialization
batch = {
    "signals": torch.randn(1, 12, 5000),
    "reference": torch.full((1, 12, 250, 4), 0.25),
    "reference_support": torch.ones(1, 12, 250, dtype=torch.bool),
}
logits = model(batch)  # [1, 71]
```

The classifier has 6,639,486 parameters and 71 class queries with 128 features each. Use `pipeline.load_classifier(path)` to load a trusted checkpoint. Checkpoint hashes and label order are listed in [CHECKPOINTS.json](CHECKPOINTS.json).

## References and Precision

The delineator receives zero-phase, fifth-order high-pass (0.67 Hz) and low-pass (40 Hz) filtered signals. The classifier receives the original mV waveforms. BG/P/QRS/T probabilities are averaged over 20 samples, quantized to float16, restored to float32, and normalized.

Precomputed inputs may include `reference: float32[B,12,250,4]` and `reference_support: bool[B,12,250]`; omit `--delineator` in this case. For raw input, optional `sample_support: bool[B,12,5000]` marks valid samples.

Inference defaults to FP32. For CUDA BF16, add `--device cuda:0 --precision bf16`. Use only trusted checkpoint files.

## Components and Tests

- Pretraining: `ecgweave.pretraining.model.ValidECGMAE`.
- Delineation: `ecgweave.delineation.model.build_segmenter`.
- Diagnosis: `ecgweave.diagnosis.stats_model.build_classifier`.
- Settings: `configs/`.

```bash
python -m pytest -q
python -m ecgweave.smoke --device cpu
```
