# ECGWeave

Delineation-guided multiscale learning for multi-label 12-lead ECG diagnosis.

## Installation

Python 3.10+ and PyTorch are required.

```bash
python -m pip install -e '.[dev]'
```

## Inference

Input: `signals` in an NPZ file, `float32[B, 12, 5000]` in mV, sampled at 500 Hz. Lead order: I, II, III, aVR, aVL, aVF, V1–V6.

```bash
ecgweave-predict --input input.npz --checkpoint stats_prior20.pt \
  --delineator delineator.pt --output prediction.npz
```

The output contains 71 logits, sigmoid scores, and label names. See [Usage](docs/REPRODUCTION.md) for model construction and precomputed references. Models and training objectives are in `src/ecgweave/`; settings are in `configs/`.

## Data and Weights

This repository provides model components and inference code. Training weights, datasets, patient splits, and the full training pipeline are not included. Dataset access follows the terms of each provider. Hospital data requests: Kang Chen, ck11208@rjh.com.cn, subject to institutional approval.

## License

[Apache-2.0](LICENSE). See [NOTICE](NOTICE) and [third-party licenses](THIRD_PARTY_LICENSES.md) for attribution.
