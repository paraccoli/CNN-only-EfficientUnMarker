# CNN-only EfficientUnMarker

A fast, lightweight, and practical watermark removal pipeline using only a CNN-based detector (MobileNetV3).  
This repository provides a minimal, reproducible baseline for universal watermark removal with a focus on speed, memory efficiency, and simplicity.

## Overview

- **Purpose:** Remove visible watermarks using a 3-stage pipeline (frequency analysis → multi-scale attack → quality safeguard) guided only by a CNN-based detector.
- **Target hardware:** RTX 5070 Ti–class GPU or similar.
- **Design goals:** <10s per image, <8GB VRAM, >90% removal success.
- **Relation to EfficientUnMarker:** Distilled version of the full EfficientUnMarker project, containing only the CNN-based (MobileNetV3) detection and attack components.

## Features

- **Stage 1 – Frequency analysis:** `torch.fft`-based spectrum analysis + MobileNetV3 features to generate a frequency band mask.
- **Stage 2 – Multi-scale attack:** 256→original resolution optimization with early stopping and a soft PSNR penalty to avoid excessive degradation.
- **Stage 3 – Optional refinement:** Additional quality refinement stage (disabled by default in this baseline).
- **Metrics:** Success flag, detection score, PSNR, runtime, and peak VRAM usage.
- **Configurable:** YAML-based configuration for attack hyperparameters and stopping criteria.

## Dataset

Evaluation uses the **Large-scale Common Watermark Dataset**:

- Source: [Kaggle - Large-scale Common Watermark Dataset](https://www.kaggle.com/datasets/kamino/largescale-common-watermark-dataset)
- Contains realistic visible watermarks for benchmarking removal algorithms.

## Quick Start

### Install dependencies

```txt
pip install -r requirements.txt
```

### Download dataset (optional)

Download the Large-scale Common Watermark Dataset from the link above and place it in `datasets/`.

### Run benchmark (example)

```txt
python experiments/benchmark_cnn_only.py \
  --config configs/fast_cnn_only.yaml \
  --input_dir <watermarked_dir> \
  --clean_dir <clean_dir> \
  --wm_suffix v2 \
  --output results/benchmarks/phaseA_cnn_only.csv
```

- `--wm_suffix v2` matches files like `xxxv2.jpg` (watermarked) to `xxx.jpg` (clean).

### Results

- Per-image metrics and aggregates are saved under `results/benchmarks/`.

## Directory Structure

```txt
core/          # Main attack pipeline and CNN detector
utils/         # Image I/O, transforms, metrics
configs/       # YAML configs (e.g., fast_cnn_only.yaml)
experiments/   # Benchmark and evaluation scripts
datasets/      # (Place downloaded datasets here)
requirements.txt
README.md
```

## Reference Performance

Measured on 44 real images from the Large-scale Common Watermark Dataset:

- Success rate: **95.5%**
- Avg runtime: **1.5 s / image**
- Peak VRAM: **0.26 GB**
- Avg PSNR: **23.5 dB**

(Results may vary depending on watermark scheme, dataset, and GPU.)

## License / Attribution

This repository is intended for research and benchmarking.  
For commercial or large-scale use, please refer to and credit the original UnMarker work:

- UnMarker: A Universal Attack on Defensive Image Watermarking  
  <https://github.com/andrekassis/ai-watermark>

Dataset credit:
- Large-scale Common Watermark Dataset
  <https://www.kaggle.com/datasets/kamino/largescale-common-watermark-dataset>
