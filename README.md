# Keyword Spotting on RISC-V with TensorFlow Lite Micro

Course final project for CMU 15-642. This repository studies keyword spotting (KWS) model training, INT8 quantization, TensorFlow Lite Micro deployment, and inference evaluation on a RISC-V software stack using Spike.

## Quick Start

From a clean machine, the shortest path from clone to a first local run is:

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install tensorflow numpy

bash tools/setup_third_party.sh
```

Then download and extract Speech Commands v0.02 so the dataset lives at:

```bash
speech-commands-v2/
```

A small smoke-test training run:

```bash
source .venv/bin/activate
python train_kws.py \
  --data-dir speech-commands-v2 \
  --model ds_cnn \
  --epochs 1 \
  --batch-size 32 \
  --max-train-samples 256 \
  --max-val-samples 128 \
  --max-test-samples 128 \
  --export-int8
```

If training succeeds, you can export an example INT8 MFCC input:

```bash
python export_int8_mfcc.py \
  --wav speech-commands-v2/yes/0a7c2a8d_nohash_0.wav \
  --model artifacts/ds_cnn/model_int8.tflite \
  --output artifacts/ds_cnn/input_int8_mfcc.npy
```

To build and run the RISC-V demo, make sure `riscv64-unknown-elf-g++`, `spike`, and `pk` are installed and visible on your machine, then run:

```bash
make check-tools
make
make run
```

## Overview

The project pipeline is:

1. Train a KWS model in TensorFlow.
2. Export an INT8 TFLite model.
3. Convert the model into a C array for TensorFlow Lite Micro.
4. Build a RISC-V ELF that runs inference with TFLM.
5. Evaluate predictions through Spike on Speech Commands test samples.

Following the *Hello Edge* methodology, two model-side tasks are wired up:

- **Task 1 — Baseline DS-CNN** at the standard 49×40 MFCC features (40 MFCC bins from a 40 ms / 20 ms STFT window). One model, reports accuracy / memory / ops. Entry point: `train_kws.py`.
- **Task 2 — Hyperparameter sweep** to find the best DS-CNN per memory/ops budget tier (S/M/L). Two-stage Plan-C: random sample + analytic budget filter, train survivors briefly, then fully retrain the top-K. Entry point: `sweep_ds_cnn.py`.

## Repository Layout

- `train_kws.py`: train a single CNN or DS-CNN model (Task 1, plus optional retrain of a sweep winner).
- `sweep_ds_cnn.py`: two-stage hyperparameter sweep that finds best DS-CNN models per S/M/L budget (Task 2).
- `export_int8_mfcc.py`: convert a WAV file into the quantized INT8 MFCC input expected by the TFLite model.
- `evaluate_spike_testset.py`: run test-set evaluation through Spike.
- `kws/`: dataset loading and model definitions.
- `tflm_demo/`: TensorFlow Lite Micro inference demo code.
- `artifacts/`: generated training and export outputs. This directory is excluded from Git and should be regenerated locally.
- `sweep_artifacts/`: outputs of `sweep_ds_cnn.py` (CSV, summary, per-tier best.keras + report.json). Excluded from Git.
- `sweep_cache/`: on-disk MFCC feature cache used by both training scripts. Excluded from Git.
- `tools/setup_third_party.sh`: helper script to clone external source dependencies locally.

## File Guide

### Top-level files

- `README.md`: project overview, setup steps, and usage instructions.
- `.gitignore`: excludes local environments, datasets, generated artifacts, and large third-party dependencies from Git.
- `Makefile`: builds and runs the RISC-V TensorFlow Lite Micro demo.
- `train_kws.py`: main training entry point for a single keyword-spotting model and INT8 export. Supports `--retrain-from-sweep` for retraining a swept architecture with full noise augmentation.
- `sweep_ds_cnn.py`: two-stage hyperparameter sweep over the DS-CNN search space (layers / filters / kernel / first stride / MFCC bins) under the S, M, L int8 memory and ops budgets.
- `export_int8_mfcc.py`: converts a WAV file into the quantized model input used by TFLM inference.
- `evaluate_spike_testset.py`: evaluates the test split by repeatedly running the RISC-V ELF on Spike.
- `Project Proposal_ Optimization and Performance Analysis of Keyword Spotting (KWS) on RISC-V Platforms.docx`: project proposal source document.
- `Project_Proposal_shengfeg_kangjiel_yantingj.pdf`: exported proposal PDF.

### `kws/`

- `kws/__init__.py`: package marker for the Python KWS utilities.
- `kws/data.py`: dataset indexing, train/validation/test split handling, audio preprocessing, MFCC extraction, and TensorFlow dataset construction. MFCC features cached on disk via `DatasetConfig.cache_dir`; the cache key includes `feature_bins` so MFCC-bin-10 and MFCC-bin-40 caches stay separate.
- `kws/models.py`: model definitions for the baseline Sainath CNNs and a fully parameterized DS-CNN (`build_ds_cnn(layers, filters, kernel, first_stride, dropout)`), plus per-layer MAC and activation-memory estimators used by the sweep's analytic budget filter.

### `tflm_demo/`

- `tflm_demo/README.md`: notes about the TensorFlow Lite Micro demo code and expected label order.
- `tflm_demo/main.cc`: minimal demo entry point that feeds one input through the model.
- `tflm_demo/kws_inference.h`: public inference wrapper interface.
- `tflm_demo/kws_inference.cc`: TFLM interpreter setup, op registration, tensor arena setup, and inference execution.
- `tflm_demo/test_input_data.h`: declaration for the built-in example INT8 input tensor.
- `tflm_demo/test_input_data.cc`: example quantized MFCC input compiled into the demo binary.

### `tools/`

- `tools/setup_third_party.sh`: clones local copies of TFLM, `riscv-isa-sim`, and `riscv-pk` into `third_party/`.
- `tools/size_report.sh`: summarizes the deployment footprint (model size, ELF Flash/RAM usage, tensor-arena peak). See [Footprint & Size Report](#footprint--size-report).

### Generated or local-only paths

- `artifacts/`: training outputs such as `.keras`, `.tflite`, reports, and generated model C arrays.
- `build/`: compiled outputs such as `kws_demo.elf`.
- `speech-commands-v2/`: local Speech Commands dataset directory.
- `third_party/`: local source checkouts of external dependencies.
- `tools/riscv/`: local machine-specific RISC-V binaries and libraries.

## Requirements

### Python

This project uses a local virtual environment at `.venv`.

Create it again if needed:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

The training and export scripts require TensorFlow and common Python packages such as NumPy. If your current environment does not already provide them, install them into `.venv` before running the scripts.

### RISC-V toolchain

The `Makefile` expects:

- `riscv64-unknown-elf-g++`
- `spike`
- `pk`

The default `pk` path in this repository is:

```bash
/opt/homebrew/Cellar/riscv-pk/main/riscv64-unknown-elf/bin/pk
```

If your local installation differs, override it when invoking `make`:

```bash
make PK=/path/to/pk
```

This repository also excludes the locally installed `tools/riscv/` directory from Git because it contains machine-specific binaries and libraries.

## Dataset

This repository does not include the Speech Commands dataset or the dataset zip because they are too large for GitHub.

Official references:

- TensorFlow Datasets page: <https://www.tensorflow.org/datasets/catalog/speech_commands>
- Direct download: <https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz>
- Paper: <https://arxiv.org/abs/1804.03209>

Download Speech Commands v0.02 separately and extract it under:

```bash
speech-commands-v2/
```

Most scripts default to that location, but you can override it with `--data-dir`.

One way to download and unpack it locally is:

```bash
curl -L https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz -o speech_commands_v0.02.tar.gz
mkdir -p speech-commands-v2
tar -xzf speech_commands_v0.02.tar.gz -C speech-commands-v2 --strip-components=1
```

## Training

Two model-side tasks live in this repo. They can be run independently.

### Task 1 — Baseline DS-CNN at 49×40 MFCC

Single training run with the standard *Hello Edge* MFCC configuration (40 MFCC bins, 40 ms frame, 20 ms stride → 49×40 features per 1 second clip). Reports accuracy / int8 memory / ops.

```bash
source .venv/bin/activate
python train_kws.py \
  --data-dir speech-commands-v2 \
  --model ds_cnn \
  --epochs 15 \
  --batch-size 64 \
  --cache-dir sweep_cache \
  --export-int8
```

The available `--model` choices are `ds_cnn`, `cnn_trad_fpool3`, and `cnn_one_fstride4` (the last two are the two Sainath & Parada 2015 baselines).

Generated outputs land in `artifacts/<model_name>/`:

- `best.keras`, `final.keras`, `history.csv`, `report.json`
- `model_int8.tflite` (when `--export-int8` is set)
- `model_data.cc`, `model_data.h` (when generated for the TFLM demo)

`report.json` contains parameter count, MACs, ops, int8 weight bytes, peak activation bytes, fp32 train/val/test accuracy, and (with `--export-int8`) int8 train/val/test accuracy.

### Task 2 — Hyperparameter sweep for best S/M/L models

Two-stage Plan-C sweep that finds the best DS-CNN per int8 memory / ops budget tier:

| Tier | Memory | Ops |
| ---- | ------ | --- |
| S    | ≤ 80 KB  | ≤ 6 MOps  |
| M    | ≤ 200 KB | ≤ 20 MOps |
| L    | ≤ 500 KB | ≤ 80 MOps |

Search space: `layers ∈ {4, 5, 6}`, `filters ∈ {64, 76, 96, 128, 172, 276}`, `kernel ∈ {(3,3), (5,3), (10,4), (20,8)}`, `first_stride ∈ {(1,1), (2,1), (1,2), (2,2)}`, `mfcc_bins ∈ {10, 40}`.

Stage 1 randomly samples configurations per tier, analytically rejects those over budget (using model parameter count + peak int8 activation memory), and trains each survivor for a few epochs. Stage 2 takes the top-K of stage 1 per tier and trains them longer with EarlyStopping; the best validation accuracy per tier is saved as the final S / M / L model.

Run the full sweep:

```bash
source .venv/bin/activate
python sweep_ds_cnn.py \
  --data-dir speech-commands-v2 \
  --batch-size 512 \
  --no-noise-aug
```

Recommended on a cloud GPU (e.g. Kaggle Tesla T4):

```bash
python sweep_ds_cnn.py \
  --data-dir /kaggle/input/datasets/sylkaladin/speech-commands-v2 \
  --batch-size 512 \
  --no-noise-aug
```

Useful flags:

- `--candidates-per-tier 30` (default) — target number of in-budget candidates to sample per tier.
- `--top-k 3` (default) — number of stage-1 winners promoted to stage 2.
- `--stage1-epochs 5 --stage2-epochs 15` (defaults) — epoch budgets.
- `--tiers S M` — restrict the sweep to specific budget tiers.
- `--no-noise-aug` — disable per-batch background-noise mixing during the sweep. Removes the per-batch wav read from the data pipeline (3–5× faster on CPU-bound machines). Recommended for the sweep itself; re-enable for the final retrain.
- `--dry-run` — only sample + analytically score candidates; skip training.

Outputs land in `sweep_artifacts/`:

```
sweep_artifacts/
├── sweep_results.csv             # every candidate, both stages: config + cost + val_acc
├── summary.json                  # final per-tier winners
├── ds_cnn_S_top{1..3}/           # stage-2 candidates kept for inspection
├── ds_cnn_M_top{1..3}/
├── ds_cnn_L_top{1..3}/
├── ds_cnn_S/{best.keras, report.json}   # final winner per tier
├── ds_cnn_M/{best.keras, report.json}
└── ds_cnn_L/{best.keras, report.json}
```

### Final retrain from a sweep result (with full noise augmentation)

The sweep itself runs with `--no-noise-aug` for speed. To obtain a production-quality model for the chosen architecture, retrain it through `train_kws.py --retrain-from-sweep`, which reads the architecture and MFCC bin count from the sweep's `report.json` and re-applies background noise augmentation by default:

```bash
python train_kws.py --retrain-from-sweep sweep_artifacts/ds_cnn_S/report.json --epochs 30 --cache-dir sweep_cache --export-int8
python train_kws.py --retrain-from-sweep sweep_artifacts/ds_cnn_M/report.json --epochs 30 --cache-dir sweep_cache --export-int8
python train_kws.py --retrain-from-sweep sweep_artifacts/ds_cnn_L/report.json --epochs 30 --cache-dir sweep_cache --export-int8
```

Outputs go to `artifacts/ds_cnn_<tier>_retrained/`.

### Common options for both scripts

- `--keywords yes no stop go` selects a custom keyword subset.
- `--max-train-samples`, `--max-val-samples`, `--max-test-samples` are useful for smoke tests.
- `--output-dir` changes the artifact location.
- `--cache-dir sweep_cache` reuses MFCC features across runs (the cache key includes `feature_bins/mel_bins/keywords`, so 10-bin and 40-bin caches coexist safely).

## Export INT8 MFCC Input

Convert a WAV file into the quantized `49x40x1` INT8 MFCC input expected by the default DS-CNN TFLite model (or `49x10x1` if the model was trained with `mfcc_bins=10`):

```bash
source .venv/bin/activate
python export_int8_mfcc.py \
  --wav speech-commands-v2/yes/0a7c2a8d_nohash_0.wav \
  --model artifacts/ds_cnn/model_int8.tflite \
  --output artifacts/ds_cnn/input_int8_mfcc.npy \
  --format npy
```

You can also export in `json` or `c_array` format with `--format`.

## Build the TFLM Demo

The default build expects the DS-CNN model data at:

```bash
artifacts/ds_cnn/model_data.cc
```

Build the RISC-V ELF:

```bash
make
```

Run it on Spike:

```bash
make run
```

Helpful checks:

```bash
make print-tools
make check-tools
```

## Footprint & Size Report

For an edge / TinyML deployment the interesting numbers are not a single "binary size" but four separate figures: the model, the code in Flash, the RAM working set, and the peak tensor-arena use at runtime. The helper script `tools/size_report.sh` prints all of them in one go.

Static-only report (no simulator run):

```bash
make                         # ensure build/kws_demo.elf is up to date
./tools/size_report.sh
```

Static + runtime report (runs the ELF under Spike and captures the actual peak arena):

```bash
RUN_SPIKE=1 ./tools/size_report.sh
```

Example output (DS-CNN INT8 build):

```
-- Model --
  int8 tflite:                  41160 B   (40.2 KB)

-- ELF sections --
  .text    (code, Flash)       139070 B
  .rodata  (const, Flash)       28624 B
  .data    (init RW, F->R)      44064 B
  .bss     (zero RW, RAM)      104440 B

-- Deployment footprint --
  Flash = text+rodata+data+sdata:   212214 B  (207.2 KB)
  RAM   = data+sdata+bss+sbss:      149080 B  (145.6 KB)

-- Runtime (spike) --
ARENA used=23872 / reserved=102400 bytes
FOOTPRINT arena_used=23872 arena_reserved=102400 input_bytes=490 output_bytes=6
```

What the numbers mean:

- **Model (`.tflite`)**: the raw int8 model file; this is what ships as a C array via `artifacts/ds_cnn/model_data.cc`.
- **Flash** (`.text + .rodata + .data + .sdata`): what you must fit in non-volatile storage on the target.
- **RAM** (`.data + .sdata + .bss + .sbss`): what the target must provide as writable memory, including the static tensor arena.
- **Arena used / reserved**: how much of the compile-time arena (`kTensorArenaSize` in [tflm_demo/kws_inference.cc](tflm_demo/kws_inference.cc)) TFLM actually needed after `AllocateTensors()`. The arena is allocated in `.bss`, so shrinking it reduces the RAM number 1:1. The `ARENA used=...` line is printed at startup; the `FOOTPRINT ...` line is printed after inference by [tflm_demo/main.cc](tflm_demo/main.cc).

Environment overrides for the script: `ELF`, `TFLITE`, `SIZE_TOOL` (default `riscv64-unknown-elf-size`), `PK`, `RUN_SPIKE`.

## Evaluate on Spike

Run Speech Commands test-set evaluation through Spike:

```bash
source .venv/bin/activate
python evaluate_spike_testset.py \
  --data-dir speech-commands-v2 \
  --model artifacts/ds_cnn/model_int8.tflite \
  --elf build/kws_demo.elf \
  --summary-json artifacts/ds_cnn/spike_test_eval_summary.json \
  --samples-csv artifacts/ds_cnn/spike_test_eval_samples.csv
```

Useful options:

- `--limit` evaluates only the first N test samples.
- `--start-index` resumes from a later point in the test split.
- `--progress-every` controls logging frequency.
- `--keywords` must match the trained model configuration.

## Third-Party Dependencies

The project depends on external repositories such as:

- TensorFlow Lite Micro
- `riscv-isa-sim`
- `riscv-pk`

They are intentionally excluded from the main GitHub repository because they are large and contain their own Git history.

Clone them locally with:

```bash
bash tools/setup_third_party.sh
```

If you prefer, you can also manage them as Git submodules instead of plain local clones.

For a public course-project repository, this is usually cleaner than committing nested repositories directly.

If you later decide to publish with submodules, that is still a good option. The main thing to avoid is committing nested `.git/` directories inside the parent repository.

## Uploading to GitHub

Before the first push:

1. Make sure large local data and generated outputs stay ignored.
2. Keep `third_party/` and `tools/riscv/` as local dependencies, or convert them to submodules later.
3. Initialize Git and create the first commit.

Typical commands:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin git@github.com:<username>/<repo>.git
git push -u origin main
```

## Notes

- `.venv/`, dataset files, model files, and build outputs are intentionally excluded from version control.
- `third_party/` and `tools/riscv/` are also excluded because they are large local dependencies.
- Some paths in the build flow are currently macOS/Homebrew-oriented and may need adjustment on another machine.
