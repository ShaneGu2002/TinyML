"""Run the DNN sweep + retrain pipeline on Modal cloud GPUs.

Usage from local Mobaxterm shell (after ``pip install modal && modal setup``)::

    # One-time: download Speech Commands v0.02 to a persistent Modal Volume
    modal run run_modal.py --step setup

    # Run the S/M/L sweep (~1-2h on T4)
    modal run run_modal.py --step sweep

    # Retrain the per-tier winners with full noise aug (~30min on T4)
    modal run run_modal.py --step retrain

    # Or chain everything
    modal run run_modal.py --step all

    # Pull the produced artifacts back to your local machine
    modal volume get kws-artifacts retrained ./artifacts_from_modal
    modal volume get kws-artifacts sweep_artifacts ./sweep_artifacts_from_modal

State is persisted across runs in three Modal Volumes:
    kws-dataset    - speech-commands-v2/
    kws-cache      - MFCC feature cache
    kws-artifacts  - sweep_artifacts/ + retrained/
"""

import modal


APP_NAME = "kws-dnn-sweep"

# Image: NVIDIA CUDA 12.5 base (provides libcuda.so via Modal's NVIDIA container
# runtime hook) + Python 3.11 + TensorFlow with bundled CUDA libs (cuDNN, cuBLAS,
# etc. matched to TF 2.21) + the repo source.
#
# debian_slim does NOT inherit NVIDIA's container hooks, so libcuda.so is missing
# even when gpu="H100" is set. nvidia/cuda:*-runtime ensures that hook is wired up.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.5.1-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("tensorflow[and-cuda]==2.21.0", "numpy")
    .add_local_dir(".", "/workspace")
)

dataset_volume = modal.Volume.from_name("kws-dataset", create_if_missing=True)
cache_volume = modal.Volume.from_name("kws-cache", create_if_missing=True)
artifacts_volume = modal.Volume.from_name("kws-artifacts", create_if_missing=True)

VOLUMES = {
    "/data": dataset_volume,
    "/cache": cache_volume,
    "/artifacts": artifacts_volume,
}

DATASET_DIR = "/data/speech-commands-v2"
CACHE_DIR = "/cache/mfcc"
SWEEP_DIR = "/artifacts/sweep_artifacts"
RETRAIN_DIR = "/artifacts/retrained"

GPU = "H100"  # high availability, ~3x faster than A10. Use "B200" only if pool isn't queued.

app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# One-time setup: download Speech Commands v0.02 into the dataset volume.
# ---------------------------------------------------------------------------
@app.function(image=image, volumes=VOLUMES, timeout=1800)
def setup_dataset() -> None:
    import os
    import subprocess
    import urllib.request

    if os.path.exists(os.path.join(DATASET_DIR, "yes")):
        sample_count = len(os.listdir(os.path.join(DATASET_DIR, "yes")))
        print(f"Dataset already present ({sample_count} samples in yes/), skipping.")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)
    url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    tar_path = "/tmp/sc_v2.tar.gz"
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, tar_path)
    print("Extracting ...")
    subprocess.run(["tar", "xzf", tar_path, "-C", DATASET_DIR], check=True)
    os.remove(tar_path)
    dataset_volume.commit()
    print("Dataset ready at", DATASET_DIR)


# ---------------------------------------------------------------------------
# Stage 1+2 sweep across S/M/L tiers.
# ---------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=4 * 3600)
def run_sweep() -> None:
    import subprocess

    cmd = [
        "python", "/workspace/sweep_dnn.py",
        "--data-dir", DATASET_DIR,
        "--cache-dir", CACHE_DIR,
        "--output-dir", SWEEP_DIR,
        "--batch-size", "512",
        "--no-noise-aug",
    ]
    print("> " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd="/workspace")
    cache_volume.commit()
    artifacts_volume.commit()


# ---------------------------------------------------------------------------
# Retrain per-tier winners with full noise augmentation and 30 epochs.
# ---------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=2 * 3600)
def retrain_tier(tier: str) -> None:
    import subprocess

    sweep_report = f"{SWEEP_DIR}/dnn_{tier}/report.json"
    cmd = [
        "python", "/workspace/train_kws.py",
        "--retrain-from-sweep", sweep_report,
        "--data-dir", DATASET_DIR,
        "--cache-dir", CACHE_DIR,
        "--output-dir", RETRAIN_DIR,
        "--epochs", "30",
        "--export-int8",
        "--int8-eval-test-only",  # skip int8 train/val eval (slow, not needed for top table)
    ]
    print(f"=== Retrain DNN tier {tier} ===")
    print("> " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd="/workspace")
    artifacts_volume.commit()


# ---------------------------------------------------------------------------
# CLI entrypoint.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(step: str = "all") -> None:
    """step: setup | sweep | retrain | all (or comma-separated list)."""
    steps = [s.strip() for s in step.split(",")] if step != "all" else ["setup", "sweep", "retrain"]

    if "setup" in steps:
        print("=== Step: setup ===")
        setup_dataset.remote()

    if "sweep" in steps:
        print("=== Step: sweep ===")
        run_sweep.remote()

    if "retrain" in steps:
        print("=== Step: retrain ===")
        for tier in ("S", "M", "L"):
            retrain_tier.remote(tier)

    print("\nDone. To pull results locally:")
    print("  modal volume get kws-artifacts retrained ./artifacts_from_modal")
    print("  modal volume get kws-artifacts sweep_artifacts ./sweep_artifacts_from_modal")
