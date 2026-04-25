"""Finalize a partially-trained GRU run.

Loads `best.keras` from a previous training, runs INT8 quantization +
evaluation, and writes the same `report.json` schema as `train_kws.py`.
This is useful when training was killed early (e.g. thermal throttling)
but we already have a usable checkpoint.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from kws.data import DatasetConfig, build_datasets
from kws.models import (
    SharedGRUUnroll,
    estimate_operations,
    estimate_peak_activation_bytes,
)
from train_kws import evaluate_tflite, export_int8_tflite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("speech-commands-v2"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".mfcc_cache"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--representative-samples", type=int, default=200)
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatasetConfig(
        data_dir=args.data_dir,
        keywords=args.keywords,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )
    datasets, splits, input_shape = build_datasets(config)

    model = tf.keras.models.load_model(
        args.checkpoint,
        custom_objects={"SharedGRUUnroll": SharedGRUUnroll},
        compile=False,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print("Evaluating FP32 …")
    fp32_train = model.evaluate(datasets["train"], return_dict=True, verbose=0)
    fp32_val = model.evaluate(datasets["val"], return_dict=True, verbose=0)
    fp32_test = model.evaluate(datasets["test"], return_dict=True, verbose=0)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "final.keras"
    model.save(final_path)

    macs = estimate_operations(model)
    int8_weights_bytes = model.count_params()
    int8_peak_activation_bytes = estimate_peak_activation_bytes(model, bytes_per_element=1)
    total_samples = len(splits["train"]) + len(splits["val"]) + len(splits["test"])

    report = {
        "model": "gru",
        "labels": config.labels,
        "input_shape": input_shape,
        "num_train_samples": len(splits["train"]),
        "num_val_samples": len(splits["val"]),
        "num_test_samples": len(splits["test"]),
        "split_ratio": {
            "train": len(splits["train"]) / total_samples,
            "val": len(splits["val"]) / total_samples,
            "test": len(splits["test"]) / total_samples,
        },
        "parameter_count": model.count_params(),
        "macs_per_inference": macs,
        "ops_per_inference": 2 * macs,
        "estimated_peak_activation_bytes_fp32": estimate_peak_activation_bytes(model),
        "int8_weights_bytes": int8_weights_bytes,
        "int8_peak_activation_bytes": int8_peak_activation_bytes,
        "int8_total_memory_bytes": int8_weights_bytes + int8_peak_activation_bytes,
        "fp32_train_accuracy": fp32_train["accuracy"],
        "fp32_val_accuracy": fp32_val["accuracy"],
        "fp32_test_accuracy": fp32_test["accuracy"],
        "fp32_test_loss": fp32_test["loss"],
        "keras_model_size_bytes": final_path.stat().st_size,
        "note": "Finalized from early-stopped checkpoint (training was throttled).",
    }

    print("Exporting INT8 TFLite …")
    tflite_path = export_int8_tflite(
        model=model,
        validation_dataset=datasets["val"],
        output_path=output_dir / "model_int8.tflite",
        rep_samples=args.representative_samples,
    )
    report["int8_tflite_size_bytes"] = tflite_path.stat().st_size

    print("Evaluating INT8 TFLite …")
    report["int8_train_accuracy"] = evaluate_tflite(tflite_path, datasets["train"])
    report["int8_val_accuracy"] = evaluate_tflite(tflite_path, datasets["val"])
    report["int8_test_accuracy"] = evaluate_tflite(tflite_path, datasets["test"])

    with (output_dir / "report.json").open("w") as fp:
        json.dump(report, fp, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
