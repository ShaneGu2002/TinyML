import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from kws.data import DatasetConfig, build_datasets
from kws.models import build_model, estimate_operations, estimate_peak_activation_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TensorFlow keyword spotting models.")
    parser.add_argument("--data-dir", type=Path, default=Path("speech-commands-v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--model",
        choices=["cnn_trad_fpool3", "cnn_one_fstride4", "ds_cnn", "gru"],
        default="ds_cnn",
    )
    parser.add_argument(
        "--gru-units",
        type=int,
        default=64,
        help="Hidden units for the GRU model (only used when --model gru).",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"],
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--no-unknown", action="store_true")
    parser.add_argument("--no-silence", action="store_true")
    parser.add_argument("--export-int8", action="store_true")
    parser.add_argument("--representative-samples", type=int, default=200)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for caching preprocessed MFCC features. Huge speedup on repeated training.",
    )
    return parser.parse_args()


def representative_dataset(dataset: tf.data.Dataset, limit: int):
    count = 0
    for features, _labels in dataset.unbatch().batch(1):
        yield [tf.cast(features, tf.float32)]
        count += 1
        if count >= limit:
            break


def evaluate_tflite(tflite_path: Path, dataset: tf.data.Dataset) -> float:
    """Run the TFLite model sample-by-sample and return top-1 accuracy.

    Handles both pure-INT8 inputs and float32 inputs (used when sequence ops
    fall back to float in the converter).
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_scale, in_zp = in_det["quantization"]
    in_dtype = in_det["dtype"]

    correct, total = 0, 0
    for features, labels in dataset.unbatch().batch(1):
        x = features.numpy().astype(np.float32)
        if np.issubdtype(in_dtype, np.integer):
            if in_scale > 0:
                x = np.round(x / in_scale + in_zp)
            x = np.clip(x, np.iinfo(in_dtype).min, np.iinfo(in_dtype).max).astype(in_dtype)
        else:
            x = x.astype(in_dtype)
        interpreter.set_tensor(in_det["index"], x)
        interpreter.invoke()
        pred = interpreter.get_tensor(out_det["index"])
        correct += int(np.argmax(pred) == int(labels.numpy()[0]))
        total += 1
    return correct / max(total, 1)


def export_int8_tflite(
    model: tf.keras.Model,
    validation_dataset: tf.data.Dataset,
    output_path: Path,
    rep_samples: int,
    allow_float_fallback: bool = False,
) -> Path:
    input_shape = (1,) + tuple(model.input_shape[1:])
    concrete_func = tf.function(model).get_concrete_function(tf.TensorSpec(input_shape, tf.float32))
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_func])
    converter.experimental_enable_resource_variables = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(validation_dataset, rep_samples)
    if allow_float_fallback:
        # Some sequence ops (e.g. GRU/LSTM) lack a pure-INT8 kernel; allow the converter
        # to keep them in float while still quantizing every other op.
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    else:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    return output_path


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    config = DatasetConfig(
        data_dir=args.data_dir,
        keywords=args.keywords,
        seed=args.seed,
        batch_size=args.batch_size,
        include_unknown=not args.no_unknown,
        include_silence=not args.no_silence,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        cache_dir=args.cache_dir,
    )

    datasets, splits, input_shape = build_datasets(config)
    model_kwargs = {}
    if args.model == "gru":
        model_kwargs["gru_units"] = args.gru_units
    model = build_model(
        args.model,
        input_shape=input_shape,
        num_classes=len(config.labels),
        **model_kwargs,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "history.csv")),
    ]

    model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("Evaluating FP32 model on train/val/test …")
    fp32_train = model.evaluate(datasets["train"], return_dict=True, verbose=0)
    fp32_val = model.evaluate(datasets["val"], return_dict=True, verbose=0)
    fp32_test = model.evaluate(datasets["test"], return_dict=True, verbose=0)

    model.save(output_dir / "final.keras")

    total_samples = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    macs = estimate_operations(model)
    int8_weights_bytes = model.count_params()
    int8_peak_activation_bytes = estimate_peak_activation_bytes(model, bytes_per_element=1)

    report = {
        "model": args.model,
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
    }

    final_model_path = output_dir / "final.keras"
    report["keras_model_size_bytes"] = final_model_path.stat().st_size

    if args.export_int8:
        tflite_path = export_int8_tflite(
            model=model,
            validation_dataset=datasets["val"],
            output_path=output_dir / "model_int8.tflite",
            rep_samples=args.representative_samples,
            allow_float_fallback=(args.model == "gru"),
        )
        report["int8_tflite_size_bytes"] = tflite_path.stat().st_size
        print("Evaluating int8 TFLite on train/val/test …")
        report["int8_train_accuracy"] = evaluate_tflite(tflite_path, datasets["train"])
        report["int8_val_accuracy"] = evaluate_tflite(tflite_path, datasets["val"])
        report["int8_test_accuracy"] = evaluate_tflite(tflite_path, datasets["test"])

    with (output_dir / "report.json").open("w") as report_file:
        json.dump(report, report_file, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
