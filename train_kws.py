import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from kws.data import DatasetConfig, build_datasets
from kws.models import build_model, estimate_peak_activation_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TensorFlow keyword spotting models.")
    parser.add_argument("--data-dir", type=Path, default=Path("speech-commands-v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model", choices=["baseline_cnn", "ds_cnn"], default="ds_cnn")
    parser.add_argument("--keywords", nargs="+", default=["yes", "no", "stop", "go"])
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
    return parser.parse_args()


def representative_dataset(dataset: tf.data.Dataset, limit: int):
    count = 0
    for features, _labels in dataset.unbatch().batch(1):
        yield [tf.cast(features, tf.float32)]
        count += 1
        if count >= limit:
            break


def export_int8_tflite(
    model: tf.keras.Model,
    validation_dataset: tf.data.Dataset,
    output_path: Path,
    rep_samples: int,
) -> Path:
    input_shape = (1,) + tuple(model.input_shape[1:])
    concrete_func = tf.function(model).get_concrete_function(tf.TensorSpec(input_shape, tf.float32))
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(validation_dataset, rep_samples)
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
    )

    datasets, splits, input_shape = build_datasets(config)
    model = build_model(args.model, input_shape=input_shape, num_classes=len(config.labels))
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

    history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=args.epochs,
        callbacks=callbacks,
    )
    test_metrics = model.evaluate(datasets["test"], return_dict=True)

    model.save(output_dir / "final.keras")

    report = {
        "model": args.model,
        "labels": config.labels,
        "input_shape": input_shape,
        "num_train_samples": len(splits["train"]),
        "num_val_samples": len(splits["val"]),
        "num_test_samples": len(splits["test"]),
        "parameter_count": model.count_params(),
        "estimated_peak_activation_bytes_fp32": estimate_peak_activation_bytes(model),
        "final_train_accuracy": history.history["accuracy"][-1],
        "best_val_accuracy": max(history.history["val_accuracy"]),
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
    }

    final_model_path = output_dir / "final.keras"
    report["keras_model_size_bytes"] = final_model_path.stat().st_size

    if args.export_int8:
        tflite_path = export_int8_tflite(
            model=model,
            validation_dataset=datasets["val"],
            output_path=output_dir / "model_int8.tflite",
            rep_samples=args.representative_samples,
        )
        report["int8_tflite_size_bytes"] = tflite_path.stat().st_size

    with (output_dir / "report.json").open("w") as report_file:
        json.dump(report, report_file, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
