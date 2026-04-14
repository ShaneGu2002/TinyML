import argparse
import csv
import json
import re
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from export_int8_mfcc import compute_mfcc, load_quantization_params, quantize_features
from kws.data import DatasetConfig, build_file_index


RESULT_RE = re.compile(r"RESULT label=(?P<label>\S+) index=(?P<index>-?\d+) score=(?P<score>-?\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the full Speech Commands test split by invoking the TFLM ELF on Spike."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("speech-commands-v2"),
        help="Speech Commands V2 dataset directory.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/ds_cnn/model_int8.tflite"),
        help="INT8 TFLite model used to read input quantization parameters.",
    )
    parser.add_argument(
        "--elf",
        type=Path,
        default=Path("build/kws_demo.elf"),
        help="RISC-V ELF that accepts an external 490-byte INT8 MFCC input file.",
    )
    parser.add_argument(
        "--spike",
        type=str,
        default="spike",
        help="Spike executable.",
    )
    parser.add_argument(
        "--pk",
        type=Path,
        default=Path("/opt/homebrew/Cellar/riscv-pk/main/riscv64-unknown-elf/bin/pk"),
        help="Path to proxy kernel (pk).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally evaluate only the first N test samples.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start from this test-set index. Useful for resuming long runs.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N samples.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("artifacts/ds_cnn/spike_test_eval_summary.json"),
        help="Where to save aggregate metrics.",
    )
    parser.add_argument(
        "--samples-csv",
        type=Path,
        default=Path("artifacts/ds_cnn/spike_test_eval_samples.csv"),
        help="Where to save per-sample predictions.",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=["yes", "no", "stop", "go"],
        help="Target keywords. Must match the trained model.",
    )
    return parser.parse_args()


def decode_audio_with_sample_rate(wav_path: Path) -> Tuple[tf.Tensor, int]:
    audio_binary = tf.io.read_file(str(wav_path))
    audio, sample_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
    return tf.squeeze(audio, axis=-1), int(sample_rate.numpy())


def slice_or_pad(audio: tf.Tensor, desired_samples: int, offset: int) -> tf.Tensor:
    audio = audio[offset:]
    audio = audio[:desired_samples]
    pad_amount = tf.maximum(0, desired_samples - tf.shape(audio)[0])
    audio = tf.pad(audio, paddings=[[0, pad_amount]])
    audio.set_shape([desired_samples])
    return audio


def build_quantized_input(
    wav_path: Path,
    offset: int,
    config: DatasetConfig,
    scale: float,
    zero_point: int,
    input_shape: List[int],
) -> np.ndarray:
    audio, sample_rate = decode_audio_with_sample_rate(wav_path)
    if sample_rate != config.sample_rate:
        raise ValueError(f"Expected {config.sample_rate} Hz audio, got {sample_rate} Hz for {wav_path}.")

    mfcc = compute_mfcc(config, slice_or_pad(audio, config.desired_samples, offset)).numpy().astype(np.float32)
    expected_shape = [1, mfcc.shape[0], mfcc.shape[1], mfcc.shape[2]]
    if input_shape != expected_shape:
        raise ValueError(f"Model expects input shape {input_shape}, but computed shape is {expected_shape}.")

    quantized = quantize_features(mfcc, scale, zero_point)
    return np.expand_dims(quantized, axis=0)


def run_spike(spike: str, pk: Path, elf: Path, input_path: Path) -> Tuple[str, int, int, str]:
    command = [spike, str(pk), str(elf), str(input_path)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    output = (completed.stdout or "") + (completed.stderr or "")
    match = RESULT_RE.search(output)
    if match is None:
        raise RuntimeError(f"Failed to parse Spike output.\nCommand: {' '.join(command)}\nOutput:\n{output}")
    return (
        match.group("label"),
        int(match.group("index")),
        int(match.group("score")),
        output.strip(),
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    config = DatasetConfig(data_dir=args.data_dir, keywords=args.keywords)
    splits = build_file_index(config)
    test_examples = splits["test"]

    if args.start_index:
        test_examples = test_examples[args.start_index :]
    if args.limit is not None:
        test_examples = test_examples[: args.limit]

    if not test_examples:
        raise ValueError("No test examples selected. Check --start-index/--limit.")

    scale, zero_point, input_shape = load_quantization_params(args.model)
    label_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    confusion: Dict[str, Counter[str]] = {label: Counter() for label in config.labels}
    sample_rows: List[Dict[str, object]] = []

    ensure_parent(args.summary_json)
    ensure_parent(args.samples_csv)

    start_time = time.time()
    with tempfile.TemporaryDirectory(prefix="spike_eval_", dir=str(Path.cwd())) as temp_dir:
        temp_input = Path(temp_dir) / "input.bin"

        for sample_number, (path_str, true_label, offset) in enumerate(test_examples, start=1):
            wav_path = Path(path_str)
            quantized = build_quantized_input(wav_path, offset, config, scale, zero_point, input_shape)
            quantized.reshape(-1).tofile(temp_input)

            predicted_label, predicted_index, predicted_score, raw_output = run_spike(
                args.spike, args.pk, args.elf, temp_input
            )
            correct = predicted_label == true_label

            label_counts[true_label] += 1
            predicted_counts[predicted_label] += 1
            confusion.setdefault(true_label, Counter())[predicted_label] += 1
            sample_rows.append(
                {
                    "dataset_index": args.start_index + sample_number - 1,
                    "wav_path": str(wav_path),
                    "offset": offset,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "predicted_index": predicted_index,
                    "predicted_score": predicted_score,
                    "correct": int(correct),
                    "raw_output": raw_output,
                }
            )

            if sample_number % args.progress_every == 0 or sample_number == len(test_examples):
                correct_so_far = sum(int(row["correct"]) for row in sample_rows)
                elapsed = time.time() - start_time
                print(
                    json.dumps(
                        {
                            "evaluated": sample_number,
                            "total_selected": len(test_examples),
                            "running_accuracy": correct_so_far / sample_number,
                            "elapsed_seconds": round(elapsed, 2),
                        }
                    ),
                    flush=True,
                )

    total = len(sample_rows)
    num_correct = sum(int(row["correct"]) for row in sample_rows)
    summary = {
        "data_dir": str(args.data_dir),
        "model": str(args.model),
        "elf": str(args.elf),
        "spike": args.spike,
        "pk": str(args.pk),
        "keywords": list(args.keywords),
        "labels": list(config.labels),
        "selected_test_examples": total,
        "dataset_start_index": args.start_index,
        "dataset_end_index": args.start_index + total - 1,
        "accuracy": num_correct / total,
        "num_correct": num_correct,
        "num_incorrect": total - num_correct,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "label_counts": dict(label_counts),
        "predicted_counts": dict(predicted_counts),
        "confusion_matrix": {label: dict(confusion[label]) for label in config.labels},
    }

    args.summary_json.write_text(json.dumps(summary, indent=2))

    with args.samples_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "dataset_index",
                "wav_path",
                "offset",
                "true_label",
                "predicted_label",
                "predicted_index",
                "predicted_score",
                "correct",
                "raw_output",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
