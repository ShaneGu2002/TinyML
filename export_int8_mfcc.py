import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from kws.data import DatasetConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a WAV file into the INT8 MFCC input expected by the DS-CNN TFLite model."
    )
    parser.add_argument("--wav", type=Path, required=True, help="Input 16 kHz mono WAV file.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/ds_cnn/model_int8.tflite"),
        help="INT8 TFLite model used to read input quantization parameters.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ds_cnn/input_int8_mfcc.npy"),
        help="Output file path.",
    )
    parser.add_argument(
        "--format",
        choices=["npy", "json", "c_array"],
        default="npy",
        help="Output serialization format.",
    )
    return parser.parse_args()


def decode_audio(wav_path: Path) -> tf.Tensor:
    audio_binary = tf.io.read_file(str(wav_path))
    audio, sample_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)

    if sample_rate.numpy() != 16000:
        raise ValueError(f"Expected 16 kHz WAV, got {sample_rate.numpy()} Hz.")
    return audio


def slice_or_pad(audio: tf.Tensor, desired_samples: int) -> tf.Tensor:
    audio = audio[:desired_samples]
    pad_amount = tf.maximum(0, desired_samples - tf.shape(audio)[0])
    audio = tf.pad(audio, paddings=[[0, pad_amount]])
    audio.set_shape([desired_samples])
    return audio


def compute_mfcc(config: DatasetConfig, audio: tf.Tensor) -> tf.Tensor:
    stft = tf.signal.stft(
        audio,
        frame_length=config.window_size_samples,
        frame_step=config.window_stride_samples,
        fft_length=config.window_size_samples,
    )
    magnitude = tf.abs(stft)
    num_spectrogram_bins = magnitude.shape[-1]
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=config.mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=config.sample_rate,
        lower_edge_hertz=config.lower_frequency,
        upper_edge_hertz=config.upper_frequency,
    )
    mel_spectrogram = tf.matmul(tf.square(magnitude), mel_weight_matrix)
    log_mel = tf.math.log(mel_spectrogram + 1e-6)
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., : config.feature_bins]
    mfcc = tf.expand_dims(mfcc, axis=-1)
    return mfcc


def load_quantization_params(model_path: Path) -> Tuple[float, int, List[int]]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    scale, zero_point = input_details["quantization"]
    input_shape = input_details["shape"].tolist()

    if input_details["dtype"] != np.int8:
        raise ValueError("Model input dtype is not int8.")
    if scale == 0:
        raise ValueError("Invalid quantization scale 0.")
    return float(scale), int(zero_point), input_shape


def quantize_features(features: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    quantized = np.round(features / scale + zero_point)
    quantized = np.clip(quantized, -128, 127).astype(np.int8)
    return quantized


def write_output(output_path: Path, array: np.ndarray, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "npy":
        np.save(output_path, array)
        return

    if output_format == "json":
        payload = {
            "shape": list(array.shape),
            "dtype": "int8",
            "data": array.reshape(-1).tolist(),
        }
        output_path.write_text(json.dumps(payload, indent=2))
        return

    if output_format == "c_array":
        flattened = array.reshape(-1)
        values = ", ".join(str(int(v)) for v in flattened)
        content = (
            "#include <stdint.h>\n\n"
            f"const int8_t g_example_mfcc[{flattened.size}] = {{\n  {values}\n}};\n"
        )
        output_path.write_text(content)
        return

    raise ValueError(f"Unsupported format: {output_format}")


def main() -> None:
    args = parse_args()
    config = DatasetConfig(data_dir=Path("speech-commands-v2"), keywords=["yes", "no", "stop", "go"])

    audio = decode_audio(args.wav)
    audio = slice_or_pad(audio, config.desired_samples)
    mfcc = compute_mfcc(config, audio).numpy().astype(np.float32)

    scale, zero_point, input_shape = load_quantization_params(args.model)
    expected_shape = [1, mfcc.shape[0], mfcc.shape[1], mfcc.shape[2]]
    if input_shape != expected_shape:
        raise ValueError(f"Model expects input shape {input_shape}, but computed shape is {expected_shape}.")

    quantized = quantize_features(mfcc, scale, zero_point)
    batched = np.expand_dims(quantized, axis=0)
    write_output(args.output, batched, args.format)

    print(
        json.dumps(
            {
                "wav": str(args.wav),
                "model": str(args.model),
                "output": str(args.output),
                "format": args.format,
                "shape": list(batched.shape),
                "dtype": "int8",
                "quantization_scale": scale,
                "quantization_zero_point": zero_point,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
