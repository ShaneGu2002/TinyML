import hashlib
import json
import math
import random
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import tensorflow as tf


BACKGROUND_NOISE_LABEL = "_background_noise_"
SILENCE_LABEL = "_silence_"
UNKNOWN_LABEL = "_unknown_"
MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1


def which_set(filename: str, val_percentage: float, test_percentage: float) -> str:
    base = Path(filename).name
    hash_name = re.sub(r"_nohash_.*$", "", base)
    digest = hashlib.sha1(hash_name.encode("utf-8")).hexdigest()
    bucket = (int(digest, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (
        100.0 / MAX_NUM_WAVS_PER_CLASS
    )
    if bucket < val_percentage:
        return "val"
    if bucket < val_percentage + test_percentage:
        return "test"
    return "train"


@dataclass
class DatasetConfig:
    data_dir: Path
    keywords: Sequence[str]
    seed: int = 42
    sample_rate: int = 16000
    clip_duration_ms: int = 1000
    window_size_ms: float = 40.0
    window_stride_ms: float = 20.0
    feature_bins: int = 40
    mel_bins: int = 40
    val_percentage: float = 10.0
    test_percentage: float = 10.0
    lower_frequency: float = 20.0
    upper_frequency: float = 4000.0
    batch_size: int = 64
    include_unknown: bool = True
    include_silence: bool = True
    unknown_ratio: float = 1.0
    silence_ratio: float = 0.2
    background_noise_prob: float = 0.8
    background_noise_volume: float = 0.1
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    cache_dir: Optional[Path] = None
    num_parallel_calls: int = tf.data.AUTOTUNE
    prefetch_buffer: int = tf.data.AUTOTUNE

    def cache_key(self) -> str:
        key = {
            "sample_rate": self.sample_rate,
            "clip_duration_ms": self.clip_duration_ms,
            "window_size_ms": self.window_size_ms,
            "window_stride_ms": self.window_stride_ms,
            "feature_bins": self.feature_bins,
            "mel_bins": self.mel_bins,
            "lower_frequency": self.lower_frequency,
            "upper_frequency": self.upper_frequency,
            "keywords": sorted(self.keywords),
            "include_unknown": self.include_unknown,
            "include_silence": self.include_silence,
            "unknown_ratio": self.unknown_ratio,
            "silence_ratio": self.silence_ratio,
            "val_percentage": self.val_percentage,
            "test_percentage": self.test_percentage,
            "seed": self.seed,
            "max_train_samples": self.max_train_samples,
            "max_val_samples": self.max_val_samples,
            "max_test_samples": self.max_test_samples,
        }
        digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()
        return digest[:16]

    @property
    def desired_samples(self) -> int:
        return self.sample_rate * self.clip_duration_ms // 1000

    @property
    def window_size_samples(self) -> int:
        return int(self.sample_rate * self.window_size_ms / 1000)

    @property
    def window_stride_samples(self) -> int:
        return int(self.sample_rate * self.window_stride_ms / 1000)

    @property
    def labels(self) -> List[str]:
        labels = list(self.keywords)
        if self.include_unknown:
            labels.append(UNKNOWN_LABEL)
        if self.include_silence:
            labels.append(SILENCE_LABEL)
        return labels


def _limit_samples(items: List[Tuple[str, str, int]], max_samples: Optional[int], seed: int) -> List[Tuple[str, str, int]]:
    if max_samples is None or len(items) <= max_samples:
        return items
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    chosen = sorted(indices[:max_samples])
    return [items[i] for i in chosen]


def _load_background_segments(noise_dir: Path, clip_samples: int, seed: int) -> List[Tuple[str, int]]:
    segments: List[Tuple[str, int]] = []
    if not noise_dir.exists():
        return segments

    rng = random.Random(seed)
    for wav_path in sorted(noise_dir.glob("*.wav")):
        with wave.open(str(wav_path), "rb") as wav_file:
            total_frames = wav_file.getnframes()
        if total_frames <= clip_samples:
            segments.append((str(wav_path), 0))
            continue

        num_segments = max(1, total_frames // clip_samples)
        max_offset = total_frames - clip_samples
        for segment_idx in range(num_segments):
            if num_segments == 1:
                offset = 0
            else:
                offset = min(max_offset, segment_idx * clip_samples)
            jitter = 0 if max_offset == 0 else rng.randint(0, min(clip_samples // 4, max_offset - offset))
            segments.append((str(wav_path), offset + jitter))
    return segments


def build_file_index(config: DatasetConfig) -> Dict[str, List[Tuple[str, str, int]]]:
    data_dir = config.data_dir

    keyword_examples: List[Tuple[str, str, int, str]] = []
    unknown_examples: List[Tuple[str, str, int, str]] = []

    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        if label in {BACKGROUND_NOISE_LABEL, data_dir.name}:
            continue

        target_list = keyword_examples if label in config.keywords else unknown_examples
        for wav_path in sorted(label_dir.glob("*.wav")):
            split = which_set(wav_path.name, config.val_percentage, config.test_percentage)
            target_list.append((str(wav_path), label, 0, split))

    splits: Dict[str, List[Tuple[str, str, int]]] = {"train": [], "val": [], "test": []}
    for path, label, offset, split in keyword_examples:
        splits[split].append((path, label, offset))

    if config.include_unknown:
        grouped_unknown: Dict[str, List[Tuple[str, str, int]]] = {"train": [], "val": [], "test": []}
        for path, _label, offset, split in unknown_examples:
            grouped_unknown[split].append((path, UNKNOWN_LABEL, offset))

        rng = random.Random(config.seed)
        for split, items in grouped_unknown.items():
            keyword_count = sum(1 for _, label, _ in splits[split] if label in config.keywords)
            sample_count = min(len(items), int(keyword_count * config.unknown_ratio))
            rng.shuffle(items)
            splits[split].extend(items[:sample_count])

    if config.include_silence:
        noise_segments = _load_background_segments(
            data_dir / BACKGROUND_NOISE_LABEL,
            clip_samples=config.desired_samples,
            seed=config.seed,
        )
        for split in ("train", "val", "test"):
            keyword_count = sum(1 for _, label, _ in splits[split] if label in config.keywords)
            sample_count = min(len(noise_segments), max(1, int(keyword_count * config.silence_ratio)))
            rng = random.Random(config.seed + hash(split) % 997)
            sampled = noise_segments[:]
            rng.shuffle(sampled)
            splits[split].extend((path, SILENCE_LABEL, offset) for path, offset in sampled[:sample_count])

    for split in splits:
        rng = random.Random(config.seed + hash(split) % 1009)
        rng.shuffle(splits[split])

    splits["train"] = _limit_samples(splits["train"], config.max_train_samples, config.seed)
    splits["val"] = _limit_samples(splits["val"], config.max_val_samples, config.seed + 1)
    splits["test"] = _limit_samples(splits["test"], config.max_test_samples, config.seed + 2)
    return splits


def _decode_audio(audio_binary: tf.Tensor) -> tf.Tensor:
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
    return tf.squeeze(audio, axis=-1)


def _slice_or_pad(audio: tf.Tensor, desired_samples: int, offset: tf.Tensor) -> tf.Tensor:
    audio = audio[offset:]
    audio = audio[:desired_samples]
    pad_amount = tf.maximum(0, desired_samples - tf.shape(audio)[0])
    audio = tf.pad(audio, paddings=[[0, pad_amount]])
    audio.set_shape([desired_samples])
    return audio


def _compute_mfcc(config: DatasetConfig, audio: tf.Tensor) -> tf.Tensor:
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


def _load_background_noise_paths(config: DatasetConfig) -> List[str]:
    noise_paths: List[str] = []
    noise_dir = config.data_dir / BACKGROUND_NOISE_LABEL
    if not noise_dir.exists():
        return noise_paths
    for wav_path in sorted(noise_dir.glob("*.wav")):
        noise_paths.append(str(wav_path))
    return noise_paths


def _build_noise_dataset(config: DatasetConfig) -> Optional[tf.data.Dataset]:
    if config.background_noise_prob <= 0:
        return None
    noise_paths = _load_background_noise_paths(config)
    if not noise_paths:
        return None
    return tf.data.Dataset.from_tensor_slices(noise_paths).repeat()


def _add_background_noise(config: DatasetConfig, audio: tf.Tensor, label: tf.Tensor, noise: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    if tf.random.uniform([]) >= config.background_noise_prob:
        return audio, label

    noise = tf.reshape(noise, [-1])
    desired_samples = config.desired_samples
    max_offset = tf.maximum(0, tf.shape(noise)[0] - desired_samples)
    offset = tf.cond(
        max_offset > 0,
        lambda: tf.random.uniform([], 0, max_offset + 1, dtype=tf.int32),
        lambda: tf.constant(0, dtype=tf.int32),
    )
    noise = _slice_or_pad(noise, desired_samples, offset)
    mixed = audio + noise * config.background_noise_volume
    mixed = tf.clip_by_value(mixed, -1.0, 1.0)
    return mixed, label


def _make_example_dataset(
    examples: Sequence[Tuple[str, str, int]],
    config: DatasetConfig,
    training: bool,
    split_name: str,
) -> tf.data.Dataset:
    label_to_index = {label: idx for idx, label in enumerate(config.labels)}
    paths = [path for path, _, _ in examples]
    labels = [label_to_index[label] for _, label, _ in examples]
    offsets = [offset for _, _, offset in examples]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, offsets))

    def load_example(path: tf.Tensor, label: tf.Tensor, offset: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        audio = _decode_audio(tf.io.read_file(path))
        audio = _slice_or_pad(audio, config.desired_samples, offset)
        features = _compute_mfcc(config, audio)
        return features, label

    dataset = dataset.map(load_example, num_parallel_calls=config.num_parallel_calls)

    if config.cache_dir is not None:
        cache_subdir = Path(config.cache_dir) / config.cache_key()
        cache_subdir.mkdir(parents=True, exist_ok=True)
        (cache_subdir / "config.json").write_text(
            json.dumps(
                {
                    "feature_bins": config.feature_bins,
                    "window_size_ms": config.window_size_ms,
                    "window_stride_ms": config.window_stride_ms,
                    "keywords": list(config.keywords),
                },
                indent=2,
            )
        )
        dataset = dataset.cache(str(cache_subdir / split_name))
    elif not training:
        dataset = dataset.cache()

    if training:
        dataset = dataset.shuffle(len(paths), seed=config.seed, reshuffle_each_iteration=True)
        noise_dataset = _build_noise_dataset(config)
        if noise_dataset is not None:
            dataset = tf.data.Dataset.zip((dataset, noise_dataset))

            def mix_noise(example: Tuple[tf.Tensor, tf.Tensor], noise_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
                features, label = example
                noise = _decode_audio(tf.io.read_file(noise_path))
                noise = tf.image.resize(noise[tf.newaxis, :, tf.newaxis], [tf.shape(features)[0], 1])[0]
                noise = tf.expand_dims(noise, axis=1)
                noise = tf.tile(noise, [1, tf.shape(features)[1], 1])
                should_mix = tf.random.uniform([]) < config.background_noise_prob
                mixed = tf.cond(
                    should_mix,
                    lambda: tf.clip_by_value(features + noise * config.background_noise_volume, -10.0, 10.0),
                    lambda: features,
                )
                return mixed, label

            dataset = dataset.map(mix_noise, num_parallel_calls=config.num_parallel_calls)

    dataset = dataset.batch(config.batch_size).prefetch(config.prefetch_buffer)
    return dataset


def build_datasets(config: DatasetConfig) -> Tuple[Dict[str, tf.data.Dataset], Dict[str, List[Tuple[str, str, int]]], Tuple[int, int, int]]:
    splits = build_file_index(config)
    datasets = {
        "train": _make_example_dataset(splits["train"], config, training=True, split_name="train"),
        "val": _make_example_dataset(splits["val"], config, training=False, split_name="val"),
        "test": _make_example_dataset(splits["test"], config, training=False, split_name="test"),
    }

    feature_frames = 1 + max(0, math.floor((config.desired_samples - config.window_size_samples) / config.window_stride_samples))
    feature_shape = (feature_frames, config.feature_bins, 1)
    return datasets, splits, feature_shape
