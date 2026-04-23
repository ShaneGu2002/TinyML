"""Hello-Edge style two-stage hyperparameter sweep for DS-CNN models.

Stage 1: per budget tier, sample random configs from the search space,
analytically reject those that exceed the int8 memory / op budget, and
train each survivor for a few epochs.

Stage 2: take the top-K of stage 1 per tier and train them for more epochs
with early stopping. The best validation accuracy per tier is kept as the
final ds_cnn_S / ds_cnn_M / ds_cnn_L model.

Outputs land under ``--output-dir`` (default ``sweep_artifacts/``)::

    sweep_artifacts/
    ├── sweep_results.csv           # every candidate, both stages
    ├── ds_cnn_S/{best.keras, report.json}
    ├── ds_cnn_M/{best.keras, report.json}
    └── ds_cnn_L/{best.keras, report.json}
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tensorflow as tf

from kws.data import DatasetConfig, build_datasets
from kws.models import build_ds_cnn, estimate_operations, estimate_peak_activation_bytes


KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

# (memory_bytes, ops) — int8 weights+peak-activation memory, ops = 2 * MACs
BUDGETS: Dict[str, Dict[str, int]] = {
    "S": {"memory_bytes": 80 * 1024,  "ops":  6_000_000},
    "M": {"memory_bytes": 200 * 1024, "ops": 20_000_000},
    "L": {"memory_bytes": 500 * 1024, "ops": 80_000_000},
}

SEARCH_SPACE = {
    "layers":       [4, 5, 6],
    "filters":      [64, 76, 96, 128, 172, 276],
    "kernel":       [(3, 3), (5, 3), (10, 4), (20, 8)],
    "first_stride": [(1, 1), (2, 1), (1, 2), (2, 2)],
    "mfcc_bins":    [10, 40],
}

CSV_COLUMNS = [
    "tier", "stage", "rank",
    "layers", "filters",
    "kernel_t", "kernel_f", "stride_t", "stride_f", "mfcc_bins",
    "params", "weight_bytes", "act_bytes", "memory_bytes",
    "macs", "ops", "epochs", "val_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("speech-commands-v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("sweep_artifacts"))
    parser.add_argument("--cache-dir", type=Path, default=Path("sweep_cache"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates-per-tier", type=int, default=30,
                        help="Target number of valid (in-budget) candidates per tier.")
    parser.add_argument("--max-sample-attempts", type=int, default=2000,
                        help="Maximum random draws per tier before giving up.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=15)
    parser.add_argument("--stage2-patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--tiers", nargs="+", choices=list(BUDGETS), default=list(BUDGETS),
                        help="Restrict the sweep to a subset of tiers.")
    parser.add_argument("--no-noise-aug", action="store_true",
                        help="Disable background-noise augmentation. Removes the per-batch "
                             "wav read from the data pipeline -> 3-5x faster epochs. "
                             "Recommended for the sweep itself; re-enable when retraining the "
                             "final S/M/L model for production accuracy.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only sample + analytically score candidates, skip training.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Search-space sampling and analytic costing
# ---------------------------------------------------------------------------


def feature_frame_count(sample_rate: int = 16000, clip_ms: int = 1000,
                        win_ms: int = 40, stride_ms: int = 20) -> int:
    desired = sample_rate * clip_ms // 1000
    win = int(sample_rate * win_ms / 1000)
    stride = int(sample_rate * stride_ms / 1000)
    return 1 + max(0, math.floor((desired - win) / stride))


def sample_config(rng: random.Random) -> Dict:
    return {
        "layers":       rng.choice(SEARCH_SPACE["layers"]),
        "filters":      rng.choice(SEARCH_SPACE["filters"]),
        "kernel":       tuple(rng.choice(SEARCH_SPACE["kernel"])),
        "first_stride": tuple(rng.choice(SEARCH_SPACE["first_stride"])),
        "mfcc_bins":    rng.choice(SEARCH_SPACE["mfcc_bins"]),
    }


def config_key(cfg: Dict) -> Tuple:
    return (cfg["layers"], cfg["filters"], cfg["kernel"], cfg["first_stride"], cfg["mfcc_bins"])


def build_candidate_model(cfg: Dict, num_classes: int) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    input_shape = (feature_frame_count(), cfg["mfcc_bins"], 1)
    return build_ds_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
        layers=cfg["layers"],
        filters=cfg["filters"],
        kernel=cfg["kernel"],
        first_stride=cfg["first_stride"],
    )


def analytic_cost(cfg: Dict, num_classes: int) -> Dict[str, int]:
    model = build_candidate_model(cfg, num_classes)
    weights = int(model.count_params())
    act_bytes = int(estimate_peak_activation_bytes(model, bytes_per_element=1))
    macs = int(estimate_operations(model))
    return {
        "params":       weights,
        "weight_bytes": weights,
        "act_bytes":    act_bytes,
        "memory_bytes": weights + act_bytes,
        "macs":         macs,
        "ops":          2 * macs,
    }


def fits_budget(cost: Dict[str, int], budget: Dict[str, int]) -> bool:
    return cost["memory_bytes"] <= budget["memory_bytes"] and cost["ops"] <= budget["ops"]


def sample_valid_candidates(budget: Dict[str, int], target: int, num_classes: int,
                            seed: int, max_attempts: int) -> List[Tuple[Dict, Dict[str, int]]]:
    rng = random.Random(seed)
    seen = set()
    valid: List[Tuple[Dict, Dict[str, int]]] = []
    attempts = 0
    while len(valid) < target and attempts < max_attempts:
        attempts += 1
        cfg = sample_config(rng)
        key = config_key(cfg)
        if key in seen:
            continue
        seen.add(key)
        try:
            cost = analytic_cost(cfg, num_classes)
        except Exception as exc:
            print(f"  skip {cfg}: build failed ({exc})")
            continue
        if fits_budget(cost, budget):
            valid.append((cfg, cost))
    if len(valid) < target:
        print(f"  warning: only found {len(valid)}/{target} valid configs in {attempts} attempts")
    return valid


# ---------------------------------------------------------------------------
# Dataset cache (one tf.data pipeline per MFCC-bin setting)
# ---------------------------------------------------------------------------


_DATASETS_CACHE: Dict[int, Tuple[Dict[str, tf.data.Dataset], Dict, Tuple[int, int, int], DatasetConfig]] = {}


def get_datasets(args: argparse.Namespace, mfcc_bins: int):
    if mfcc_bins in _DATASETS_CACHE:
        return _DATASETS_CACHE[mfcc_bins]
    config = DatasetConfig(
        data_dir=args.data_dir,
        keywords=KEYWORDS,
        seed=args.seed,
        batch_size=args.batch_size,
        feature_bins=mfcc_bins,
        mel_bins=mfcc_bins,
        cache_dir=args.cache_dir,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        background_noise_prob=0.0 if args.no_noise_aug else 0.8,
    )
    datasets, splits, input_shape = build_datasets(config)
    _DATASETS_CACHE[mfcc_bins] = (datasets, splits, input_shape, config)
    print(f"  built dataset (mfcc_bins={mfcc_bins}): "
          f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    return _DATASETS_CACHE[mfcc_bins]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def train_candidate(cfg: Dict, num_classes: int, datasets: Dict[str, tf.data.Dataset],
                    epochs: int, learning_rate: float,
                    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
                    verbose: int = 2) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    model = build_candidate_model(cfg, num_classes)
    compile_model(model, learning_rate)
    history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=epochs,
        callbacks=callbacks or [],
        verbose=verbose,
    )
    return model, history


def append_csv_row(csv_path: Path, row: Dict) -> None:
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


def make_csv_row(tier: str, stage: int, rank: int, cfg: Dict, cost: Dict[str, int],
                 epochs: int, val_accuracy: Optional[float]) -> Dict:
    return {
        "tier": tier, "stage": stage, "rank": rank,
        "layers": cfg["layers"], "filters": cfg["filters"],
        "kernel_t": cfg["kernel"][0], "kernel_f": cfg["kernel"][1],
        "stride_t": cfg["first_stride"][0], "stride_f": cfg["first_stride"][1],
        "mfcc_bins": cfg["mfcc_bins"],
        "params": cost["params"], "weight_bytes": cost["weight_bytes"],
        "act_bytes": cost["act_bytes"], "memory_bytes": cost["memory_bytes"],
        "macs": cost["macs"], "ops": cost["ops"],
        "epochs": epochs,
        "val_accuracy": "" if val_accuracy is None else f"{val_accuracy:.6f}",
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "sweep_results.csv"
    with csv_path.open("w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()

    # Probe dataset to learn label count (10 keywords + unknown + silence by default)
    probe = DatasetConfig(
        data_dir=args.data_dir, keywords=KEYWORDS, seed=args.seed,
        feature_bins=SEARCH_SPACE["mfcc_bins"][0], mel_bins=SEARCH_SPACE["mfcc_bins"][0],
    )
    num_classes = len(probe.labels)
    print(f"num_classes = {num_classes} (labels = {probe.labels})")

    summary: Dict[str, Dict] = {}

    for tier_idx, tier in enumerate(args.tiers):
        budget = BUDGETS[tier]
        print(f"\n=== Tier {tier}: <= {budget['memory_bytes']//1024} KB, "
              f"<= {budget['ops']/1e6:.0f} MOps ===")

        candidates = sample_valid_candidates(
            budget=budget,
            target=args.candidates_per_tier,
            num_classes=num_classes,
            seed=args.seed + 100 * tier_idx,
            max_attempts=args.max_sample_attempts,
        )
        print(f"  selected {len(candidates)} in-budget candidates")

        if args.dry_run:
            for i, (cfg, cost) in enumerate(candidates, start=1):
                append_csv_row(csv_path, make_csv_row(tier, 0, i, cfg, cost, 0, None))
            continue

        # ---------------- Stage 1 ----------------
        stage1_results: List[Tuple[Dict, Dict[str, int], float]] = []
        for i, (cfg, cost) in enumerate(candidates, start=1):
            print(f"\n[{tier} S1 {i}/{len(candidates)}] cfg={cfg}  "
                  f"mem={cost['memory_bytes']//1024}KB ops={cost['ops']/1e6:.2f}M params={cost['params']}")
            datasets, _, _, _ = get_datasets(args, cfg["mfcc_bins"])
            try:
                _, history = train_candidate(cfg, num_classes, datasets,
                                              epochs=args.stage1_epochs,
                                              learning_rate=args.learning_rate)
                val_acc = float(max(history.history.get("val_accuracy", [0.0])))
            except Exception as exc:
                print(f"  training failed: {exc}")
                continue
            stage1_results.append((cfg, cost, val_acc))
            append_csv_row(csv_path, make_csv_row(tier, 1, i, cfg, cost, args.stage1_epochs, val_acc))

        if not stage1_results:
            print(f"[{tier}] no successful Stage-1 runs, skipping tier")
            continue

        stage1_results.sort(key=lambda item: item[2], reverse=True)
        top_k = stage1_results[: args.top_k]
        print(f"\n[{tier}] Stage-1 top {len(top_k)}:")
        for rank, (cfg, _, va) in enumerate(top_k, start=1):
            print(f"  #{rank} val_acc={va:.4f}  {cfg}")

        # ---------------- Stage 2 ----------------
        stage2_results: List[Tuple[Dict, Dict[str, int], float, Dict, Path, tf.keras.Model]] = []
        for rank, (cfg, cost, _) in enumerate(top_k, start=1):
            cand_dir = args.output_dir / f"ds_cnn_{tier}_top{rank}"
            cand_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = cand_dir / "best.keras"
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(ckpt_path), monitor="val_accuracy",
                    mode="max", save_best_only=True,
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", mode="max",
                    patience=args.stage2_patience, restore_best_weights=True,
                ),
                tf.keras.callbacks.CSVLogger(str(cand_dir / "history.csv")),
            ]
            print(f"\n[{tier} S2 {rank}/{len(top_k)}] cfg={cfg}")
            datasets, splits, _, _ = get_datasets(args, cfg["mfcc_bins"])
            model, history = train_candidate(cfg, num_classes, datasets,
                                              epochs=args.stage2_epochs,
                                              learning_rate=args.learning_rate,
                                              callbacks=callbacks)
            best_val = float(max(history.history.get("val_accuracy", [0.0])))
            test_metrics = model.evaluate(datasets["test"], return_dict=True, verbose=0)
            stage2_results.append((cfg, cost, best_val, test_metrics, cand_dir, model))
            append_csv_row(csv_path, make_csv_row(tier, 2, rank, cfg, cost, args.stage2_epochs, best_val))

        if not stage2_results:
            print(f"[{tier}] no successful Stage-2 runs, skipping tier")
            continue

        stage2_results.sort(key=lambda item: item[2], reverse=True)
        best_cfg, best_cost, best_val, best_test, best_dir, best_model = stage2_results[0]

        final_dir = args.output_dir / f"ds_cnn_{tier}"
        final_dir.mkdir(parents=True, exist_ok=True)
        best_model.save(final_dir / "best.keras")
        report = {
            "tier": tier,
            "budget": budget,
            "config": {
                "layers":       best_cfg["layers"],
                "filters":      best_cfg["filters"],
                "kernel":       list(best_cfg["kernel"]),
                "first_stride": list(best_cfg["first_stride"]),
                "mfcc_bins":    best_cfg["mfcc_bins"],
            },
            "cost": best_cost,
            "val_accuracy": best_val,
            "test_accuracy": float(best_test["accuracy"]),
            "test_loss": float(best_test["loss"]),
            "stage2_epochs": args.stage2_epochs,
            "num_classes": num_classes,
            "source_candidate_dir": str(best_dir),
        }
        (final_dir / "report.json").write_text(json.dumps(report, indent=2))
        summary[tier] = report
        print(f"\n[{tier}] BEST cfg={best_cfg}  val={best_val:.4f}  test={best_test['accuracy']:.4f}")

    print("\n=== Sweep complete ===")
    print(json.dumps(summary, indent=2))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
