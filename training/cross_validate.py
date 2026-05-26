"""
K-fold cross-validation for MobileNetV2 trained from scratch on mel spectrograms.
Train+val data is combined and split into k folds; test set is kept separate.

Usage:
    python cross_validate.py --split split2
    python cross_validate.py --split split2 --folds 3 --epochs 30
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# ── logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    log = logging.getLogger("cv")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
    for h in [logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")]:
        h.setFormatter(fmt)
        log.addHandler(h)
    log.propagate = False
    return log


log: logging.Logger = logging.getLogger("cv")


class EpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, fold: int) -> None:
        super().__init__()
        self._fold = fold
        self._t0 = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log.info(
            "fold %d | epoch %3d | loss %.4f | acc %.4f | val_loss %.4f | val_acc %.4f | %.1fs",
            self._fold, epoch + 1,
            logs.get("loss", 0), logs.get("accuracy", 0),
            logs.get("val_loss", 0), logs.get("val_accuracy", 0),
            time.time() - self._t0,
        )


# ── config ────────────────────────────────────────────────────────────────────

TARGET_SR = 22050
SEGMENT_DURATION = 4
EXPECTED_SAMPLES = TARGET_SR * SEGMENT_DURATION
N_MELS = 128
IMG_SIZE = 128
N_FFT = 2048
HOP_LENGTH = 512

CATEGORIES = ["Music", "Speech", "Other"]
LABEL_MAP = {cat: i for i, cat in enumerate(CATEGORIES)}

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
DROPOUT = 0.3
N_FOLDS = 5


# ── audio → spectrogram ───────────────────────────────────────────────────────

def wav_to_melspec(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    if len(y) < EXPECTED_SAMPLES:
        y = np.pad(y, (0, EXPECTED_SAMPLES - len(y)))
    y = y[:EXPECTED_SAMPLES]

    mel = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    img = (mel_norm * 255).astype(np.float32)
    img = tf.image.resize(img[:, :, np.newaxis], [IMG_SIZE, IMG_SIZE]).numpy()
    img_3ch = np.repeat(img, 3, axis=-1)

    return tf.keras.applications.mobilenet_v2.preprocess_input(img_3ch)


# ── data loading ──────────────────────────────────────────────────────────────

def load_paths(split_dir: Path) -> tuple[list[Path], list[int]]:
    paths, labels = [], []
    for cat in CATEGORIES:
        cat_dir = split_dir / cat
        if not cat_dir.exists():
            log.warning("missing category dir: %s", cat_dir)
            continue
        files = sorted(cat_dir.rglob("*.wav"))
        paths.extend(files)
        labels.extend([LABEL_MAP[cat]] * len(files))
        log.info("    %-8s %d segments", cat, len(files))
    return paths, labels


def make_dataset(paths: list[Path], labels: list[int],
                 batch_size: int, shuffle: bool = False,
                 cache: bool = True) -> tf.data.Dataset:
    def load_fn(path, label):
        spec = tf.py_function(
            lambda p: wav_to_melspec(Path(p.numpy().decode())),
            [path], tf.float32,
        )
        spec.set_shape((IMG_SIZE, IMG_SIZE, 3))
        return spec, label

    ds = tf.data.Dataset.from_tensor_slices(([str(p) for p in paths], labels))
    if shuffle:
        ds = ds.shuffle(4096, seed=42)
    ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ── model ─────────────────────────────────────────────────────────────────────

def build_model(dropout: float = DROPOUT) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None,
    )
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(len(CATEGORIES), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2_scratch")


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_confusion(y_true, y_pred, out_path: Path, title: str = "") -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_cv_summary(fold_accs: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(fold_accs) + 1), fold_accs, color="steelblue", alpha=0.8)
    ax.axhline(np.mean(fold_accs), color="red", linestyle="--",
               label=f"mean = {np.mean(fold_accs):.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Val accuracy")
    ax.set_title("Cross-validation results")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("CV summary plot saved: %s", out_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="K-fold CV for MobileNetV2 from scratch")
    parser.add_argument("--split", default="split2", choices=["split1", "split2", "split3", "split4"])
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    args = parser.parse_args()

    dataset_root = Path("./data") / args.split
    out_dir = Path("./models") / f"cv_scratch_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    global log
    log = setup_logging(out_dir / f"cv_{run_id}.log")
    log.info("CV | split=%s | folds=%d | epochs=%d | dropout=%.2f | weights=None",
             args.split, args.folds, args.epochs, args.dropout)

    # ── load train+val (CV pool) and test ─────────────────────────────────────
    log.info("Loading train+val data from %s", dataset_root)
    all_paths, all_labels = [], []
    for subset in ("train", "val"):
        log.info("[%s]", subset)
        p, l = load_paths(dataset_root / subset)
        all_paths.extend(p)
        all_labels.extend(l)

    log.info("CV pool: %d segments total", len(all_paths))

    log.info("Loading test data...")
    log.info("[test]")
    test_paths, test_labels = load_paths(dataset_root / "test")
    y_test = np.array(test_labels, dtype=np.int32)
    test_ds = make_dataset(test_paths, test_labels, BATCH_SIZE, cache=False)

    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels, dtype=np.int32)

    # ── k-fold cross-validation ───────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_val_accs: list[float] = []
    fold_reports: list[str] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_paths, all_labels), start=1):
        log.info("=" * 60)
        log.info("FOLD %d / %d", fold, args.folds)
        log.info("  train=%d  val=%d", len(train_idx), len(val_idx))

        fold_paths_train = list(all_paths[train_idx])
        fold_labels_train = list(all_labels[train_idx])
        fold_paths_val = list(all_paths[val_idx])
        fold_labels_val = list(all_labels[val_idx])

        arr = np.array(fold_labels_train)
        weights = compute_class_weight("balanced", classes=np.unique(arr), y=arr)
        class_weights = {int(c): float(w) for c, w in zip(np.unique(arr), weights)}

        train_ds = make_dataset(fold_paths_train, fold_labels_train, BATCH_SIZE, shuffle=True)
        val_ds   = make_dataset(fold_paths_val,   fold_labels_val,   BATCH_SIZE)

        tf.keras.backend.clear_session()
        model = build_model(dropout=args.dropout)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LR),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        checkpoint_path = out_dir / f"fold{fold}_best.keras"
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            class_weight=class_weights,
            callbacks=[
                EpochLogger(fold),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(checkpoint_path),
                    monitor="val_accuracy",
                    save_best_only=True,
                    verbose=0,
                ),
            ],
            verbose=0,
        )

        model.load_weights(str(checkpoint_path))
        y_val_true = fold_labels_val
        y_val_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
        val_acc = np.mean(np.array(y_val_pred) == np.array(y_val_true))
        report = classification_report(y_val_true, y_val_pred,
                                       target_names=CATEGORIES, zero_division=0)

        fold_val_accs.append(val_acc)
        fold_reports.append(report)
        log.info("Fold %d val accuracy: %.4f\n%s", fold, val_acc, report)

        plot_confusion(y_val_true, y_val_pred,
                       out_dir / f"fold{fold}_confusion.png",
                       title=f"Fold {fold} (val_acc={val_acc:.4f})")

    # ── CV summary ────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("CROSS-VALIDATION SUMMARY")
    for i, acc in enumerate(fold_val_accs, 1):
        log.info("  Fold %d: %.4f", i, acc)
    log.info("  Mean:   %.4f", np.mean(fold_val_accs))
    log.info("  Std:    %.4f", np.std(fold_val_accs))

    plot_cv_summary(fold_val_accs, out_dir / "cv_summary.png")

    # ── evaluate best fold on test set ────────────────────────────────────────
    best_fold = int(np.argmax(fold_val_accs)) + 1
    log.info("Best fold: %d (val_acc=%.4f) — evaluating on test set", best_fold,
             fold_val_accs[best_fold - 1])

    best_model = tf.keras.models.load_model(str(out_dir / f"fold{best_fold}_best.keras"))
    y_test_pred = np.argmax(best_model.predict(test_ds, verbose=0), axis=1)
    test_acc = np.mean(y_test_pred == y_test)
    test_report = classification_report(y_test, y_test_pred,
                                        target_names=CATEGORIES, zero_division=0)
    log.info("Test accuracy (best fold): %.4f\n%s", test_acc, test_report)
    plot_confusion(y_test, y_test_pred,
                   out_dir / "best_fold_test_confusion.png",
                   title=f"Best fold {best_fold} — test set")

    log.info("All outputs saved in: %s", out_dir.resolve())


if __name__ == "__main__":
    main()
