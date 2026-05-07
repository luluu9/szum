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
from sklearn.utils.class_weight import compute_class_weight

# Claude code użyty do pisania logowania, debugowania i przedewszyskim do pomocy w różnicach pomiędzy TenserFlow, a PyTorch. Nie znam się na TenserFlow więc promptowałem Claude code bardziej jakbym to pisał w PyTorch.


def setup_logging(log_path: Path) -> logging.Logger:
    log = logging.getLogger("mobilenet")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
    for h in [logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")]:
        h.setFormatter(fmt)
        log.addHandler(h)
    log.propagate = False
    return log


log: logging.Logger = logging.getLogger("mobilenet")


class EpochLogger(tf.keras.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()
        self._t0 = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log.info(
            "epoch %3d | loss %.4f | acc %.4f | val_loss %.4f | val_acc %.4f | %.1fs",
            epoch + 1,
            logs.get("loss", 0), logs.get("accuracy", 0),
            logs.get("val_loss", 0), logs.get("val_accuracy", 0),
            time.time() - self._t0,
        )


TARGET_SR = 22050
SEGMENT_DURATION = 4
EXPECTED_SAMPLES = TARGET_SR * SEGMENT_DURATION
N_MELS = 128
IMG_SIZE = 128
N_FFT = 2048
HOP_LENGTH = 512

CATEGORIES = ["Music", "Speech", "Other"]
LABEL_MAP = {cat: i for i, cat in enumerate(CATEGORIES)}
MODEL_NAMES = {"split1": "MODEL1", "split2": "MODEL2", "split3": "MODEL3", "split4": "MODEL4"}

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
FROZEN_LAYERS = 100
DROPOUT = 0.3


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


def load_split_paths(split_dir: Path) -> tuple[list[Path], list[int]]:
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


def subsample(paths: list[Path], labels: list[int],
              fraction: float, seed: int = 42) -> tuple[list[Path], list[int]]:
    rng = np.random.default_rng(seed)
    n = max(len(CATEGORIES), int(len(paths) * fraction))
    idx = rng.choice(len(paths), n, replace=False)
    return [paths[i] for i in idx], [labels[i] for i in idx]


def make_dataset(paths: list[Path], labels: list[int],
                 batch_size: int, shuffle: bool = False,
                 cache: bool = True) -> tf.data.Dataset:
    str_paths = [str(p) for p in paths]

    def load_fn(path, label):
        spec = tf.py_function(
            lambda p: wav_to_melspec(Path(p.numpy().decode())),
            [path], tf.float32,
        )
        spec.set_shape((IMG_SIZE, IMG_SIZE, 3))
        return spec, label

    ds = tf.data.Dataset.from_tensor_slices((str_paths, labels))
    if shuffle:
        ds = ds.shuffle(4096, seed=42)
    ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(frozen_layers: int = FROZEN_LAYERS, dropout: float = DROPOUT) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    for layer in base.layers[:frozen_layers]:
        layer.trainable = False
    trainable = sum(1 for l in base.layers if l.trainable)
    log.info("MobileNetV2: %d/%d layers trainable", trainable, len(base.layers))

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(len(CATEGORIES), activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="mobilenetv2_head")


def get_class_weights(labels: list[int]) -> dict[int, float]:
    arr = np.array(labels)
    weights = compute_class_weight("balanced", classes=np.unique(arr), y=arr)
    cw = {int(c): float(w) for c, w in zip(np.unique(arr), weights)}
    log.info("Class weights: %s", {CATEGORIES[k]: f"{v:.2f}" for k, v in cw.items()})
    return cw


def plot_history(history: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("loss", "accuracy")):
        ax.plot(history[metric], label="train")
        ax.plot(history[f"val_{metric}"], linestyle="--", label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"Training {metric}")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.suptitle("MobileNetV2 training", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Training curves saved: %s", out_path)


def plot_confusion(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    log.info("Confusion matrix saved: %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on mel spectrograms")
    parser.add_argument("--split", default="split2", choices=["split1", "split2", "split3", "split4"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--frozen-layers", type=int, default=FROZEN_LAYERS)
    args = parser.parse_args()

    dataset_root = Path("./data") / args.split
    model_name = MODEL_NAMES[args.split]
    out_dir = Path("./models") / f"mobilenet_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    global log
    log = setup_logging(out_dir / f"train_{run_id}.log")
    log.info(
        "Run %s | split=%s | model=%s | epochs=%d | fraction=%.2f | dropout=%.2f | frozen=%d",
        run_id, args.split, model_name, args.epochs,
        args.fraction, args.dropout, args.frozen_layers,
    )

    log.info("Loading paths from %s", dataset_root)
    splits: dict[str, tuple[list[Path], list[int]]] = {}
    for subset in ("train", "val", "test"):
        log.info("[%s]", subset)
        splits[subset] = load_split_paths(dataset_root / subset)

    train_paths, train_labels = splits["train"]
    if args.fraction < 1.0:
        train_paths, train_labels = subsample(train_paths, train_labels, args.fraction)
        log.info("Subsampled train set to %d files (fraction=%.2f)", len(train_paths), args.fraction)

    val_paths,  val_labels  = splits["val"]
    test_paths, test_labels = splits["test"]
    y_test = np.array(test_labels, dtype=np.int32)
    class_weights = get_class_weights(train_labels)

    train_ds = make_dataset(train_paths, train_labels, BATCH_SIZE, shuffle=True)
    val_ds   = make_dataset(val_paths,   val_labels,   BATCH_SIZE)
    test_ds  = make_dataset(test_paths,  test_labels,  BATCH_SIZE, cache=False)

    model = build_model(frozen_layers=args.frozen_layers, dropout=args.dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_path = out_dir / "best_weights.keras"
    log.info("Training | train=%d  val=%d  epochs=%d", len(train_paths), len(val_paths), args.epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=[
            EpochLogger(),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ],
        verbose=0,
    )

    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    acc = np.mean(y_pred == y_test)
    report = classification_report(y_test, y_pred, target_names=CATEGORIES, zero_division=0)
    log.info("Test accuracy=%.4f\n%s", acc, report)

    save_path = Path("./models") / f"{model_name}.keras"
    model.save(save_path)
    log.info("Best model saved as: %s", save_path)

    plot_history(history.history, out_dir / "training_history.png")
    plot_confusion(y_test, y_pred, out_dir / "confusion.png")
    log.info("All outputs saved in: %s", out_dir.resolve())


if __name__ == "__main__":
    main()
