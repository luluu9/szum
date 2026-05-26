"""
Evaluate a saved MobileNetV2 checkpoint on a chosen dataset split.

Usage:
    python evaluate.py --model models/mobilenet_scratch_split2/best_weights.keras --data data/split2/test
    python evaluate.py --model models/MODEL2_scratch.keras --data data/split2/val
    python evaluate.py --model models/MODEL2.keras --data data/split2/test
"""

import argparse
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

TARGET_SR = 22050
SEGMENT_DURATION = 4
EXPECTED_SAMPLES = TARGET_SR * SEGMENT_DURATION
N_MELS = 128
IMG_SIZE = 128
N_FFT = 2048
HOP_LENGTH = 512

CATEGORIES = ["Music", "Speech", "Other"]
LABEL_MAP = {cat: i for i, cat in enumerate(CATEGORIES)}


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


def load_paths(data_dir: Path) -> tuple[list[Path], list[int]]:
    paths, labels = [], []
    for cat in CATEGORIES:
        cat_dir = data_dir / cat
        if not cat_dir.exists():
            print(f"[warn] missing: {cat_dir}")
            continue
        files = sorted(cat_dir.rglob("*.wav"))
        paths.extend(files)
        labels.extend([LABEL_MAP[cat]] * len(files))
        print(f"  {cat:<8} {len(files)} segments")
    return paths, labels


def make_dataset(paths: list[Path], labels: list[int], batch_size: int = 32) -> tf.data.Dataset:
    def load_fn(path, label):
        spec = tf.py_function(
            lambda p: wav_to_melspec(Path(p.numpy().decode())),
            [path], tf.float32,
        )
        spec.set_shape((IMG_SIZE, IMG_SIZE, 3))
        return spec, label

    ds = tf.data.Dataset.from_tensor_slices(([str(p) for p in paths], labels))
    ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def plot_confusion(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved model checkpoint")
    parser.add_argument("--model", required=True, help="Path to .keras checkpoint")
    parser.add_argument("--data", required=True,
                        help="Path to dataset directory with Music/Speech/Other subdirs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--confusion", action="store_true",
                        help="Save confusion matrix PNG next to the model file")
    args = parser.parse_args()

    model_path = Path(args.model)
    data_dir = Path(args.data)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Model : {model_path}")
    print(f"Data  : {data_dir}")
    print("Loading data...")
    paths, labels = load_paths(data_dir)
    if not paths:
        raise RuntimeError("No .wav files found.")

    print(f"Loading model...")
    model = tf.keras.models.load_model(str(model_path))

    print("Running inference...")
    ds = make_dataset(paths, labels, args.batch_size)
    y_true = np.array(labels, dtype=np.int32)
    y_pred = np.argmax(model.predict(ds, verbose=1), axis=1)

    acc = np.mean(y_pred == y_true)
    print(f"\nAccuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=CATEGORIES, zero_division=0))

    if args.confusion:
        out_path = model_path.parent / f"confusion_{data_dir.name}.png"
        plot_confusion(y_true, y_pred, out_path)


if __name__ == "__main__":
    main()
