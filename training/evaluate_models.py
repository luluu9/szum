import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from train_kacper import BATCH_SIZE, CATEGORIES, load_split_paths, make_dataset


_orig_bn_init = tf.keras.layers.BatchNormalization.__init__

def _patched_bn_init(self, **kwargs):
    kwargs.pop("renorm", None)
    kwargs.pop("renorm_clipping", None)
    kwargs.pop("renorm_momentum", None)
    _orig_bn_init(self, **kwargs)

tf.keras.layers.BatchNormalization.__init__ = _patched_bn_init

MODELS_DIR = Path("./models")
DATA_DIR = Path("../data")
OUT_DIR = Path("./results")

MODELS = {
    "MODEL1": "split1",
    "MODEL2": "split2",
    "MODEL3": "split3",
}

all_results = {}

for model_name, split in MODELS.items():
    model = tf.keras.models.load_model(str(MODELS_DIR / f"{model_name}.keras"))
    out = OUT_DIR / model_name
    out.mkdir(parents=True, exist_ok=True)

    model_results = {}

    for subset in ("train", "val", "test"):
        subset_dir = DATA_DIR / split / subset
        if not subset_dir.exists():
            print(f"{model_name}/{subset}: brak danych, pomijam")
            continue

        paths, labels = load_split_paths(subset_dir)
        if not paths:
            continue

        ds = make_dataset(paths, labels, BATCH_SIZE, cache=False)
        y_true = np.array(labels)
        y_pred = np.argmax(model.predict(ds, verbose=0), axis=1)

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        model_results[subset] = (acc, macro_f1)

        print(f"\n{model_name} | {subset} | acc={acc:.4f}  macro-F1={macro_f1:.4f}")
        print(classification_report(y_true, y_pred, target_names=CATEGORIES, zero_division=0))

        # log błędnie sklasyfikowanych plików
        misclassified = [
            f"TRUE={CATEGORIES[y_true[i]]}  PRED={CATEGORIES[y_pred[i]]}  {paths[i]}"
            for i in range(len(paths)) if y_true[i] != y_pred[i]
        ]
        log_path = out / f"{subset}_errors.txt"
        log_path.write_text(
            f"{model_name} | {subset} | {len(misclassified)}/{len(paths)} błędów\n\n"
            + "\n".join(misclassified),
            encoding="utf-8",
        )

        # wykres metryk
        fig, ax1 = plt.subplots(figsize=(5, 4))
        ax1.bar(["Accuracy", "Macro F1"], [acc, macro_f1], color=["steelblue", "darkorange"])
        ax1.set_ylim(0, 1.15)
        ax1.set_title(f"{model_name} - {subset}")
        for i, v in enumerate([acc, macro_f1]):
            ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)
        plt.tight_layout()
        plt.savefig(out / f"{subset}.png", dpi=150)
        plt.close()

        # confusion matrix osobno
        cm = confusion_matrix(y_true, y_pred)
        fig, ax2 = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=cm.astype(str), fmt="", cmap="Blues",
            xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax2,
            annot_kws={"size": 14},
        )
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("True")
        ax2.set_title(f"{model_name} - {subset} - confusion matrix")
        plt.tight_layout()
        plt.savefig(out / f"{subset}_cm.png", dpi=150)
        plt.close()

    all_results[model_name] = model_results

# porownanie wszystkich modeli
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
x = np.arange(len(MODELS))

for ax, (metric_idx, metric_name) in zip(axes, [(0, "Accuracy"), (1, "Macro F1")]):
    for i, subset in enumerate(("train", "val", "test")):
        values = [all_results[m].get(subset, (None, None))[metric_idx] for m in MODELS]
        valid_x = [x[j] + (i - 1) * 0.25 for j, v in enumerate(values) if v is not None]
        valid_v = [v for v in values if v is not None]
        ax.bar(valid_x, valid_v, 0.25, label=subset)
    ax.set_xticks(x)
    ax.set_xticklabels(list(MODELS.keys()))
    ax.set_ylim(0, 1.15)
    ax.set_title(metric_name)
    ax.legend()

plt.suptitle("Porownanie modeli")
plt.tight_layout()
plt.savefig(OUT_DIR / "comparison.png", dpi=150)
plt.close()

print(f"\nWyniki zapisane w: {OUT_DIR.resolve()}")
