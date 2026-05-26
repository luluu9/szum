import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from train_kacper import BATCH_SIZE, CATEGORIES, make_dataset

_orig_bn_init = tf.keras.layers.BatchNormalization.__init__

def _patched_bn_init(self, **kwargs):
    kwargs.pop("renorm", None)
    kwargs.pop("renorm_clipping", None)
    kwargs.pop("renorm_momentum", None)
    _orig_bn_init(self, **kwargs)

tf.keras.layers.BatchNormalization.__init__ = _patched_bn_init

# mapowanie nazw katalogów w test-outside → indeksy CATEGORIES
DIR_TO_LABEL = {"music": 0, "speech": 1, "noise": 2}  # Music=0, Speech=1, Other=2

MODELS_DIR = Path("./models")
TEST_DIR = Path("../data/test-outside/test")
OUT_DIR = Path("./results/outside")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["MODEL1", "MODEL2", "MODEL3"]


def load_outside(test_dir):
    paths, labels = [], []
    for dir_name, label in DIR_TO_LABEL.items():
        cat_dir = test_dir / dir_name
        if not cat_dir.exists():
            print(f"  brak katalogu: {cat_dir}")
            continue
        files = sorted(cat_dir.rglob("*.wav"))
        paths.extend(files)
        labels.extend([label] * len(files))
        print(f"  {dir_name} ({CATEGORIES[label]}): {len(files)} plików")
    return paths, labels


print("Wczytywanie danych...")
paths, labels = load_outside(TEST_DIR)
y_true = np.array(labels)
print(f"Łącznie: {len(paths)} plików\n")

all_results = {}

for model_name in MODELS:
    print(f"=== {model_name} ===")
    model = tf.keras.models.load_model(str(MODELS_DIR / f"{model_name}.keras"))

    ds = make_dataset(paths, labels, BATCH_SIZE, cache=False)
    y_pred = np.argmax(model.predict(ds, verbose=0), axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    all_results[model_name] = (acc, macro_f1)

    print(f"acc={acc:.4f}  macro-F1={macro_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=CATEGORIES, zero_division=0))

    # log błędów
    misclassified = [
        f"TRUE={CATEGORIES[y_true[i]]}  PRED={CATEGORIES[y_pred[i]]}  {paths[i]}"
        for i in range(len(paths)) if y_true[i] != y_pred[i]
    ]
    (OUT_DIR / f"{model_name}_errors.txt").write_text(
        f"{model_name} | outside-test | {len(misclassified)}/{len(paths)} błędów\n\n"
        + "\n".join(misclassified),
        encoding="utf-8",
    )

    # wykres metryk
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Accuracy", "Macro F1"], [acc, macro_f1], color=["steelblue", "darkorange"])
    ax.set_ylim(0, 1.15)
    ax.set_title(f"{model_name} — outside test")
    for i, v in enumerate([acc, macro_f1]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{model_name}_metrics.png", dpi=150)
    plt.close()

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax)
    thresh = cm.max() / 2
    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            ax.text(j + 0.5, i + 0.5, str(cm[i, j]),
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} — outside test — confusion matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{model_name}_cm.png", dpi=150)
    plt.close()

# porównanie modeli
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
x = np.arange(len(MODELS))
for ax, (metric_idx, metric_name) in zip(axes, [(0, "Accuracy"), (1, "Macro F1")]):
    values = [all_results[m][metric_idx] for m in MODELS]
    bars = ax.bar(x, values, color=["steelblue", "darkorange", "seagreen"])
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 1.15)
    ax.set_title(metric_name)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")

plt.suptitle("Porównanie modeli — outside test")
plt.tight_layout()
plt.savefig(OUT_DIR / "comparison.png", dpi=150)
plt.close()

print(f"\nWyniki zapisane w: {OUT_DIR.resolve()}")
