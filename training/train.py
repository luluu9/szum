from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


SPECTROGRAMS_DIR = Path("./spectrograms")
MODELS_DIR = Path("./models")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5
FINE_TUNE_AT = 100  # unfreeze base layers from this index onward

CLASS_NAMES = ["Music", "Other", "Speech"]  # label 0, 1, 2
NUM_CLASSES = len(CLASS_NAMES)

def load_dataset(split):
    split_dir = SPECTROGRAMS_DIR / split
    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        seed=11,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model(freeze_base):
    base = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = not freeze_base

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return models.Model(inputs, outputs), base


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def plot_history(history, fine_history=None, output_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    if fine_history is not None:
        offset = len(acc)
        acc += fine_history.history["accuracy"]
        val_acc += fine_history.history["val_accuracy"]
        loss += fine_history.history["loss"]
        val_loss += fine_history.history["val_loss"]
        for ax in (ax1, ax2):
            ax.axvline(offset, color="gray", linestyle="--", label="Fine-tune start")

    epochs_range = range(len(acc))
    ax1.plot(epochs_range, acc, label="Train accuracy")
    ax1.plot(epochs_range, val_acc, label="Val accuracy")
    ax1.set_title("Accuracy")
    ax1.legend()

    ax2.plot(epochs_range, loss, label="Train loss")
    ax2.plot(epochs_range, val_loss, label="Val loss")
    ax2.set_title("Loss (Categorical Cross-Entropy)")
    ax2.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"[INFO] Training plot saved to {output_path}")
    plt.show()


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = load_dataset("train")
    val_ds = load_dataset("val")

    # Phase 1: train only the head (base frozen)
    print("[INFO] Phase 1 – training classification head (base frozen)")
    model, base = build_model(freeze_base=True)
    compile_model(model, LEARNING_RATE)
    model.summary()

    best_ckpt = MODELS_DIR / "best_head.keras"
    callbacks_head = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        ModelCheckpoint(str(best_ckpt), save_best_only=True, monitor="val_accuracy"),
    ]
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks_head,
    )

    # Phase 2: fine-tune top layers of the base
    print(f"[INFO] Phase 2 – fine-tuning base from layer {FINE_TUNE_AT}")
    base.trainable = True
    for layer in base.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    compile_model(model, FINE_TUNE_LR)

    best_ft_ckpt = MODELS_DIR / "best_finetuned.keras"
    callbacks_ft = [
        EarlyStopping(patience=7, restore_best_weights=True, monitor="val_accuracy"),
        ModelCheckpoint(str(best_ft_ckpt), save_best_only=True, monitor="val_accuracy"),
    ]
    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks_ft,
    )

    model.save(MODELS_DIR / "mobilenetv2_audio_classifier.keras")
    print("[INFO] Model saved.")

    plot_history(history_head, history_ft, MODELS_DIR / "training_plot.png")

    # Evaluate on test set
    test_ds = load_dataset("test")
    loss, acc = model.evaluate(test_ds)
    print(f"\n[RESULT] Test loss: {loss:.4f}  |  Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
