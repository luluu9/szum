from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


INPUT_DIR = Path("./dataset")
OUTPUT_DIR = Path("./spectrograms")

SPLITS = ["train", "val", "test"]
CLASSES = ["Music", "Other", "Speech"]

LABEL_MAP = {"Music": 0, "Other": 1, "Speech": 2}

TARGET_SR = 22050
N_MELS = 128
IMAGE_SIZE = (224, 224)


def wav_to_mel_db(audio_path: Path) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    return librosa.power_to_db(S, ref=np.max)


def save_spectrogram_image(S_db: np.ndarray, output_path: Path) -> None:
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    img_data = (plt.cm.viridis(S_norm)[:, :, :3] * 255).astype(np.uint8)
    img_data = img_data[::-1]  # low frequencies at bottom
    Image.fromarray(img_data).resize(IMAGE_SIZE, Image.LANCZOS).save(output_path)


def process_split(split_name: str) -> None:
    for class_name in CLASSES:
        class_input_dir = INPUT_DIR / split_name / class_name
        class_output_dir = OUTPUT_DIR / split_name / class_name

        if not class_input_dir.exists():
            print(f"[SKIP] {class_input_dir} nie istnieje")
            continue

        class_output_dir.mkdir(parents=True, exist_ok=True)
        label = LABEL_MAP[class_name]

        wav_files = [p for p in class_input_dir.rglob("*") if p.suffix.lower() == ".wav"]
        for wav_file in wav_files:
            try:
                S_db = wav_to_mel_db(wav_file)
                output_path = class_output_dir / (wav_file.stem + ".png")
                save_spectrogram_image(S_db, output_path)
                print(f"[OK] {wav_file.name} -> {output_path.name} (label={label})")
            except Exception as e:
                print(f"[ERROR] {wav_file}: {e}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in SPLITS:
        process_split(split_name)
    print("\nDone")


if __name__ == "__main__":
    main()
