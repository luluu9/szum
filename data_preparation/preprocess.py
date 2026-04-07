import math
from os import path
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


INPUT_DIR = Path("./split_data")
OUTPUT_DIR = Path("./dataset")

SPLITS = ["train", "val", "test"]

SEGMENT_DURATION_SECONDS = 4
TARGET_SR = 22050


def normalize_audio(y: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak == 0:
        return y.astype(np.float32)
    return (y / peak).astype(np.float32)


def collect_audio_files(split_dir: Path) -> list[Path]:
    return [
        p for p in split_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"
    ]


def make_unique_output_path(
    output_split_dir: Path, stem: str, segment_idx: int
) -> Path:
    candidate = output_split_dir / f"{stem}-{segment_idx}.wav"
    if not candidate.exists():
        return candidate

    raise Exception(f"Duplikat dla {path}")


def split_and_save_file(audio_path: Path, output_split_dir: Path) -> int:
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)

    samples_per_segment = SEGMENT_DURATION_SECONDS * sr
    num_segments = len(y) // samples_per_segment

    saved_count = 0
    base_name = audio_path.stem

    for i in range(num_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        chunk = y[start:end]

        chunk = normalize_audio(chunk)

        output_path = make_unique_output_path(output_split_dir, base_name, i)
        sf.write(output_path, chunk, sr)
        saved_count += 1

    return saved_count


def process_split(split_name: str) -> None:
    input_split_dir = INPUT_DIR / split_name
    output_split_dir = OUTPUT_DIR / split_name
    output_split_dir.mkdir(parents=True, exist_ok=True)
    files = collect_audio_files(input_split_dir)
    total_segments = 0
    for audio_file in files:
        try:
            count = split_and_save_file(audio_file, output_split_dir)
            total_segments += count
            print(f"[OK] {audio_file} -> {count} segmentów")
        except Exception as e:
            print(f"[ERROR] {audio_file}: {e}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in SPLITS:
        process_split(split_name)

    print(f"\nEnd")


if __name__ == "__main__":
    main()
