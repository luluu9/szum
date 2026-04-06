import shutil
from pathlib import Path
import wave
import contextlib

BASE_DIR = Path("d:/STUDIA/MGR3/SZUM")
DATASETS_DIR = BASE_DIR / "datasets"
MERGED_DIR = DATASETS_DIR / "MERGED"

CLASSES = ["Music", "Speech", "Other"]

def setup_merged_dir():
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
        
    for cls in CLASSES:
        (MERGED_DIR / cls).mkdir(parents=True, exist_ok=True)

def process_demand():
    demand_dir = DATASETS_DIR / "demand"
    target_dir = MERGED_DIR / "Other"
    
    for env_dir in demand_dir.iterdir():
        if env_dir.is_dir():
            ch_file = env_dir / "ch01.wav"
            if ch_file.exists():
                dst_name = f"demand_{env_dir.name}_ch01.wav"
                dst_path = target_dir / dst_name
                shutil.copy2(ch_file, dst_path)

def process_gtzan():
    gtzan_dir = DATASETS_DIR / "gtzan"
    target_dir = MERGED_DIR / "Music"
    
    for genre_dir in gtzan_dir.iterdir():
        if genre_dir.is_dir():
            for wav_file in genre_dir.glob("*.wav"):
                dst_name = f"gtzan_{wav_file.name}"
                dst_path = target_dir / dst_name
                shutil.copy2(wav_file, dst_path)

def get_wav_duration(file_path):
    try:
        with contextlib.closing(wave.open(str(file_path), 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except Exception:
        return 0.0

def process_musan():
    musan_dir = DATASETS_DIR / "musan"
    noise_dir = musan_dir / "noise"
    target_other = MERGED_DIR / "Other"
    
    bg_noises = set()
    fs_annotations = noise_dir / "free-sound" / "ANNOTATIONS"
    if fs_annotations.exists():
        with open(fs_annotations, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("noise-free-sound"):
                    bg_noises.add(line)
                    
    musan_noise_sec = 0.0
    for source in ["free-sound", "sound-bible"]:
        src_dir = noise_dir / source
        if not src_dir.exists():
            continue
        for wav_file in src_dir.glob("*.wav"):
            musan_noise_sec += get_wav_duration(wav_file)
            base_name = wav_file.stem
            if base_name in bg_noises:
                dst_name = f"musan_bg_{base_name}.wav"
            else:
                dst_name = f"musan_{base_name}.wav"
            shutil.copy2(wav_file, target_other / dst_name)
            
    total_other_sec = 0.0
    for wav_file in target_other.glob("*.wav"):
        total_other_sec += get_wav_duration(wav_file)
        
    print(f"Target duration (Other): {total_other_sec:.2f} seconds ({total_other_sec/3600:.2f} h)")

if __name__ == "__main__":
    setup_merged_dir()
    process_demand()
    process_gtzan()
    process_musan()
