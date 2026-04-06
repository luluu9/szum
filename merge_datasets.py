import shutil
from pathlib import Path

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

def process_musan():
    pass

if __name__ == "__main__":
    setup_merged_dir()
    process_demand()
    process_gtzan()
