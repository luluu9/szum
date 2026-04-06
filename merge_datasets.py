import os
import shutil
import wave
import contextlib
from pathlib import Path

BASE_DIR = Path("d:/STUDIA/MGR3/SZUM")
DATASETS_DIR = BASE_DIR / "datasets"
MERGED_DIR = DATASETS_DIR / "MERGED"

CLASSES = ["Music", "Speech", "Other"]

def get_wav_duration(file_path):
    try:
        with contextlib.closing(wave.open(str(file_path), 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except Exception:
        return 0.0

def get_total_duration(directory):
    total_sec = 0.0
    if directory.exists():
        for wav_file in directory.glob("*.wav"):
            total_sec += get_wav_duration(wav_file)
    return total_sec

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

def process_gtzan(target_sec):
    gtzan_dir = DATASETS_DIR / "gtzan"
    target_dir = MERGED_DIR / "Music"
    
    genres = [d for d in gtzan_dir.iterdir() if d.is_dir()]
    if not genres:
        return
        
    budget_per_genre = target_sec / len(genres)
    
    for genre_dir in genres:
        copied_sec_for_genre = 0.0
        for wav_file in genre_dir.glob("*.wav"):
            if copied_sec_for_genre >= budget_per_genre:
                break
                
            dst_name = f"gtzan_{wav_file.name}"
            dst_path = target_dir / dst_name
            shutil.copy2(wav_file, dst_path)
            copied_sec_for_genre += get_wav_duration(wav_file)

def process_musan_noise():
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
                    
    for source in ["free-sound", "sound-bible"]:
        src_dir = noise_dir / source
        if not src_dir.exists():
            continue
        for wav_file in src_dir.glob("*.wav"):
            base_name = wav_file.stem
            if base_name in bg_noises:
                dst_name = f"musan_bg_{base_name}.wav"
            else:
                dst_name = f"musan_{base_name}.wav"
            shutil.copy2(wav_file, target_other / dst_name)

def process_musan_speech(missing_sec):
    musan_dir = DATASETS_DIR / "musan"
    speech_dir = musan_dir / "speech"
    target_speech = MERGED_DIR / "Speech"
    
    english_librivox = set()
    lv_annotations = speech_dir / "librivox" / "ANNOTATIONS"
    if lv_annotations.exists():
        with open(lv_annotations, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3 and "english" in parts:
                    english_librivox.add(parts[0])
                    
    us_gov_files = []
    src_us_gov = speech_dir / "us-gov"
    if src_us_gov.exists():
        us_gov_files.extend(list(src_us_gov.glob("*.wav")))
            
    librivox_files = []
    src_librivox = speech_dir / "librivox"
    if src_librivox.exists():
        for wav_file in src_librivox.glob("*.wav"):
            if wav_file.stem in english_librivox:
                librivox_files.append(wav_file)
                
    budget_per_source = missing_sec / 2.0
    
    copied_us_gov = 0.0
    for wav_file in us_gov_files:
        if copied_us_gov >= budget_per_source:
            break
            
        dur = get_wav_duration(wav_file)
        dst_name = f"musan_{wav_file.name}"
        shutil.copy2(wav_file, target_speech / dst_name)
        copied_us_gov += dur
        
    copied_librivox = 0.0
    for wav_file in librivox_files:
        if copied_librivox >= budget_per_source:
            break
            
        dur = get_wav_duration(wav_file)
        dst_name = f"musan_{wav_file.name}"
        shutil.copy2(wav_file, target_speech / dst_name)
        copied_librivox += dur

def process_musan_music(missing_sec):
    musan_dir = DATASETS_DIR / "musan"
    music_dir = musan_dir / "music"
    target_music = MERGED_DIR / "Music"
    
    files_to_copy = []
    for source in ["fma", "fma-western-art", "hd-classical", "jamendo", "rfm"]:
        src_dir = music_dir / source
        if src_dir.exists():
            files_to_copy.extend(src_dir.glob("*.wav"))
            
    copied_sec = 0.0
    for wav_file in files_to_copy:
        if copied_sec >= missing_sec:
            break
            
        dur = get_wav_duration(wav_file)
        dst_name = f"musan_{wav_file.name}"
        shutil.copy2(wav_file, target_music / dst_name)
        copied_sec += dur

if __name__ == "__main__":
    setup_merged_dir()
    process_demand()
    process_musan_noise()
    
    target_sec = get_total_duration(MERGED_DIR / "Other")
    print(f"Target limit seconds: {target_sec:.2f}")
    
    process_gtzan(target_sec)
    
    curr_speech_sec = get_total_duration(MERGED_DIR / "Speech")
    missing_speech = target_sec - curr_speech_sec
    print(f"Need {missing_speech:.2f} more seconds for Speech")
    process_musan_speech(missing_speech)
    
    curr_music_sec = get_total_duration(MERGED_DIR / "Music")
    missing_music = target_sec - curr_music_sec
    print(f"Need {missing_music:.2f} more seconds for Music")
    process_musan_music(missing_music)

