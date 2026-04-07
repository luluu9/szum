import random
import shutil
from pathlib import Path


SOURCE_DIR = Path("./MERGED")
OUTPUT_DIR = Path("./split_data")

MAIN_CATEGORIES = ["Music", "Speech", "Other"]

SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}

RANDOM_SEED = 42


def load_blacklist():
    with open("blacklist.txt", "r") as f:
        blacklist = {line.strip() for line in f if line.strip()}
    return blacklist


def collect_files_for_category(
    source_dir: Path, category: str, blacklist: set
) -> list[Path]:
    category_path = source_dir / category

    files = [
        path
        for path in category_path.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    ]

    filtered_files = []
    for path in files:
        filename = path.name

        if filename in blacklist:
            print(f"Skipped: {filename}")
            continue

        filtered_files.append(path)

    return filtered_files


def split_files(
    files: list[Path], split_ratios: dict, random: random.Random
) -> dict[str, list[Path]]:
    files = files.copy()
    random.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * split_ratios["train"])
    n_val = int(n_total * split_ratios["val"])
    n_test = n_total - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    assert len(train_files) + len(val_files) + len(test_files) == n_total
    assert len(test_files) == n_test

    return {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }


def copy_files(
    split_name: str,
    category: str,
    files: list[Path],
    source_dir: Path,
    output_dir: Path,
) -> None:
    category_root = source_dir / category

    for src_file in files:
        relative_path = src_file.relative_to(category_root)
        dst_file = output_dir / split_name / category / relative_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)


# Tutaj i w main gpt dopisał logowanie, bo już nie miałem cierpliwości na to ~ Kacper Chodubski
def print_summary(summary: dict[str, dict[str, int]]) -> None:
    for split_name, categories in summary.items():
        print(f"\n[{split_name}]")
        total = 0
        for category, count in categories.items():
            print(f"{category}: {count}")
            total += count
        print(f"  Razem: {total}")


def main():
    rng = random.Random(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "train": {},
        "val": {},
        "test": {},
    }

    for category in MAIN_CATEGORIES:
        files = collect_files_for_category(SOURCE_DIR, category, load_blacklist())

        split_result = split_files(files, SPLIT_RATIOS, rng)

        for split_name, split_files_list in split_result.items():
            copy_files(
                split_name=split_name,
                category=category,
                files=split_files_list,
                source_dir=SOURCE_DIR,
                output_dir=OUTPUT_DIR,
            )
            summary[split_name][category] = len(split_files_list)

        print(f"[OK] Kategoria '{category}' podzielona:")
        print(
            f"     train={len(split_result['train'])}, "
            f"val={len(split_result['val'])}, "
            f"test={len(split_result['test'])}"
        )

    print_summary(summary)
    print(f"\nGotowe. Wynik zapisany w: {OUTPUT_DIR.resolve()}")


from pathlib import Path


# Test na przeciek między zbiorami test / val / train napisany przy pomocy gpt ~ Kacper Chodubski
def get_all_files(root):
    return {str(p.relative_to(root)) for p in root.rglob("*.wav") if p.is_file()}


def test_no_data_leakage(base_dir="./split_data"):
    base = Path(base_dir)

    train_files = get_all_files(base / "train")
    val_files = get_all_files(base / "val")
    test_files = get_all_files(base / "test")

    train_val = train_files & val_files
    train_test = train_files & test_files
    val_test = val_files & test_files

    assert not train_val, f"Pliki w train i val: {train_val}"
    assert not train_test, f"Pliki w train i test: {train_test}"
    assert not val_test, f"Pliki w val i test: {val_test}"

    print("Brak przecieków między zbiorami")


if __name__ == "__main__":
    main()
    test_no_data_leakage()
