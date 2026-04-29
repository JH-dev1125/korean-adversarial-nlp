from pathlib import Path
from datasets import load_dataset


def main():
    raw_dir = Path("data/raw/kmhas")
    raw_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("jeanlee/kmhas_korean_hate_speech")
    ds.save_to_disk(str(raw_dir))

    print("데이터셋 구조:")
    print(ds)

    print("\n컬럼:")
    print(ds["train"].column_names)

    print("\n첫 번째 샘플:")
    print(ds["train"][0])

    print("\n데이터 개수:")
    for split in ds.keys():
        print(split, len(ds[split]))

    print(f"\n저장 위치: {raw_dir}")


if __name__ == "__main__":
    main()