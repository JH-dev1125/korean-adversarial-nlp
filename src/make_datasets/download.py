# ====================================================
# src/utils/download_datasets.py
# K-HATERS, KOLD, K-MHaS 데이터셋 다운로드 및 확인 스크립트
#
# 이 파일이 하는 일:
#   1. Hugging Face에서 K-HATERS, KOLD, K-MHaS 다운로드
#   2. data/raw/ 폴더에 저장
#   3. 데이터 구조 및 샘플 출력
#
# 실행 방법:
#   python src/utils/download_datasets.py
# ====================================================

from pathlib import Path
from datasets import load_dataset


def download(name, hf_id, raw_subdir):
    """데이터셋을 다운로드하고 구조를 출력한다."""
    raw_dir = Path(f"data/raw/{raw_subdir}")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print(f"{name} 데이터셋 다운로드 중...")
    print("=" * 50)

    # ds는 Hugging Face DatasetDict이다.
    # save_to_disk로 저장하면 preprocess.py에서 load_from_disk로 다시 읽을 수 있다.
    ds = load_dataset(hf_id)
    ds.save_to_disk(str(raw_dir))

    print("\n데이터셋 구조:")
    print(ds)

    print("\n컬럼:")
    print(ds["train"].column_names)

    print("\n첫 번째 샘플:")
    print(ds["train"][0])

    print("\n데이터 개수:")
    # split은 train/validation/test처럼 데이터셋이 나뉜 이름이다.
    for split in ds.keys():
        print(split, len(ds[split]))

    print(f"\n저장 위치: {raw_dir}")


if __name__ == "__main__":
    download("K-HATERS", "humane-lab/K-HATERS", "khaters")
    print()
    download("KOLD", "nayohan/KOLD", "kold")
    print()
    download("K-MHaS", "jeanlee/kmhas_korean_hate_speech", "kmhas")

    print("\n" + "=" * 50)
    print(" K-HATERS + KOLD + K-MHaS 다운로드 완료")
    print("저장 위치: data/raw/khaters/, data/raw/kold/, data/raw/kmhas/")
    print("=" * 50)
