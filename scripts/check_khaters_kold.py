# ====================================================
# scripts/check_khaters_kold.py
# K-HATERS, KOLD 데이터셋 다운로드 및 확인 스크립트
#
# 이 파일이 하는 일:
#   1. Hugging Face에서 K-HATERS, KOLD 다운로드
#   2. data/raw/ 폴더에 저장
#   3. 데이터 구조 및 샘플 출력
#
# 실행 방법:
#   python scripts/check_khaters_kold.py
# ====================================================

from pathlib import Path
from datasets import load_dataset


# ====================================================
# K-HATERS 다운로드 및 확인
# ====================================================
def check_khaters():
    raw_dir = Path("data/raw/khaters")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("K-HATERS 데이터셋 다운로드 중...")
    print("=" * 50)

    ds = load_dataset("humane-lab/K-HATERS")
    ds.save_to_disk(str(raw_dir))

    print("\n데이터셋 구조:")
    print(ds)

    print("\n컬럼:")
    print(ds["train"].column_names)

    print("\n첫 번째 샘플:")
    print(ds["train"][0])

    print("\n데이터 개수:")
    for split in ds.keys():
        print(split, len(ds[split]))

    print(f"\n저장 위치: {raw_dir}")


# ====================================================
# KOLD 다운로드 및 확인
# ====================================================
def check_kold():
    raw_dir = Path("data/raw/kold")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 50)
    print("KOLD 데이터셋 다운로드 중...")
    print("=" * 50)

    ds = load_dataset("nayohan/KOLD")
    ds.save_to_disk(str(raw_dir))

    print("\n데이터셋 구조:")
    print(ds)

    print("\n컬럼:")
    print(ds["train"].column_names)

    print("\n첫 번째 샘플:")
    print(ds["train"][0])

    print("\n데이터 개수:")
    for split in ds.keys():
        print(split, len(ds[split]))

    print(f"\n저장 위치: {raw_dir}")


# ====================================================
# 메인 실행
# ====================================================
if __name__ == "__main__":
    check_khaters()
    check_kold()

    print("\n" + "=" * 50)
    print("✅ K-HATERS + KOLD 다운로드 완료!")
    print("저장 위치: data/raw/khaters/, data/raw/kold/")
    print("=" * 50)
