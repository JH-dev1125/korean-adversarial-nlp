"""
src/attacks/run_all_attacks.py

세 가지 공격을 강도별로 실행해 data/augmented/에 저장한다.

실행 위치:
    프로젝트 최상위 폴더

실행 명령:
    python src/attacks/run_all_attacks.py

반복 변형 개수 변경:
    NUM_VARIANTS 값을 바꾸면 된다.
    예: 5이면 label=1인 각 원문마다 같은 공격/강도에 대해 5개 변형 생성.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# 프로젝트 루트를 import path에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.attacks.phoneme_sub import PhonemeSubAttack
from src.attacks.visual_sub import VisualSubAttack
from src.attacks.romanize import RomanizeAttack

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "augmented"

INTENSITIES = [0.1, 0.2, 0.3]
NUM_VARIANTS = 5  # 같은 원문에 대해 생성할 변형 개수
RANDOM_SEED = 42


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"입력 파일을 찾을 수 없습니다: {INPUT_PATH}\n"
            "먼저 python src/utils/preprocess.py 를 실행해 data/processed/test.csv를 만들어주세요."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)

    attack_classes = [PhonemeSubAttack, VisualSubAttack, RomanizeAttack]

    print("=" * 60)
    print("공격 데이터 생성 시작")
    print(f"입력 파일: {INPUT_PATH}")
    print(f"반복 변형 개수: label=1 행마다 {NUM_VARIANTS}개")
    print("=" * 60)

    for attack_cls in attack_classes:
        for intensity in INTENSITIES:
            attack = attack_cls(intensity=intensity, seed=RANDOM_SEED)
            attacked_df = attack.apply_to_dataset(df, num_variants=NUM_VARIANTS)

            attack_type = attack.attack_type
            output_path = OUTPUT_DIR / f"test_{attack_type}_{intensity}.csv"
            attacked_df.to_csv(output_path, index=False, encoding="utf-8-sig")

            print(
                f"저장 완료: {output_path} | "
                f"총 {len(attacked_df)}행 | "
                f"혐오 변형 행 {(attacked_df['label'] == 1).sum()}개"
            )

    print("=" * 60)
    print("공격 데이터 생성 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
