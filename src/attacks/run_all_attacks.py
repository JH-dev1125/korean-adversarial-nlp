"""
src/attacks/run_all_attacks.py

단일 공격과 복합 공격을 강도별로 실행하여 data/augmented/에 저장한다.

실행 위치:
    프로젝트 최상위 폴더

실행 명령:
    python -m src.attacks.run_all_attacks
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.attacks.coda_manip import CodaManipAttack
from src.attacks.compound_attack import CompoundAttack
from src.attacks.emoji_insert import EmojiInsertAttack
from src.attacks.jamo_split import JamoSplitAttack
from src.attacks.korean_to_english_typing import KoreanToEngTypingAttack
from src.attacks.liaison import LiaisonAttack
from src.attacks.phoneme_sub import PhonemeSubAttack
from src.attacks.romanize import RomanizeAttack
from src.attacks.spacing import SpacingAttack
from src.attacks.visual_sub import VisualSubAttack

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "augmented"

INTENSITIES = [0.1, 0.2, 0.3]
NUM_VARIANTS = 5
RANDOM_SEED = 42


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"입력 파일을 찾을 수 없습니다: {INPUT_PATH}\n"
            "먼저 python -m src.utils.preprocess 를 실행해 data/processed/test.csv를 만들어주세요."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)

    attack_classes = [
        PhonemeSubAttack,           # 음소 치환
        VisualSubAttack,            # 시각적 유사 문자
        RomanizeAttack,             # 로마자 혼용
        JamoSplitAttack,            # 자모 분리
        CodaManipAttack,            # 받침 탈락/삽입
        LiaisonAttack,              # 연음 역이용
        SpacingAttack,              # 띄어쓰기 조작
        EmojiInsertAttack,          # 이모지/특수문자 삽입
        KoreanToEngTypingAttack,    # 영타 변환
        CompoundAttack,             # 2개 이상 공격 조합
    ]

    total_files = len(attack_classes) * len(INTENSITIES)

    print("=" * 60)
    print("공격 데이터 생성 시작")
    print(f"입력 파일: {INPUT_PATH}")
    print(f"반복 변형 개수: {NUM_VARIANTS}개")
    print(f"총 생성 파일 수: {total_files}개")
    print("=" * 60)

    for attack_cls in attack_classes:
        for intensity in INTENSITIES:
            attack = attack_cls(intensity=intensity, seed=RANDOM_SEED)
            attacked_df = attack.apply_to_dataset(df, num_variants=NUM_VARIANTS)

            attack_type = attack.attack_type
            output_path = OUTPUT_DIR / f"test_{attack_type}_{intensity}.csv"
            attacked_df.to_csv(output_path, index=False, encoding="utf-8-sig")

            print(
                f"저장 완료: test_{attack_type}_{intensity}.csv | "
                f"총 {len(attacked_df)}행 | "
                f"label=1 {(attacked_df['label'] == 1).sum()}행 | "
                f"label=0 {(attacked_df['label'] == 0).sum()}행"
            )

            if attack_type == "compound":
                print("  - component_attacks, compound_metadata 컬럼 포함")

    print("=" * 60)
    print("공격 데이터 생성 완료")
    print(f"저장 위치: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
