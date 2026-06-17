"""
src/attacks/run_all_attacks.py

단일 공격과 복합 공격을 강도별로 실행하여 data/augmented/에 저장한다.

실행 위치:
    프로젝트 최상위 폴더

실행 명령:
    python -m src.attacks.run_all_attacks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 프로젝트 루트 경로. 이 파일은 src/attacks 안에 있으므로 parents[2]가 최상위 폴더이다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# python src/attacks/run_all_attacks.py처럼 직접 실행해도 src 패키지를 찾을 수 있게 보조한다.
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

# 공격을 적용할 원본 테스트셋.
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

# 공격 적용 결과 CSV가 저장될 폴더.
OUTPUT_DIR = PROJECT_ROOT / "data" / "augmented"

# 공격 강도 목록. 0.1/0.2/0.3은 각각 변형 후보의 약 10%/20%/30%를 바꾼다는 뜻이다.
INTENSITIES = [0.1, 0.2, 0.3]

# 원문 하나당 생성할 변형 개수. variant_id=1~5로 저장된다.
NUM_VARIANTS = 5

# 모든 공격 데이터 생성의 재현성을 위한 기본 시드.
RANDOM_SEED = 42


def main() -> None:
    """모든 공격 유형과 강도 조합을 순회하며 변형 테스트 CSV를 생성한다."""
    parser = argparse.ArgumentParser(
        description="모든 공격 유형과 강도 조합의 변형 테스트셋 생성"
    )
    parser.add_argument(
        "--attack_label0",
        action="store_true",
        help="label=0 정상 문장에도 공격을 적용. 기본값은 label=1만 공격",
    )
    args = parser.parse_args()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"입력 파일을 찾을 수 없습니다: {INPUT_PATH}\n"
            "먼저 python -m src.make_datasets.preprocess 를 실행해 data/processed/test.csv를 만들어주세요."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # df는 data/processed/test.csv 전체이다.
    # 기본값은 label=1만 변형하고 label=0은 복사한다.
    # --attack_label0 옵션이 있으면 label=0도 변형한다.
    df = pd.read_csv(INPUT_PATH)

    # 실행할 공격 클래스 목록.
    # 아래 순서대로 data/augmented/test_{attack_type}_{intensity}.csv가 만들어진다.
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

    # 생성될 CSV 총 개수. 진행 상황 출력용이다.
    total_files = len(attack_classes) * len(INTENSITIES)

    print("=" * 60)
    print("공격 데이터 생성 시작")
    print(f"입력 파일: {INPUT_PATH}")
    print(f"반복 변형 개수: {NUM_VARIANTS}개")
    print(
        "공격 대상: "
        + ("label=0, label=1 전체" if args.attack_label0 else "label=1만")
    )
    print(f"총 생성 파일 수: {total_files}개")
    print("=" * 60)

    for attack_cls in attack_classes:
        for intensity in INTENSITIES:
            # attack은 현재 공격 유형/강도를 가진 공격 객체이다.
            attack = attack_cls(intensity=intensity, seed=RANDOM_SEED)
            # attacked_df는 원본 test.csv보다 NUM_VARIANTS배 많은 행을 가진다.
            attacked_df = attack.apply_to_dataset(
                df,
                num_variants=NUM_VARIANTS,
                attack_label0=args.attack_label0,
            )

            # attack_type은 각 공격 클래스의 class 변수이다. 예: "jamo", "spacing".
            attack_type = attack.attack_type
            # label=0도 공격한 파일은 기존 label=1 전용 파일을 덮어쓰지 않도록
            # 파일명에 all_labels를 붙인다.
            filename = (
                f"test_{attack_type}_all_labels_{intensity}.csv"
                if args.attack_label0
                else f"test_{attack_type}_{intensity}.csv"
            )
            output_path = OUTPUT_DIR / filename
            attacked_df.to_csv(output_path, index=False, encoding="utf-8-sig")

            print(
                f"저장 완료: {filename} | "
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
