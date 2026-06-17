"""
src/attacks/compound_attack.py

복합 공격 생성 모듈.

하나의 문장에 2개 또는 3개의 단일 공격을 순차적으로 적용한다.
각 변형 결과에는 어떤 공격이 어떤 비율로 적용되었는지 JSON 메타데이터를 남긴다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# 프로젝트 루트 경로. 이 파일은 src/attacks 안에 있으므로 parents[2]가 최상위 폴더이다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# python src/attacks/compound_attack.py처럼 직접 실행해도 src 패키지를 찾을 수 있게 보조한다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.attacks.base_attack import BaseAttack
from src.attacks.coda_manip import CodaManipAttack
from src.attacks.emoji_insert import EmojiInsertAttack
from src.attacks.jamo_split import JamoSplitAttack
from src.attacks.korean_to_english_typing import KoreanToEngTypingAttack
from src.attacks.liaison import LiaisonAttack
from src.attacks.phoneme_sub import PhonemeSubAttack
from src.attacks.romanize import RomanizeAttack
from src.attacks.spacing import SpacingAttack
from src.attacks.visual_sub import VisualSubAttack


# 복합 공격이 조합 대상으로 삼는 단일 공격 클래스 목록이다.
# CompoundAttack은 여기서 2개 또는 3개를 무작위로 골라 순차 적용한다.
DEFAULT_ATTACK_CLASSES = [
    PhonemeSubAttack,
    VisualSubAttack,
    RomanizeAttack,
    JamoSplitAttack,
    CodaManipAttack,
    LiaisonAttack,
    SpacingAttack,
    EmojiInsertAttack,
    KoreanToEngTypingAttack,
]


class CompoundAttack(BaseAttack):
    """2개 이상의 단일 공격을 조합하는 복합 공격."""

    attack_type = "compound"

    def __init__(
        self,
        intensity: float = 0.2,
        seed: int | None = 42,
        min_attacks: int = 2,
        max_attacks: int = 3,
        attack_classes: list[type[BaseAttack]] | None = None,
    ) -> None:
        super().__init__(intensity=intensity, seed=seed)

        # seed는 복합 공격 내부에서 component별 seed를 새로 뽑을 때도 사용한다.
        self.seed = seed
        # attack_classes를 따로 넘기면 실험적으로 특정 공격만 조합할 수 있다.
        self.attack_classes = attack_classes or DEFAULT_ATTACK_CLASSES
        # min_attacks/max_attacks는 한 문장에 섞을 공격 개수 범위이다.
        self.min_attacks = min_attacks
        self.max_attacks = max_attacks

        if self.min_attacks < 2:
            raise ValueError("복합 공격은 최소 2개 이상의 공격을 포함해야 합니다.")
        if self.max_attacks < self.min_attacks:
            raise ValueError("max_attacks는 min_attacks보다 크거나 같아야 합니다.")
        if self.max_attacks > len(self.attack_classes):
            raise ValueError("max_attacks가 사용 가능한 공격 개수보다 큽니다.")

    def _choose_components(self) -> list[tuple[type[BaseAttack], float]]:
        """이번 변형에 사용할 공격 클래스와 적용 비율을 선택한다."""
        # n_attacks는 이번 문장에 실제로 적용할 공격 개수이다.
        n_attacks = self.rng.randint(self.min_attacks, self.max_attacks)
        # selected_classes는 DEFAULT_ATTACK_CLASSES 중 중복 없이 뽑은 공격 클래스들이다.
        selected_classes = self.rng.sample(self.attack_classes, k=n_attacks)

        # raw_weights는 공격별 상대 비중이다.
        # 0.8~1.2 범위로 흔들어 모든 공격이 너무 균등하게만 배분되지 않게 한다.
        raw_weights = [self.rng.uniform(0.8, 1.2) for _ in range(n_attacks)]
        weight_sum = sum(raw_weights)
        # ratios는 합이 1이 되는 공격별 비율이다.
        # component_intensity = 전체 intensity * ratio 로 사용된다.
        ratios = [weight / weight_sum for weight in raw_weights]

        return list(zip(selected_classes, ratios))

    def attack_text_with_metadata(self, text: str) -> tuple[str, dict[str, Any]]:
        """복합 공격을 적용하고 JSON으로 저장 가능한 메타데이터를 반환한다."""
        # attacked_text는 공격이 순차 적용되며 계속 업데이트되는 현재 문장이다.
        attacked_text = str(text)
        # components에는 각 단계의 공격 종류, 비율, 실제 변경 여부가 기록된다.
        components = []

        for order, (attack_cls, ratio) in enumerate(self._choose_components(), start=1):
            # component_intensity는 전체 공격 강도를 각 공격 비율만큼 나눈 값이다.
            # 너무 작아져 아무 변형도 안 되는 일을 줄이기 위해 최소 0.01로 제한한다.
            component_intensity = max(0.01, min(1.0, self.intensity * ratio))
            # 각 component 공격도 자체 rng를 가지므로 매 단계마다 별도 seed를 준다.
            component_seed = self.rng.randint(0, 2**31 - 1)
            attack = attack_cls(intensity=component_intensity, seed=component_seed)

            # before_text와 attacked_text를 비교해 실제로 문자가 바뀌었는지 기록한다.
            before_text = attacked_text
            attacked_text = attack.attack_text(attacked_text)

            components.append(
                {
                    "order": order,
                    "attack_type": attack.attack_type,
                    "ratio": round(ratio, 4),
                    "component_intensity": round(component_intensity, 4),
                    "changed": before_text != attacked_text,
                }
            )

        # metadata는 JSON 문자열로 compound_metadata 컬럼에 저장된다.
        metadata = {
            "attack_type": self.attack_type,
            "total_intensity": self.intensity,
            "num_attacks": len(components),
            "components": components,
        }
        return attacked_text, metadata

    def attack_text(self, text: str) -> str:
        """BaseAttack 인터페이스와 호환되는 단일 텍스트 공격 함수."""
        attacked_text, _ = self.attack_text_with_metadata(text)
        return attacked_text

    def apply_to_dataset(
        self,
        df: pd.DataFrame,
        num_variants: int = 5,
        attack_label0: bool = False,
    ) -> pd.DataFrame:
        """
        데이터프레임에 복합 공격을 적용한다.

        기본값은 label=1 텍스트만 복합 공격으로 변형한다.
        attack_label0=True이면 label=0 정상 텍스트에도 복합 공격을 적용한다.
        """
        required = {"text", "label", "source"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"입력 데이터에 필요한 컬럼이 없습니다: {sorted(missing)}")
        if num_variants < 1:
            raise ValueError("num_variants는 1 이상이어야 합니다.")

        # rows는 최종 CSV로 저장할 행(dict)의 목록이다.
        rows = []

        for _, row in df.iterrows():
            # original_text는 공격 전 원문으로, 변형 결과와 비교/추적할 때 사용한다.
            original_text = str(row["text"])
            # label=1이면 항상 공격 적용, label=0은 attack_label0=True일 때만 공격 적용.
            label = int(row["label"])
            source = row["source"]
            should_attack = label == 1 or attack_label0

            if should_attack:
                # 같은 원문에서 같은 변형 결과가 반복 저장되는 것을 줄이기 위한 집합이다.
                seen = set()
                for variant_id in range(1, num_variants + 1):
                    attacked_text, metadata = self.attack_text_with_metadata(original_text)

                    retry = 0
                    while attacked_text in seen and retry < 10:
                        attacked_text, metadata = self.attack_text_with_metadata(original_text)
                        retry += 1

                    seen.add(attacked_text)
                    rows.append(
                        {
                            "text": attacked_text,
                            "label": label,
                            "source": source,
                            "original_text": original_text,
                            "attack_type": self.attack_type,
                            "intensity": self.intensity,
                            "variant_id": variant_id,
                            "attack_label0": attack_label0,
                            "component_attacks": "+".join(
                                component["attack_type"]
                                for component in metadata["components"]
                            ),
                            # ensure_ascii=False로 저장해야 한글 메타데이터가 사람이 읽기 좋게 남는다.
                            "compound_metadata": json.dumps(
                                metadata,
                                ensure_ascii=False,
                            ),
                        }
                    )
            else:
                # 정상 문장은 공격하지 않는다.
                # 단, label=1과 행 수를 맞추기 위해 같은 num_variants 개수만큼 복사한다.
                metadata = {
                    "attack_type": self.attack_type,
                    "total_intensity": self.intensity,
                    "num_attacks": 0,
                    "components": [],
                    "note": "label=0 samples are copied without attack",
                    "attack_label0": attack_label0,
                }
                for variant_id in range(1, num_variants + 1):
                    rows.append(
                        {
                            "text": original_text,
                            "label": label,
                            "source": source,
                            "original_text": original_text,
                            "attack_type": self.attack_type,
                            "intensity": self.intensity,
                            "variant_id": variant_id,
                            "attack_label0": attack_label0,
                            "component_attacks": "",
                            "compound_metadata": json.dumps(
                                metadata,
                                ensure_ascii=False,
                            ),
                        }
                    )

        return pd.DataFrame(
            rows,
            columns=[
                "text",
                "label",
                "source",
                "original_text",
                "attack_type",
                "intensity",
                "variant_id",
                "attack_label0",
                "component_attacks",
                "compound_metadata",
            ],
        )


def main() -> None:
    """명령행에서 복합 공격 CSV 하나를 생성할 때 사용하는 진입점."""
    parser = argparse.ArgumentParser(description="복합 공격 데이터셋 생성")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "test.csv",
        help="입력 CSV 경로",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="출력 CSV 경로",
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=0.2,
        help="전체 공격 강도",
    )
    parser.add_argument(
        "--num_variants",
        type=int,
        default=5,
        help="문장당 생성할 변형 수",
    )
    parser.add_argument("--seed", type=int, default=42, help="난수 시드")
    parser.add_argument(
        "--attack_label0",
        action="store_true",
        help="label=0 정상 문장에도 복합 공격을 적용",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {args.input}")

    # --output을 생략하면 intensity에 맞는 기본 파일명을 자동 생성한다.
    output_path = args.output
    if output_path is None:
        output_path = (
            PROJECT_ROOT
            / "data"
            / "augmented"
            / (
                f"test_compound_all_labels_{args.intensity}.csv"
                if args.attack_label0
                else f"test_compound_{args.intensity}.csv"
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 입력 test.csv를 읽고 CompoundAttack을 적용한다.
    df = pd.read_csv(args.input)

    attack = CompoundAttack(intensity=args.intensity, seed=args.seed)
    attacked_df = attack.apply_to_dataset(
        df,
        num_variants=args.num_variants,
        attack_label0=args.attack_label0,
    )
    attacked_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"복합 공격 데이터 저장 완료: {output_path}")
    print(f"총 행 수: {len(attacked_df)}")


if __name__ == "__main__":
    main()
