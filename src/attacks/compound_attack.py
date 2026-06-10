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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

        self.seed = seed
        self.attack_classes = attack_classes or DEFAULT_ATTACK_CLASSES
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
        n_attacks = self.rng.randint(self.min_attacks, self.max_attacks)
        selected_classes = self.rng.sample(self.attack_classes, k=n_attacks)

        raw_weights = [self.rng.uniform(0.8, 1.2) for _ in range(n_attacks)]
        weight_sum = sum(raw_weights)
        ratios = [weight / weight_sum for weight in raw_weights]

        return list(zip(selected_classes, ratios))

    def attack_text_with_metadata(self, text: str) -> tuple[str, dict[str, Any]]:
        """복합 공격을 적용하고 JSON으로 저장 가능한 메타데이터를 반환한다."""
        attacked_text = str(text)
        components = []

        for order, (attack_cls, ratio) in enumerate(self._choose_components(), start=1):
            component_intensity = max(0.01, min(1.0, self.intensity * ratio))
            component_seed = self.rng.randint(0, 2**31 - 1)
            attack = attack_cls(intensity=component_intensity, seed=component_seed)

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

    def apply_to_dataset(self, df: pd.DataFrame, num_variants: int = 5) -> pd.DataFrame:
        """
        데이터프레임에 복합 공격을 적용한다.

        label=1 텍스트는 복합 공격으로 변형하고, label=0 텍스트는 기존 평가 방식과
        맞추기 위해 변형하지 않은 채 같은 개수만큼 복사한다.
        """
        required = {"text", "label", "source"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"입력 데이터에 필요한 컬럼이 없습니다: {sorted(missing)}")
        if num_variants < 1:
            raise ValueError("num_variants는 1 이상이어야 합니다.")

        rows = []

        for _, row in df.iterrows():
            original_text = str(row["text"])
            label = int(row["label"])
            source = row["source"]

            if label == 1:
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
                            "component_attacks": "+".join(
                                component["attack_type"]
                                for component in metadata["components"]
                            ),
                            "compound_metadata": json.dumps(
                                metadata,
                                ensure_ascii=False,
                            ),
                        }
                    )
            else:
                metadata = {
                    "attack_type": self.attack_type,
                    "total_intensity": self.intensity,
                    "num_attacks": 0,
                    "components": [],
                    "note": "label=0 samples are copied without attack",
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
                "component_attacks",
                "compound_metadata",
            ],
        )


def main() -> None:
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
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {args.input}")

    output_path = args.output
    if output_path is None:
        output_path = (
            PROJECT_ROOT
            / "data"
            / "augmented"
            / f"test_compound_{args.intensity}.csv"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    attack = CompoundAttack(intensity=args.intensity, seed=args.seed)
    attacked_df = attack.apply_to_dataset(df, num_variants=args.num_variants)
    attacked_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"복합 공격 데이터 저장 완료: {output_path}")
    print(f"총 행 수: {len(attacked_df)}")


if __name__ == "__main__":
    main()
