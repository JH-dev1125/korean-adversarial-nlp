"""
src/attacks/base_attack.py

모든 공격 클래스가 상속받는 공통 틀.

핵심 원칙:
- label=1인 혐오/공격 텍스트는 여러 번 무작위 변형
- label=0인 정상 텍스트는 변형하지 않고 같은 개수만큼 복사
- 모든 원문이 같은 num_variants 개수만큼 출력되어 라벨 비율 유지
- 출력 컬럼:
  text, label, source, original_text, attack_type, intensity, variant_id
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence

import pandas as pd


class BaseAttack:
    """모든 공격의 공통 부모 클래스."""

    attack_type: str = "base"

    def __init__(self, intensity: float = 0.2, seed: int | None = 42):
        if not 0 < intensity <= 1:
            raise ValueError("intensity는 0보다 크고 1 이하의 값이어야 합니다.")
        self.intensity = intensity
        self.rng = random.Random(seed)

    def attack_text(self, text: str) -> str:
        """각 공격 파일에서 구현."""
        raise NotImplementedError

    def _sample_positions(self, positions: Sequence[int]) -> List[int]:
        """
        변형 가능한 위치 목록에서 intensity 비율만큼 무작위 선택.

        예: 변형 가능한 위치가 10개이고 intensity=0.2면 2개 선택.
        단, 가능한 위치가 1개 이상이면 최소 1개는 변형한다.
        """
        if not positions:
            return []

        n_change = max(1, math.ceil(len(positions) * self.intensity))
        n_change = min(n_change, len(positions))
        return self.rng.sample(list(positions), n_change)

    def apply_to_dataset(self, df: pd.DataFrame, num_variants: int = 5) -> pd.DataFrame:
        """
        데이터프레임에 공격 적용.

        Args:
            df: text, label, source 컬럼을 가진 데이터프레임
            num_variants: 각 원문에 대해 생성/복사할 개수

        Returns:
            text, label, source, original_text, attack_type, intensity, variant_id 컬럼을 가진 데이터프레임
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
                # 혐오 텍스트는 같은 원문에 대해 여러 번 무작위 변형 생성
                seen = set()
                for variant_id in range(1, num_variants + 1):
                    attacked_text = self.attack_text(original_text)

                    # 우연히 같은 변형이 반복되면 몇 번 더 시도
                    retry = 0
                    while attacked_text in seen and retry < 10:
                        attacked_text = self.attack_text(original_text)
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
                        }
                    )
            else:
                # 정상 텍스트는 변형하지 않고 같은 개수만큼 복사
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
            ],
        )
