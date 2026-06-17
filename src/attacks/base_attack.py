"""
src/attacks/base_attack.py

모든 공격 클래스가 상속받는 공통 틀.

핵심 원칙:
- 기본값은 label=1인 혐오/공격 텍스트만 여러 번 무작위 변형
- label=0인 정상 텍스트는 기본적으로 변형하지 않고 같은 개수만큼 복사
- attack_label0=True이면 label=0 정상 텍스트도 같은 방식으로 변형
- 모든 원문이 같은 num_variants 개수만큼 출력되어 라벨 비율 유지
- 출력 컬럼:
  text, label, source, original_text, attack_type, intensity, variant_id, attack_label0
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence

import pandas as pd


class BaseAttack:
    """
    모든 공격의 공통 부모 클래스.

    단일 공격 클래스들은 이 클래스를 상속받고 attack_text()만 각자 방식으로 구현한다.
    apply_to_dataset()은 모든 공격이 같은 CSV 구조를 만들도록 공통으로 제공된다.
    """

    # attack_type은 결과 CSV의 attack_type 컬럼에 저장되는 공격 이름이다.
    # 자식 클래스에서 "jamo", "visual"처럼 반드시 덮어쓴다.
    attack_type: str = "base"

    def __init__(self, intensity: float = 0.2, seed: int | None = 42):
        """
        intensity:
            변형 가능한 위치 중 몇 비율을 바꿀지 나타내는 공격 강도.
            예를 들어 0.2이면 변형 가능한 글자/위치의 약 20%를 바꾼다.
        seed:
            같은 입력에서 같은 무작위 선택이 나오게 만드는 난수 시드.
            None을 주면 실행할 때마다 달라질 수 있다.
        """
        if not 0 < intensity <= 1:
            raise ValueError("intensity는 0보다 크고 1 이하의 값이어야 합니다.")
        self.intensity = intensity
        # random 모듈 전역 난수 대신 객체별 rng를 써서 공격별 재현성을 유지한다.
        self.rng = random.Random(seed)

    def attack_text(self, text: str) -> str:
        """각 공격 파일에서 구현."""
        raise NotImplementedError

    def _sample_positions(self, positions: Sequence[int]) -> List[int]:
        """
        변형 가능한 위치 목록에서 intensity 비율만큼 무작위 선택.

        positions:
            글자 인덱스 또는 공백 삽입 위치처럼 "변형 후보"를 나타내는 정수 목록.
        반환값:
            실제로 변형할 위치들의 리스트.

        예: 변형 가능한 위치가 10개이고 intensity=0.2면 2개 선택.
        단, 가능한 위치가 1개 이상이면 최소 1개는 변형한다.
        """
        if not positions:
            return []

        # 변형 개수는 올림 처리한다. 3개 후보에서 intensity=0.1이면 최소 1개를 바꾼다.
        n_change = max(1, math.ceil(len(positions) * self.intensity))
        # 후보 개수보다 많이 뽑는 상황을 방지한다.
        n_change = min(n_change, len(positions))
        return self.rng.sample(list(positions), n_change)

    def apply_to_dataset(
        self,
        df: pd.DataFrame,
        num_variants: int = 5,
        attack_label0: bool = False,
    ) -> pd.DataFrame:
        """
        데이터프레임에 공격 적용.

        Args:
            df: text, label, source 컬럼을 가진 데이터프레임
            num_variants: 각 원문에 대해 생성/복사할 개수
            attack_label0: True이면 label=0 정상 문장에도 공격을 적용한다.

        Returns:
            text, label, source, original_text, attack_type, intensity, variant_id,
            attack_label0 컬럼을 가진 데이터프레임
        """
        # 입력 CSV가 공통 컬럼을 갖고 있는지 확인한다.
        required = {"text", "label", "source"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"입력 데이터에 필요한 컬럼이 없습니다: {sorted(missing)}")

        if num_variants < 1:
            raise ValueError("num_variants는 1 이상이어야 합니다.")

        # rows는 최종 DataFrame을 만들기 위한 dict 목록이다.
        rows = []

        for _, row in df.iterrows():
            # original_text는 공격 전 문장이다. 결과 CSV에서 원문 추적에 사용한다.
            original_text = str(row["text"])
            # label=1은 혐오/공격 문장, label=0은 일반 문장이다.
            label = int(row["label"])
            # source는 khaters/kold/kmhas처럼 원래 데이터셋 이름이다.
            source = row["source"]

            should_attack = label == 1 or attack_label0

            if should_attack:
                # 공격 대상 텍스트는 같은 원문에 대해 여러 번 무작위 변형 생성
                # seen은 같은 원문에서 완전히 같은 변형이 반복 저장되는 것을 줄이기 위한 집합이다.
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
                            "attack_label0": attack_label0,
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
                            "attack_label0": attack_label0,
                        }
                    )

        # 컬럼 순서를 명시해 모든 공격 CSV가 같은 구조를 갖게 한다.
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
            ],
        )
