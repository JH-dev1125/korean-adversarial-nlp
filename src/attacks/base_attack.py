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

from __future__ import annotations #타입 힌트를 줘서 읽기 쉽게 해줌

import math #올림함수 ceil()등 불러오기 위함
import random #변형 위치 고를 때 필요함
from typing import List, Sequence #타입 힌트를 위해 List와 Sequence가져옴

import pandas as pd #표 형태를 다루는 라이브러리, pd.DataFrame만들기 위해 불러옴


class BaseAttack:
    """모든 공격의 공통 부모 클래스."""

    attack_type: str = "base" #각 변형 방법마다 덮어씀, 변형 방법 정의

    def __init__(self, intensity: float = 0.2, seed: int | None = 42): #변형 강도 조정, seed는 무작위성의 재형 가능성을 제공
        if not 0 < intensity <= 1: #변형 강도 범위
            raise ValueError("intensity는 0보다 크고 1 이하의 값이어야 합니다.")
        self.intensity = intensity #객체 안에 저장
        self.rng = random.Random(seed) #랜덤 생성기 만듦. 재현성을 위해 seed 정하고 랜덤 생성기를 객체에 저장

    def attack_text(self, text: str) -> str: #각 변형 방법에서 상속받기 위함
        """각 공격 파일에서 구현."""
        raise NotImplementedError

    def _sample_positions(self, positions: Sequence[int]) -> List[int]: #positions는 정수들의 순서 있는 자료, 리스트를 반환
        """
        변형 가능한 위치 목록에서 intensity 비율만큼 무작위 선택.

        예: 변형 가능한 위치가 10개이고 intensity=0.2면 2개 선택.
        단, 가능한 위치가 1개 이상이면 최소 1개는 변형한다.
        """
        if not positions:
            return []

        n_change = max(1, math.ceil(len(positions) * self.intensity)) #변형 개수 선택
        n_change = min(n_change, len(positions)) #변형 개수 선택
        return self.rng.sample(list(positions), n_change) #실제로 무작위 위치를 뽑아서 반환. self.rng.sample()은 리스트에서 몇개 뽑음

    def apply_to_dataset(self, df: pd.DataFrame, num_variants: int = 5) -> pd.DataFrame: #num_variants는 원문 하나당 생성할 행 개수
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
        if missing: #입력 데이터에서 빠진 컬럼 찾음
            raise ValueError(f"입력 데이터에 필요한 컬럼이 없습니다: {sorted(missing)}")

        if num_variants < 1:
            raise ValueError("num_variants는 1 이상이어야 합니다.")

        rows = []

        for _, row in df.iterrows(): #df행 하나씩 읽음
            original_text = str(row["text"]) #아래 세 줄에서는 원문을 저장
            label = int(row["label"]) #혐오표현인가?
            source = row["source"] #데이터셋

            if label == 1:
                # 혐오 텍스트는 같은 원문에 대해 여러 번 무작위 변형 생성
                seen = set() #생성한 변형 결과 저장
                for variant_id in range(1, num_variants + 1): #여러개 변형 만듦
                    attacked_text = self.attack_text(original_text) #변형 데이터 생성

                    # 우연히 같은 변형이 반복되면 몇 번 더 시도
                    retry = 0
                    while attacked_text in seen and retry < 10: #중복된 데이터가 생기지 않도록 함
                        attacked_text = self.attack_text(original_text)
                        retry += 1 #무한 루프에 빠지지 않도록 함

                    seen.add(attacked_text)
                    rows.append( #데이터를 잘 저장, row와 구분
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
                    rows.append(#df의 rows 정함
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

        return pd.DataFrame( #column정하고 반환
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
