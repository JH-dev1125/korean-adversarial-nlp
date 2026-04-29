"""
src/attacks/coda_manip.py

받침 탈락/삽입 공격.
받침이 있는 음절은 받침을 제거하고,
받침이 없는 음절은 랜덤 받침을 삽입한다.

예시:
    밥 → 바  (받침 탈락)
    바 → 박  (받침 삽입)
    새끼 → 색끼 (받침 삽입)
"""

from __future__ import annotations

from .base_attack import BaseAttack
from .hangul_utils import (
    decompose_syllable,
    compose_syllable,
    is_hangul_syllable,
    JONGSEONG,
)

# 삽입할 받침 후보 목록 (자주 쓰이는 것만)
JONG_CANDIDATES = ["ㄱ", "ㄴ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㄷ", "ㅈ", "ㅎ"]


class CodaManipAttack(BaseAttack):
    attack_type = "coda"

    def _is_replaceable(self, ch: str) -> bool:
        return is_hangul_syllable(ch)

    def _manipulate_coda(self, ch: str) -> str:
        """
        받침 있으면 제거, 없으면 랜덤 삽입
        """
        cho, jung, jong = decompose_syllable(ch)

        if jong:
            # 받침 있으면 제거
            return compose_syllable(cho, jung, "")
        else:
            # 받침 없으면 랜덤 삽입
            new_jong = self.rng.choice(JONG_CANDIDATES)
            return compose_syllable(cho, jung, new_jong)

    def attack_text(self, text: str) -> str:
        chars = list(str(text))
        positions = [i for i, ch in enumerate(chars) if self._is_replaceable(ch)]
        selected = self._sample_positions(positions)

        for i in selected:
            chars[i] = self._manipulate_coda(chars[i])

        return "".join(chars)
