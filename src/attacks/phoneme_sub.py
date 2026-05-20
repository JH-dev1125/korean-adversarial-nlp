"""
src/attacks/phoneme_sub.py

음소 치환 공격.
PHISH 논문의 아이디어처럼 발음이 유사한 자모끼리 look-up table을 만들어 교체한다.
"""

from __future__ import annotations

from .base_attack import BaseAttack
from .hangul_utils import decompose_syllable, compose_syllable, is_hangul_syllable


class PhonemeSubAttack(BaseAttack):
    attack_type = "phoneme"

    CHO_SUB = {#변형 가능 초성
        "ㅂ": ["ㅍ"], "ㅍ": ["ㅂ"],
        "ㄷ": ["ㅌ"], "ㅌ": ["ㄷ"],
        "ㄱ": ["ㅋ"], "ㅋ": ["ㄱ"],
        "ㅅ": ["ㅆ"], "ㅆ": ["ㅅ"],
        "ㅈ": ["ㅊ"], "ㅊ": ["ㅈ"],
    }
    
    JUNG_SUB = {#변형 가능 중성
        "ㅏ": ["ㅑ"], "ㅑ": ["ㅏ"],
        "ㅓ": ["ㅕ"], "ㅕ": ["ㅓ"],
        "ㅗ": ["ㅛ"], "ㅛ": ["ㅗ"],
        "ㅜ": ["ㅠ"], "ㅠ": ["ㅜ"],
    }

    def _is_replaceable(self, ch: str) -> bool: #변형 가능 모음이나 자음 있으면 공격 가능
        if not is_hangul_syllable(ch):
            return False
        cho, jung, _ = decompose_syllable(ch)
        return cho in self.CHO_SUB or jung in self.JUNG_SUB

    def _replace_syllable(self, ch: str) -> str:
        cho, jung, jong = decompose_syllable(ch)

        candidates = []
        if cho in self.CHO_SUB: #교체 가능하면 바꿈
            for new_cho in self.CHO_SUB[cho]: #하나 밖에 없긴 한데 하나씩 변형해봄
                candidates.append(compose_syllable(new_cho, jung, jong)) #가능한 조합을 저장
        if jung in self.JUNG_SUB: #초성이랑 똑같이 진행
            for new_jung in self.JUNG_SUB[jung]:
                candidates.append(compose_syllable(cho, new_jung, jong))

        if not candidates:
            return ch #바꿀거 없으면 그대로
        return self.rng.choice(candidates) #바꿈

    def attack_text(self, text: str) -> str:
        chars = list(str(text))
        positions = [i for i, ch in enumerate(chars) if self._is_replaceable(ch)]
        selected = self._sample_positions(positions)

        for i in selected:
            chars[i] = self._replace_syllable(chars[i]) #변형함

        return "".join(chars)
