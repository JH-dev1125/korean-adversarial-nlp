"""
src/attacks/visual_sub.py

시각적 유사 문자 공격.
한글 자모 또는 음절 내부 자모를 비슷하게 보이는 문자로 바꾼다.
"""

from __future__ import annotations

from .base_attack import BaseAttack
from .hangul_utils import decompose_syllable, compose_syllable, is_hangul_syllable


class VisualSubAttack(BaseAttack):
    attack_type = "visual"

    # 완성형 음절 내부에서 다른 한글 자모로 바꿔도 조합 가능한 경우
    CHO_SUB = {
        "ㅇ": ["ㅎ"],
        "ㄱ": ["ㅋ"],
        "ㅂ": ["ㅃ"],
        "ㅈ": ["ㅊ"],
        "ㅅ": ["ㅆ"],
    }

    JUNG_SUB = {
        "ㅣ": ["ㅟ"],
        "ㅡ": ["ㅜ"],
        "ㅗ": ["ㅛ"],
        "ㅜ": ["ㅠ"],
    }

    # 독립 자모나 일반 문자 자체를 직접 바꾸는 경우
    CHAR_SUB = {
        "ㅇ": ["0", "O"],
        "ㅣ": ["l", "I"],
        "ㅡ": ["-", "_"],
        "ㄱ": ["ᆨ"],
        "ㅁ": ["□"],
        "ㅋ": ["ᄏ"],
        "ㅎ": ["ᄒ"],
    }

    def _is_replaceable(self, ch: str) -> bool:
        if ch in self.CHAR_SUB:
            return True
        if is_hangul_syllable(ch):
            cho, jung, _ = decompose_syllable(ch)
            return cho in self.CHO_SUB or jung in self.JUNG_SUB
        return False

    def _replace_char(self, ch: str) -> str:
        if ch in self.CHAR_SUB:
            return self.rng.choice(self.CHAR_SUB[ch])

        cho, jung, jong = decompose_syllable(ch)
        candidates = []

        if cho in self.CHO_SUB:
            for new_cho in self.CHO_SUB[cho]:
                candidates.append(compose_syllable(new_cho, jung, jong))
        if jung in self.JUNG_SUB:
            for new_jung in self.JUNG_SUB[jung]:
                candidates.append(compose_syllable(cho, new_jung, jong))

        if not candidates:
            return ch
        return self.rng.choice(candidates)

    def attack_text(self, text: str) -> str:
        chars = list(str(text))
        positions = [i for i, ch in enumerate(chars) if self._is_replaceable(ch)]
        selected = self._sample_positions(positions)

        for i in selected:
            chars[i] = self._replace_char(chars[i])

        return "".join(chars)
