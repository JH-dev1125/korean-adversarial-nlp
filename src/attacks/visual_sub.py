"""
src/attacks/visual_sub.py

시각적 유사 문자 공격.
한글 자모 또는 음절 내부 자모를 비슷하게 보이는 문자로 바꾼다.
"""

from __future__ import annotations

from .base_attack import BaseAttack
from .hangul_utils import decompose_syllable, compose_syllable, is_hangul_syllable


class VisualSubAttack(BaseAttack):
    """비슷해 보이는 자모/문자로 바꿔 필터를 우회하는 시각적 유사 문자 공격."""

    # 결과 CSV의 attack_type 컬럼에 저장되는 이름이다.
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
        """문자 하나가 시각적 치환 대상인지 확인한다."""
        if ch in self.CHAR_SUB:
            return True
        if is_hangul_syllable(ch):
            # 완성형 한글은 내부 초성/중성을 바꿔도 조합 가능한 경우만 대상이 된다.
            cho, jung, _ = decompose_syllable(ch)
            return cho in self.CHO_SUB or jung in self.JUNG_SUB
        return False

    def _replace_char(self, ch: str) -> str:
        """
        문자 하나를 시각적으로 유사한 후보 중 하나로 바꾼다.

        CHAR_SUB에 직접 들어 있는 독립 자모/문자는 바로 치환하고,
        완성형 한글 음절은 초성 또는 중성을 바꾼 새 음절 후보를 만든다.
        """
        if ch in self.CHAR_SUB:
            return self.rng.choice(self.CHAR_SUB[ch])

        cho, jung, jong = decompose_syllable(ch)
        # candidates에는 조합 가능한 새 완성형 음절이 들어간다.
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
        """문장 안에서 시각적 치환 가능한 위치를 골라 변형한다."""
        chars = list(str(text))
        positions = [i for i, ch in enumerate(chars) if self._is_replaceable(ch)]
        selected = self._sample_positions(positions)

        for i in selected:
            chars[i] = self._replace_char(chars[i])

        return "".join(chars)
