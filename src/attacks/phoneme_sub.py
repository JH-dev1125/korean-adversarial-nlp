"""
src/attacks/phoneme_sub.py

음소 치환 공격.
PHISH 논문의 아이디어처럼 발음이 유사한 자모끼리 look-up table을 만들어 교체한다.
"""

from __future__ import annotations

from .base_attack import BaseAttack
from .hangul_utils import decompose_syllable, compose_syllable, is_hangul_syllable


class PhonemeSubAttack(BaseAttack):
    """발음이 비슷한 초성/중성/종성을 다른 자모로 바꾸는 공격."""

    # 결과 CSV의 attack_type 컬럼에 저장되는 이름이다.
    attack_type = "phoneme"

    # CHO_SUB는 초성 치환표이다.
    # key는 원래 초성, value는 바꿀 수 있는 초성 후보 목록이다.
    CHO_SUB = {
        "ㅂ": ["ㅍ"], "ㅍ": ["ㅂ"],
        "ㄷ": ["ㅌ"], "ㅌ": ["ㄷ"],
        "ㄱ": ["ㅋ"], "ㅋ": ["ㄱ"],
        "ㅅ": ["ㅆ"], "ㅆ": ["ㅅ"],
        "ㅈ": ["ㅊ"], "ㅊ": ["ㅈ"],
    }
    
    # JUNG_SUB는 중성 치환표이다.
    # 예: "ㅏ"와 "ㅑ"처럼 발음/모양이 가까운 모음을 서로 바꾼다.
    JUNG_SUB = {
        "ㅏ": ["ㅑ"], "ㅑ": ["ㅏ"],
        "ㅓ": ["ㅕ"], "ㅕ": ["ㅓ"],
        "ㅗ": ["ㅛ"], "ㅛ": ["ㅗ"],
        "ㅜ": ["ㅠ"], "ㅠ": ["ㅜ"],
    }

    # JONG_SUB는 종성 치환표이다.
    # 한국어 받침은 실제 발음에서 여러 자모가 같은 소리로 중화되므로 그 관계를 이용한다.
    JONG_SUB = {
        # 종성 ㄱ 계열: [ㄱ]으로 발음될 수 있음
        "ㄱ": ["ㅋ", "ㄲ"],
        "ㅋ": ["ㄱ", "ㄲ"],
        "ㄲ": ["ㄱ", "ㅋ"],

        # 종성 ㄷ 계열: [ㄷ]으로 발음될 수 있음
        "ㄷ": ["ㅌ", "ㅅ", "ㅆ", "ㅈ", "ㅊ", "ㅎ"],
        "ㅌ": ["ㄷ", "ㅅ", "ㅆ", "ㅈ", "ㅊ", "ㅎ"],
        "ㅅ": ["ㄷ", "ㅌ", "ㅆ", "ㅈ", "ㅊ", "ㅎ"],
        "ㅆ": ["ㄷ", "ㅌ", "ㅅ", "ㅈ", "ㅊ", "ㅎ"],
        "ㅈ": ["ㄷ", "ㅌ", "ㅅ", "ㅆ", "ㅊ", "ㅎ"],
        "ㅊ": ["ㄷ", "ㅌ", "ㅅ", "ㅆ", "ㅈ", "ㅎ"],
        "ㅎ": ["ㄷ", "ㅌ", "ㅅ", "ㅆ", "ㅈ", "ㅊ"],

        # 종성 ㅂ 계열: [ㅂ]으로 발음될 수 있음
        "ㅂ": ["ㅍ"],
        "ㅍ": ["ㅂ"],
    }

    def _is_replaceable(self, ch: str) -> bool:
        """치환표에 포함된 초성/중성/종성을 가진 한글 음절인지 확인한다."""
        if not is_hangul_syllable(ch):
            return False
        cho, jung, jong = decompose_syllable(ch)
        return cho in self.CHO_SUB or jung in self.JUNG_SUB or jong in self.JONG_SUB

    def _replace_syllable(self, ch: str) -> str:
        """
        음절 하나를 가능한 후보 중 하나로 치환한다.

        candidates:
            초성 치환, 중성 치환, 종성 치환으로 만들 수 있는 모든 새 음절 후보.
        """
        cho, jung, jong = decompose_syllable(ch)

        candidates = []
        if cho in self.CHO_SUB:
            for new_cho in self.CHO_SUB[cho]:
                candidates.append(compose_syllable(new_cho, jung, jong))
        if jung in self.JUNG_SUB:
            for new_jung in self.JUNG_SUB[jung]:
                candidates.append(compose_syllable(cho, new_jung, jong))
        if jong in self.JONG_SUB:
            for new_jong in self.JONG_SUB[jong]:
                candidates.append(compose_syllable(cho, jung, new_jong))

        if not candidates:
            return ch
        return self.rng.choice(candidates)

    def attack_text(self, text: str) -> str:
        """문장 안에서 음소 치환 가능한 음절을 찾아 일부를 바꾼다."""
        chars = list(str(text))
        positions = [i for i, ch in enumerate(chars) if self._is_replaceable(ch)]
        selected = self._sample_positions(positions)

        for i in selected:
            chars[i] = self._replace_syllable(chars[i])

        return "".join(chars)
