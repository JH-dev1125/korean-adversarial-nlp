"""
src/attacks/jamo_split.py

자모 분리 공격.
한글 음절을 초성+중성+종성으로 분리해서 표기한다.

예시:
    바보 → ㅂㅏㅂㅗ
    새끼 → ㅅㅐㄲㅣ
"""

from __future__ import annotations

from .base_attack import BaseAttack
from .hangul_utils import decompose_syllable, is_hangul_syllable


class JamoSplitAttack(BaseAttack):
    attack_type = "jamo"

    def _is_replaceable(self, ch: str) -> bool:
        # 한글 완성형 음절만 분리 가능
        return is_hangul_syllable(ch)

    def _split_syllable(self, ch: str) -> str:
        """
        음절을 자모로 분리
        예: '바' → 'ㅂㅏ', '밥' → 'ㅂㅏㅂ'
        """
        cho, jung, jong = decompose_syllable(ch)
        return cho + jung + jong  # 종성 없으면 jong=""이라 자동으로 안 붙음

    def attack_text(self, text: str) -> str:
        chars = list(str(text))
        positions = [i for i, ch in enumerate(chars) if self._is_replaceable(ch)]
        selected = self._sample_positions(positions)

        # 뒤에서부터 처리해야 인덱스 안 밀림
        # (한 글자가 여러 글자로 바뀌기 때문)
        for i in sorted(selected, reverse=True):
            split = self._split_syllable(chars[i])
            chars[i:i+1] = list(split)

        return "".join(chars)
