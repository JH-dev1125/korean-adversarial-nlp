"""
src/attacks/romanize.py

로마자 혼용 공격.
국립국어원 로마자 표기법을 단순화한 look-up table로 선택된 한글 음절을 영문자로 바꾼다.
"""

from __future__ import annotations

from .base_attack import BaseAttack
from .hangul_utils import decompose_syllable, is_hangul_syllable


class RomanizeAttack(BaseAttack):
    attack_type = "romanize"

    CHO_ROMA = {
        "ㄱ": "g", "ㄲ": "kk", "ㄴ": "n", "ㄷ": "d", "ㄸ": "tt",
        "ㄹ": "r", "ㅁ": "m", "ㅂ": "b", "ㅃ": "pp", "ㅅ": "s",
        "ㅆ": "ss", "ㅇ": "", "ㅈ": "j", "ㅉ": "jj", "ㅊ": "ch",
        "ㅋ": "k", "ㅌ": "t", "ㅍ": "p", "ㅎ": "h",
    }

    JUNG_ROMA = {
        "ㅏ": "a", "ㅐ": "ae", "ㅑ": "ya", "ㅒ": "yae", "ㅓ": "eo",
        "ㅔ": "e", "ㅕ": "yeo", "ㅖ": "ye", "ㅗ": "o", "ㅘ": "wa",
        "ㅙ": "wae", "ㅚ": "oe", "ㅛ": "yo", "ㅜ": "u", "ㅝ": "wo",
        "ㅞ": "we", "ㅟ": "wi", "ㅠ": "yu", "ㅡ": "eu", "ㅢ": "ui",
        "ㅣ": "i",
    }

    JONG_ROMA = {
        "": "", "ㄱ": "k", "ㄲ": "k", "ㄳ": "k", "ㄴ": "n", "ㄵ": "n",
        "ㄶ": "n", "ㄷ": "t", "ㄹ": "l", "ㄺ": "k", "ㄻ": "m",
        "ㄼ": "l", "ㄽ": "l", "ㄾ": "l", "ㄿ": "p", "ㅀ": "l",
        "ㅁ": "m", "ㅂ": "p", "ㅄ": "p", "ㅅ": "t", "ㅆ": "t",
        "ㅇ": "ng", "ㅈ": "t", "ㅊ": "t", "ㅋ": "k", "ㅌ": "t",
        "ㅍ": "p", "ㅎ": "t",
    }

    def _is_replaceable(self, ch: str) -> bool:
        return is_hangul_syllable(ch)

    def _romanize_syllable(self, ch: str) -> str:
        cho, jung, jong = decompose_syllable(ch)
        return self.CHO_ROMA[cho] + self.JUNG_ROMA[jung] + self.JONG_ROMA[jong]

    def attack_text(self, text: str) -> str:
        chars = list(str(text))
        positions = [i for i, ch in enumerate(chars) if self._is_replaceable(ch)]
        selected = self._sample_positions(positions)

        for i in selected:
            chars[i] = self._romanize_syllable(chars[i])

        return "".join(chars)
