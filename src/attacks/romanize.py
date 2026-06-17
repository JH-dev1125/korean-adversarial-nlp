"""
src/attacks/romanize.py

로마자 혼용 공격.
국립국어원 로마자 표기법을 단순화한 look-up table로 선택된 한글 음절을 영문자로 바꾼다.
"""

from __future__ import annotations

from .base_attack import BaseAttack
from .hangul_utils import decompose_syllable, is_hangul_syllable


class RomanizeAttack(BaseAttack):
    """선택된 한글 음절을 로마자 또는 일부 로마자 혼용 형태로 바꾸는 공격."""

    # 결과 CSV의 attack_type 컬럼에 저장되는 이름이다.
    attack_type = "romanize"

    # 초성 로마자 표기표. 받침 위치가 아닌 초성 위치에서의 표기를 단순화해 사용한다.
    CHO_ROMA = {
        "ㄱ": "g", "ㄲ": "kk", "ㄴ": "n", "ㄷ": "d", "ㄸ": "tt",
        "ㄹ": "r", "ㅁ": "m", "ㅂ": "b", "ㅃ": "pp", "ㅅ": "s",
        "ㅆ": "ss", "ㅇ": "", "ㅈ": "j", "ㅉ": "jj", "ㅊ": "ch",
        "ㅋ": "k", "ㅌ": "t", "ㅍ": "p", "ㅎ": "h",
    }

    # 중성 로마자 표기표.
    JUNG_ROMA = {
        "ㅏ": "a", "ㅐ": "ae", "ㅑ": "ya", "ㅒ": "yae", "ㅓ": "eo",
        "ㅔ": "e", "ㅕ": "yeo", "ㅖ": "ye", "ㅗ": "o", "ㅘ": "wa",
        "ㅙ": "wae", "ㅚ": "oe", "ㅛ": "yo", "ㅜ": "u", "ㅝ": "wo",
        "ㅞ": "we", "ㅟ": "wi", "ㅠ": "yu", "ㅡ": "eu", "ㅢ": "ui",
        "ㅣ": "i",
    }

    # 종성 로마자 표기표. 실제 국어 로마자 표기법을 실험용으로 단순화했다.
    JONG_ROMA = {
        "": "", "ㄱ": "k", "ㄲ": "k", "ㄳ": "k", "ㄴ": "n", "ㄵ": "n",
        "ㄶ": "n", "ㄷ": "t", "ㄹ": "l", "ㄺ": "k", "ㄻ": "m",
        "ㄼ": "l", "ㄽ": "l", "ㄾ": "l", "ㄿ": "p", "ㅀ": "l",
        "ㅁ": "m", "ㅂ": "p", "ㅄ": "p", "ㅅ": "t", "ㅆ": "t",
        "ㅇ": "ng", "ㅈ": "t", "ㅊ": "t", "ㅋ": "k", "ㅌ": "t",
        "ㅍ": "p", "ㅎ": "t",
    }

    def _is_replaceable(self, ch: str) -> bool:
        """한글 완성형 음절이면 로마자 변환 대상이다."""
        return is_hangul_syllable(ch)

    def _romanize_syllable(self, ch: str) -> str:
        """음절 전체를 로마자 문자열로 바꾼다. 예: '바' -> 'ba'."""
        cho, jung, jong = decompose_syllable(ch)
        return self.CHO_ROMA[cho] + self.JUNG_ROMA[jung] + self.JONG_ROMA[jong]
    
    def _partial_romanize_syllable(self, ch: str) -> str:
        """
        초성/중성/종성 중 하나만 로마자로 바꾼다.

        parts:
            ("cho", 초성로마자, 원래초성)처럼 어느 부분을 바꿀 수 있는지 저장한 후보 목록.
        target:
            이번에 실제로 바꿀 부분 이름. "cho", "jung", "jong" 중 하나이다.
        """
        cho, jung, jong = decompose_syllable(ch)

        parts = []

        # 초성 후보
        parts.append(("cho", self.CHO_ROMA[cho], cho))

        # 중성 후보
        parts.append(("jung", self.JUNG_ROMA[jung], jung))

        # 종성 후보: 받침이 있을 때만
        if jong != "":
            parts.append(("jong", self.JONG_ROMA[jong], jong))

        target, roma, original = self.rng.choice(parts)

        if target == "cho":
            return roma + jung + jong

        if target == "jung":
            return cho + roma + jong

        if target == "jong":
            return cho + jung + roma

        return ch

    def attack_text(self, text: str) -> str:
        """문장 안의 한글 음절 일부를 전체 로마자 또는 부분 로마자로 바꾼다."""
        chars = list(str(text))
        positions = [i for i, ch in enumerate(chars) if self._is_replaceable(ch)]
        selected = self._sample_positions(positions)

        for i in selected:
            # 50% 확률로 음절 전체를 로마자화하고, 나머지는 일부 자모만 로마자화한다.
            if self.rng.random() < 0.5:
                chars[i] = self._romanize_syllable(chars[i])
            else:
                chars[i] = self._partial_romanize_syllable(chars[i])

        return "".join(chars)
