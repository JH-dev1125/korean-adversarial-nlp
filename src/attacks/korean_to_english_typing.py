"""
src/attacks/korean_to_english_typing.py

영타 변환 공격.
한글 키보드 레이아웃 기준으로, 한글을 영문 자판으로 입력했을 때
나오는 문자열로 변환한다.

예시:
    시발  → tlqkf
    개새끼 → rPtPrl
    존나  → whssk
    병신  → qudtls

원리:
    한글 자모를 키보드 위치 기준으로 대응하는 영문자로 변환
    예: ㅅ → r (키보드에서 같은 위치)
        ㅣ → l
        ㅂ → q
        ㅏ → k
        ㄹ → f
"""

from __future__ import annotations

from .base_attack import BaseAttack
from .hangul_utils import decompose_syllable, is_hangul_syllable

# 한글 자모 → 영문자 키보드 매핑 (두벌식 기준)
JAMO_TO_ENG = {
    # 자음 (초성/종성)
    "ㄱ": "r", "ㄲ": "R", "ㄴ": "s", "ㄷ": "e", "ㄸ": "E",
    "ㄹ": "f", "ㅁ": "a", "ㅂ": "q", "ㅃ": "Q", "ㅅ": "t",
    "ㅆ": "T", "ㅇ": "d", "ㅈ": "w", "ㅉ": "W", "ㅊ": "c",
    "ㅋ": "z", "ㅌ": "x", "ㅍ": "v", "ㅎ": "g",

    # 모음
    "ㅏ": "k", "ㅐ": "o", "ㅑ": "i", "ㅒ": "O", "ㅓ": "j",
    "ㅔ": "p", "ㅕ": "u", "ㅖ": "P", "ㅗ": "h", "ㅘ": "hk",
    "ㅙ": "ho", "ㅚ": "hl", "ㅛ": "y", "ㅜ": "n", "ㅝ": "nj",
    "ㅞ": "np", "ㅟ": "nl", "ㅠ": "b", "ㅡ": "m", "ㅢ": "ml",
    "ㅣ": "l",

    # 복합 받침
    "ㄳ": "rt", "ㄵ": "sw", "ㄶ": "sg", "ㄺ": "fr", "ㄻ": "fa",
    "ㄼ": "fq", "ㄽ": "ft", "ㄾ": "fx", "ㄿ": "fv", "ㅀ": "fg",
    "ㅄ": "qt",
}


class KoreanToEngTypingAttack(BaseAttack):
    attack_type = "engtyping"

    def _convert_syllable(self, ch: str) -> str:
        """
        한글 음절 하나를 영타 문자열로 변환
        예: '시' → 'tl', '발' → 'qkf'
        """
        cho, jung, jong = decompose_syllable(ch)

        result = ""
        result += JAMO_TO_ENG.get(cho, "")   # 초성 변환
        result += JAMO_TO_ENG.get(jung, "")  # 중성 변환
        result += JAMO_TO_ENG.get(jong, "")  # 종성 변환 (없으면 "")

        return result if result else ch

    def _is_replaceable(self, ch: str) -> bool:
        return is_hangul_syllable(ch)

    def attack_text(self, text: str) -> str:
        chars = list(str(text))
        positions = [i for i, ch in enumerate(chars) if self._is_replaceable(ch)]
        selected = self._sample_positions(positions)

        # 뒤에서부터 처리 (한 글자가 여러 글자로 바뀌기 때문)
        for i in sorted(selected, reverse=True):
            converted = self._convert_syllable(chars[i])
            chars[i:i+1] = list(converted)

        return "".join(chars)
