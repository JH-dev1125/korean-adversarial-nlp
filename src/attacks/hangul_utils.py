"""
src/attacks/hangul_utils.py

한글 음절 분해/조합 유틸리티.
jamo 라이브러리 없이 유니코드 공식을 사용하므로 Python 3.12.5에서 바로 사용 가능.
"""

from __future__ import annotations

CHOSEONG = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

JUNGSEONG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]

JONGSEONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

CHO_INDEX = {j: i for i, j in enumerate(CHOSEONG)}
JUNG_INDEX = {j: i for i, j in enumerate(JUNGSEONG)}
JONG_INDEX = {j: i for i, j in enumerate(JONGSEONG)}

HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3
N_JUNG = 21
N_JONG = 28


def is_hangul_syllable(ch: str) -> bool:
    return len(ch) == 1 and HANGUL_BASE <= ord(ch) <= HANGUL_END


def decompose_syllable(ch: str) -> tuple[str, str, str]:
    if not is_hangul_syllable(ch):
        raise ValueError(f"한글 완성형 음절이 아닙니다: {ch}")

    code = ord(ch) - HANGUL_BASE
    cho = code // (N_JUNG * N_JONG)
    jung = (code % (N_JUNG * N_JONG)) // N_JONG
    jong = code % N_JONG
    return CHOSEONG[cho], JUNGSEONG[jung], JONGSEONG[jong]


def compose_syllable(cho: str, jung: str, jong: str = "") -> str:
    if cho not in CHO_INDEX:
        raise ValueError(f"올바르지 않은 초성입니다: {cho}")
    if jung not in JUNG_INDEX:
        raise ValueError(f"올바르지 않은 중성입니다: {jung}")
    if jong not in JONG_INDEX:
        raise ValueError(f"올바르지 않은 종성입니다: {jong}")

    code = (
        HANGUL_BASE
        + CHO_INDEX[cho] * N_JUNG * N_JONG
        + JUNG_INDEX[jung] * N_JONG
        + JONG_INDEX[jong]
    )
    return chr(code)
