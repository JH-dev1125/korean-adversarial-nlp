"""
src/attacks/hangul_utils.py

한글 음절 분해/조합 유틸리티.
jamo 라이브러리 없이 유니코드 공식을 사용하므로 Python 3.12.5에서 바로 사용 가능.
"""

from __future__ import annotations

# 초성 목록. 유니코드 한글 완성형 공식에서 초성 인덱스 0~18에 대응한다.
CHOSEONG = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

# 중성 목록. 유니코드 한글 완성형 공식에서 중성 인덱스 0~20에 대응한다.
JUNGSEONG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]

# 종성 목록. 첫 번째 값 ""은 받침이 없는 경우를 뜻한다.
JONGSEONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

# 자모 문자열을 유니코드 조합 공식에 필요한 숫자 인덱스로 빠르게 바꾸기 위한 역방향 표이다.
CHO_INDEX = {j: i for i, j in enumerate(CHOSEONG)}
JUNG_INDEX = {j: i for i, j in enumerate(JUNGSEONG)}
JONG_INDEX = {j: i for i, j in enumerate(JONGSEONG)}

# 한글 완성형 음절은 '가'(0xAC00)부터 '힣'(0xD7A3)까지 연속 배치되어 있다.
HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3

# 유니코드 공식에서 한 초성마다 중성 21개, 각 중성마다 종성 28개가 배치된다.
N_JUNG = 21
N_JONG = 28

def is_hangul_syllable(ch: str) -> bool:
    """
    입력 문자가 한글 완성형 음절인지 확인한다.

    예:
        "가" -> True
        "ㄱ" -> False  # 자모 단독 문자는 완성형 음절이 아니다.
        "a" -> False
    """
    return len(ch) == 1 and HANGUL_BASE <= ord(ch) <= HANGUL_END

def decompose_syllable(ch: str) -> tuple[str, str, str]:
    """
    한글 완성형 음절 하나를 초성, 중성, 종성으로 분해한다.

    ch:
        길이 1의 한글 완성형 음절.
    반환값:
        (초성, 중성, 종성) 튜플. 받침이 없으면 종성은 ""이다.
    """
    if not is_hangul_syllable(ch):
        raise ValueError(f"한글 완성형 음절이 아닙니다: {ch}")
    
    # code는 '가'를 0으로 봤을 때 ch가 몇 번째 음절인지 나타내는 상대 위치이다.
    code = ord(ch) - HANGUL_BASE
    # 초성은 중성*종성 묶음 단위로 나뉜다.
    cho = code // (N_JUNG * N_JONG)
    # 중성은 한 초성 블록 안에서 종성 묶음 단위로 나뉜다.
    jung = (code % (N_JUNG * N_JONG)) // N_JONG
    # 종성은 가장 안쪽 인덱스이다.
    jong = code % N_JONG
    return CHOSEONG[cho], JUNGSEONG[jung], JONGSEONG[jong]

def compose_syllable(cho: str, jung: str, jong: str = "") -> str:
    """
    초성, 중성, 종성을 다시 한글 완성형 음절 하나로 조합한다.

    cho:
        CHOSEONG에 포함된 초성.
    jung:
        JUNGSEONG에 포함된 중성.
    jong:
        JONGSEONG에 포함된 종성. 받침이 없으면 "".
    """
    if cho not in CHO_INDEX:
        raise ValueError(f"올바르지 않은 초성입니다: {cho}")
    if jung not in JUNG_INDEX:
        raise ValueError(f"올바르지 않은 중성입니다: {jung}")
    if jong not in JONG_INDEX:
        raise ValueError(f"올바르지 않은 종성입니다: {jong}")

    # 유니코드 한글 조합 공식:
    # '가' + 초성인덱스*21*28 + 중성인덱스*28 + 종성인덱스
    code = (
        HANGUL_BASE
        + CHO_INDEX[cho] * N_JUNG * N_JONG
        + JUNG_INDEX[jung] * N_JONG
        + JONG_INDEX[jong]
    )
    return chr(code)
