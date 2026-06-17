"""
src/attacks/liaison.py

연음 역이용 공격.
받침이 있는 음절 뒤에 모음으로 시작하는 음절이 오면
발음 나는 대로 표기한다.

예시:
    먹어 → 머거
    닭이 → 달기
    밥을 → 바블
"""

from __future__ import annotations

from .base_attack import BaseAttack
from .hangul_utils import (
    decompose_syllable,
    compose_syllable,
    is_hangul_syllable,
)

# 연음으로 다음 음절 초성으로 이동할 수 있는 단일 받침 목록이다.
SINGLE_JONGS = [ "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ",
    "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]

# 겹받침은 두 자음 중 일부를 앞 음절 받침으로 남기고, 일부를 다음 음절 초성으로 보낼 수 있다.
# 값은 (앞 음절에 남길 받침, 뒤 음절 초성으로 보낼 자음) 순서이다.
CLUSTER_JONGS = {
    "ㄳ": ("ㄱ", "ㅅ"),
    "ㄵ": ("ㄴ", "ㅈ"),
    "ㄶ": ("ㄴ", "ㅎ"),
    "ㄺ": ("ㄹ", "ㄱ"),
    "ㄻ": ("ㄹ", "ㅁ"),
    "ㄼ": ("ㄹ", "ㅂ"),
    "ㄽ": ("ㄹ", "ㅅ"),
    "ㄾ": ("ㄹ", "ㅌ"),
    "ㄿ": ("ㄹ", "ㅍ"),
    "ㅀ": ("ㄹ", "ㅎ"),
    "ㅄ": ("ㅂ", "ㅅ"),
}

class LiaisonAttack(BaseAttack):
    """받침+모음 시작 음절을 발음 나는 대로 바꾸는 연음 역이용 공격."""

    # 결과 CSV의 attack_type 컬럼에 저장되는 이름이다.
    attack_type = "liaison"

    def _apply_liaison(self, ch1: str, ch2: str) -> tuple[str, str] | None:
        """
        ch1(받침 있는 음절) + ch2(ㅇ으로 시작하는 음절) 연음 처리
        받침을 다음 음절 초성으로 이동

        예: '먹' + '어' → '머' + '거'
        반환: 변환된 (ch1, ch2) 또는 변환 불가시 None
        """
        if not (is_hangul_syllable(ch1) and is_hangul_syllable(ch2)):
            return None

        # ch1은 앞 음절, ch2는 뒤 음절이다.
        # jong1이 뒤 음절 초성으로 이동할 수 있는지 확인한다.
        cho1, jung1, jong1 = decompose_syllable(ch1)
        cho2, jung2, jong2 = decompose_syllable(ch2)

        # 받침 없거나, 다음 음절 초성이 ㅇ이 아니면 연음 안 일어남
        if not jong1 or cho2 != "ㅇ":
            return None

        # 연음: 받침을 다음 음절 초성으로
        # 단자음 받침 처리
        
        if jong1 in SINGLE_JONGS:
            new_ch1 = compose_syllable(cho1, jung1, "")       # 받침 제거
            new_ch2 = compose_syllable(jong1, jung2, jong2)   # 받침을 초성으로

            return new_ch1, new_ch2
        
        if jong1 in CLUSTER_JONGS:
            remain_jong, new_cho = CLUSTER_JONGS[jong1]
            # 앞 자음은 받침으로 유지 , 뒤 자음은 다음 음절 초성으로 보냄.
            new_ch1 = compose_syllable(cho1,jung1,remain_jong)
            new_ch2 = compose_syllable(new_cho,jung2,jong2)
            return new_ch1, new_ch2
        
        return None

    def attack_text(self, text: str) -> str:
        """
        문장 안에서 연음 적용 가능한 인접 음절 쌍을 찾아 일부를 바꾼다.

        pair_positions:
            연음 적용이 가능한 앞 음절의 인덱스 목록.
            i가 들어 있으면 chars[i], chars[i+1] 쌍이 변형 대상이다.
        """
        chars = list(str(text))

        # 연음 가능한 위치 찾기 (i, i+1 쌍)
        pair_positions = []
        for i in range(len(chars) - 1):
            result = self._apply_liaison(chars[i], chars[i+1])
            if result is not None:
                pair_positions.append(i)

        # intensity만큼 선택
        selected = self._sample_positions(pair_positions)

        # 뒤에서부터 처리 (인덱스 안 밀리게)
        for i in sorted(selected, reverse=True):
            result = self._apply_liaison(chars[i], chars[i+1])
            if result:
                chars[i], chars[i+1] = result

        return "".join(chars)
