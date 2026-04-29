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


class LiaisonAttack(BaseAttack):
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

        cho1, jung1, jong1 = decompose_syllable(ch1)
        cho2, jung2, jong2 = decompose_syllable(ch2)

        # 받침 없거나, 다음 음절 초성이 ㅇ이 아니면 연음 안 일어남
        if not jong1 or cho2 != "ㅇ":
            return None

        # 연음: 받침을 다음 음절 초성으로
        # jong1이 단자음인지 확인 (복합 받침은 처리 복잡해서 제외)
        single_jongs = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
        if jong1 not in single_jongs:
            return None

        new_ch1 = compose_syllable(cho1, jung1, "")       # 받침 제거
        new_ch2 = compose_syllable(jong1, jung2, jong2)   # 받침을 초성으로

        return new_ch1, new_ch2

    def attack_text(self, text: str) -> str:
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
