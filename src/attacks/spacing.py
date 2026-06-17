"""
띄어쓰기 조작 공격.
단어 중간에 공백을 추가하거나 기존 공백을 제거한다.

예시:
    나쁜놈 → 나 쁜 놈  (공백 추가)
    나쁜 놈 → 나쁜놈    (공백 제거)
"""

from __future__ import annotations

from .base_attack import BaseAttack
from .hangul_utils import is_hangul_syllable


class SpacingAttack(BaseAttack):
    """한글 사이 공백을 추가하거나 기존 공백을 제거하는 공격."""

    # 결과 CSV의 attack_type 컬럼에 저장되는 이름이다.
    attack_type = "spacing"

    def attack_text(self, text: str) -> str:
        """
        문장 내 공백 조작 가능 위치를 찾아 일부만 변형한다.

        add_positions:
            ("add", i) 형태. chars[i]와 chars[i+1] 사이에 공백을 넣을 수 있다는 뜻이다.
        remove_positions:
            ("remove", i) 형태. chars[i]가 기존 공백이라 제거할 수 있다는 뜻이다.
        all_positions:
            추가 후보와 제거 후보를 합친 전체 조작 후보 목록.
        """
        chars = list(str(text))

        # 두 가지 조작 가능한 위치 찾기
        # 1. 공백 추가 가능: 한글 음절 사이 (공백 아닌 곳)
        # 2. 공백 제거 가능: 기존 공백 위치
        add_positions = []    # 공백 추가할 위치 (i와 i+1 사이)
        remove_positions = [] # 공백 제거할 위치

        for i in range(len(chars) - 1):
            if is_hangul_syllable(chars[i]) and is_hangul_syllable(chars[i+1]):
                add_positions.append(("add", i))
            elif chars[i] == " ":
                remove_positions.append(("remove", i))

        all_positions = add_positions + remove_positions

        if not all_positions:
            return text

        # intensity만큼 선택
        # _sample_positions는 정수 목록을 받으므로 all_positions 자체가 아니라 인덱스 목록을 넘긴다.
        indices = list(range(len(all_positions)))
        selected_indices = self._sample_positions(indices)
        selected = [all_positions[i] for i in selected_indices]

        # 뒤에서부터 처리해야 앞쪽 삽입/삭제 때문에 뒤쪽 인덱스가 밀리지 않는다.
        selected_sorted = sorted(selected, key=lambda x: x[1], reverse=True)

        for action, i in selected_sorted:
            if action == "add":
                # i와 i+1 사이에 공백 삽입
                chars.insert(i + 1, " ")
            elif action == "remove" and i < len(chars) and chars[i] == " ":
                # 공백 제거
                chars.pop(i)

        return "".join(chars)
