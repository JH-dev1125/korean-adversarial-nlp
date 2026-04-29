"""
src/attacks/spacing.py

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
    attack_type = "spacing"

    def attack_text(self, text: str) -> str:
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
        indices = list(range(len(all_positions)))
        selected_indices = self._sample_positions(indices)
        selected = [all_positions[i] for i in selected_indices]

        # 뒤에서부터 처리
        selected_sorted = sorted(selected, key=lambda x: x[1], reverse=True)

        for action, i in selected_sorted:
            if action == "add":
                # i와 i+1 사이에 공백 삽입
                chars.insert(i + 1, " ")
            elif action == "remove" and i < len(chars) and chars[i] == " ":
                # 공백 제거
                chars.pop(i)

        return "".join(chars)
