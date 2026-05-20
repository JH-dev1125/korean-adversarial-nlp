"""
이모지/특수문자 삽입 공격.
한글 음절 사이에 이모지나 특수문자를 삽입한다.

예시:
    바보 → 바★보
    새끼 → 새😊끼
"""

from __future__ import annotations

from .base_attack import BaseAttack
from .hangul_utils import is_hangul_syllable

# 삽입할 이모지/특수문자 후보 목록
INSERT_CANDIDATES = [
    # 1순위: 키보드 기호 (가장 흔함)
    ".", ",", "-", "_", "|", "/", "~", "^",
    # 2순위: Shift+숫자 기호
    "!", "@", "#", "$", "%", "*", "+", "=", "?",
    # 3순위: 숫자
    "0", "1", "2",
    # 4순위: 한글 자판 특수문자
    "※", "·",
    # 5순위: 보이지 않는 문자 (실제 우회에서 많이 쓰임)
    "\u200b",  # Zero-width space
    "\u3164",  # 한글 채움 문자
    # 6순위: 기존 특수기호 중 현실적인 것만 유지
    "★", "♥", "▶",
]


class EmojiInsertAttack(BaseAttack):
    attack_type = "emoji"

    def attack_text(self, text: str) -> str:
        chars = list(str(text))

        # 한글 음절 사이 위치 찾기
        positions = []
        for i in range(len(chars) - 1):
            if is_hangul_syllable(chars[i]) and is_hangul_syllable(chars[i+1]):
                positions.append(i)

        if not positions:
            return text
        #positions중 랜덤 변형위치를 _sample_positions로 정하기
        selected = self._sample_positions(positions)

        # 뒤에서부터 삽입 (인덱스 안 밀리게)
        
        for i in sorted(selected, reverse=True):
            insert_char = self.rng.choice(INSERT_CANDIDATES)
            chars.insert(i + 1, insert_char)

        return "".join(chars)
