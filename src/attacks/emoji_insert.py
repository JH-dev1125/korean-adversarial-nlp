"""
src/attacks/emoji_insert.py

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
    "★", "☆", "♥", "♡", "■", "□", "●", "○",
    "▶", "◀", "♠", "♣", "♦", "※", "▲", "▼",
    "😊", "😂", "🤣", "👍", "🙏", "😍", "🔥", "💯",
    "^", "*", "~", "!", "?", "#", "@", "%",
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

        selected = self._sample_positions(positions)

        # 뒤에서부터 삽입 (인덱스 안 밀리게)
        for i in sorted(selected, reverse=True):
            insert_char = self.rng.choice(INSERT_CANDIDATES)
            chars.insert(i + 1, insert_char)

        return "".join(chars)
