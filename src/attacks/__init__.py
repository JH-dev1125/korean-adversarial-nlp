"""
src/attacks 패키지 초기화 파일.

다른 코드에서 아래처럼 간단히 import할 수 있도록 공격 클래스들을 한곳에서 다시 내보낸다.
예:
    from src.attacks import JamoSplitAttack, CompoundAttack
"""

from .base_attack import BaseAttack
from .phoneme_sub import PhonemeSubAttack
from .visual_sub import VisualSubAttack
from .romanize import RomanizeAttack
from .jamo_split import JamoSplitAttack
from .coda_manip import CodaManipAttack
from .liaison import LiaisonAttack
from .spacing import SpacingAttack
from .emoji_insert import EmojiInsertAttack
from .korean_to_english_typing import KoreanToEngTypingAttack
from .compound_attack import CompoundAttack

# __all__은 from src.attacks import * 를 사용할 때 공개할 이름 목록이다.
# 프로젝트에서 지원하는 공격 클래스 전체를 명시해 자동완성/문서화에도 도움이 된다.
__all__ = [
    "BaseAttack",
    "PhonemeSubAttack",
    "VisualSubAttack",
    "RomanizeAttack",
    "JamoSplitAttack",
    "CodaManipAttack",
    "LiaisonAttack",
    "SpacingAttack",
    "EmojiInsertAttack",
    "KoreanToEngTypingAttack",
    "CompoundAttack",
]
