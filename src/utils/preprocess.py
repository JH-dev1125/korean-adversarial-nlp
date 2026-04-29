# ====================================================
# src/utils/preprocess.py
# 세 데이터셋 전처리 및 통합 코드
#
# 이 파일이 하는 일:
#   1. data/raw/ 에서 세 데이터셋 불러오기
#   2. 각 데이터셋 라벨을 혐오(1)/정상(0)으로 통일
#   3. 세 데이터셋 합치기
#   4. 80/10/10으로 분할
#   5. data/processed/ 에 CSV로 저장
#
# 실행 방법:
#   python src/utils/preprocess.py
#
# 라벨 처리 방식:
#   K-HATERS: Normal → 0, 나머지(L1_hate, L2_hate, Offensive) → 1
#   KOLD:     OFF=False → 0, OFF=True → 1
#   K-MHaS:  Politics, Gender, Profanity, Origin, Age 중
#             하나라도 있으면 → 1, 나머지 → 0
# ====================================================

import os
import pandas as pd
from datasets import load_from_disk
from sklearn.model_selection import train_test_split

# ── 경로 설정 ─────────────────────────────────────────
RAW_DIR = "data/raw"
SAVE_DIR = "data/processed"
os.makedirs(SAVE_DIR, exist_ok=True)

# K-MHaS에서 혐오(1)로 처리할 카테고리 번호
# 0: Politics, 1: Origin, 2: Physical, 3: Age,
# 4: Gender, 5: Religion, 6: Race, 7: Profanity, 8: Not Hate Speech
KMHAS_HATE_LABELS = {0, 1, 3, 4, 7}  # Politics, Origin, Age, Gender, Profanity


# ====================================================
# K-HATERS 전처리
# ====================================================
def process_khaters():
    """
    K-HATERS 데이터셋 전처리

    원본 라벨:
        - 'Normal'   → 정상(0)
        - 'L1_hate'  → 혐오(1) (명시적 혐오)
        - 'L2_hate'  → 혐오(1) (암묵적 혐오)
        - 'Offensive' → 혐오(1) (공격적 표현)
    """
    print("K-HATERS 전처리 중...")

    ds = load_from_disk(f"{RAW_DIR}/khaters")

    all_data = []
    for split in ds.keys():
        df = pd.DataFrame(ds[split])
        all_data.append(df)
    df = pd.concat(all_data, ignore_index=True)

    # text 컬럼 확인 (K-HATERS는 'comment' 또는 'text')
    if "comment" in df.columns:
        df = df.rename(columns={"comment": "text"})

    # 라벨 변환: Normal=0, 나머지=1
    df["label"] = df["label"].apply(
        lambda x: 0 if str(x).lower() == "normal" else 1
    )

    # 필요한 컬럼만 선택 + source 추가
    df = df[["text", "label"]].copy()
    df["source"] = "khaters"

    # 빈 텍스트 제거
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]

    print(f"  총 {len(df)}개 | 정상: {(df['label']==0).sum()} | 혐오: {(df['label']==1).sum()}")
    return df


# ====================================================
# KOLD 전처리
# ====================================================
def process_kold():
    """
    KOLD 데이터셋 전처리

    원본 라벨:
        - OFF = False → 정상(0)
        - OFF = True  → 혐오(1)

    KOLD는 train split만 있어서 우리가 직접 분할함
    """
    print("KOLD 전처리 중...")

    ds = load_from_disk(f"{RAW_DIR}/kold")

    # KOLD는 train split만 있음
    df = pd.DataFrame(ds["train"])

    # text 컬럼 확인 (KOLD는 'comment' 컬럼)
    if "comment" in df.columns:
        df = df.rename(columns={"comment": "text"})

    # 라벨 변환: False=0, True=1
    df["label"] = df["OFF"].apply(lambda x: 1 if x else 0)

    # 필요한 컬럼만 선택 + source 추가
    df = df[["text", "label"]].copy()
    df["source"] = "kold"

    # 빈 텍스트 제거
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]

    print(f"  총 {len(df)}개 | 정상: {(df['label']==0).sum()} | 혐오: {(df['label']==1).sum()}")
    return df


# ====================================================
# K-MHaS 전처리
# ====================================================
def process_kmhas():
    """
    K-MHaS 데이터셋 전처리

    원본 라벨: 멀티라벨 형식 (리스트)
        0: Politics  (정치성향 차별)  → 혐오(1)
        1: Origin    (출신 차별)      → 혐오(1)
        2: Physical  (외모 차별)      → 정상(0) - 제외
        3: Age       (연령 차별)      → 혐오(1)
        4: Gender    (성별 차별)      → 혐오(1)
        5: Religion  (종교 차별)      → 정상(0) - 제외
        6: Race      (인종 차별)      → 정상(0) - 제외
        7: Profanity (혐오 욕설)      → 혐오(1)
        8: Not Hate Speech            → 정상(0)

    K-HATERS, KOLD와 겹치는 카테고리만 혐오(1)로 처리:
    Politics, Origin, Age, Gender, Profanity
    """
    print("K-MHaS 전처리 중...")

    ds = load_from_disk(f"{RAW_DIR}/kmhas")

    all_data = []
    for split in ds.keys():
        df = pd.DataFrame(ds[split])
        all_data.append(df)
    df = pd.concat(all_data, ignore_index=True)

    # 라벨 변환
    # label 컬럼이 리스트 형식: [0, 3, 7] 같은 형태
    def convert_label(label_list):
        # label_list가 문자열로 저장된 경우 변환
        if isinstance(label_list, str):
            label_list = eval(label_list)
        # KMHAS_HATE_LABELS와 겹치는 라벨이 하나라도 있으면 혐오(1)
        for l in label_list:
            if l in KMHAS_HATE_LABELS:
                return 1
        return 0

    df["label"] = df["label"].apply(convert_label)

    # 필요한 컬럼만 선택 + source 추가
    df = df[["text", "label"]].copy()
    df["source"] = "kmhas"

    # 빈 텍스트 제거
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]

    print(f"  총 {len(df)}개 | 정상: {(df['label']==0).sum()} | 혐오: {(df['label']==1).sum()}")
    return df


# ====================================================
# 세 데이터셋 통합 및 분할
# ====================================================
def merge_and_split(khaters_df, kold_df, kmhas_df):
    """
    세 데이터셋을 합치고 80/10/10으로 분할

    Args:
        khaters_df: K-HATERS 전처리 결과
        kold_df:    KOLD 전처리 결과
        kmhas_df:   K-MHaS 전처리 결과
    """
    print("\n세 데이터셋 통합 중...")

    # 합치기
    df = pd.concat([khaters_df, kold_df, kmhas_df], ignore_index=True)

    # 섞기 (random_state=42: 항상 같은 방식으로 섞기)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"통합 후 총 {len(df)}개")
    print(f"  정상(0): {(df['label']==0).sum()}")
    print(f"  혐오(1): {(df['label']==1).sum()}")

    # 1차 분할: 학습(80%) vs 나머지(20%)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]   # 혐오/정상 비율 유지
    )

    # 2차 분할: 검증(10%) vs 테스트(10%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["label"]
    )

    # CSV 저장
    train_df.to_csv(f"{SAVE_DIR}/train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(f"{SAVE_DIR}/val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(f"{SAVE_DIR}/test.csv", index=False, encoding="utf-8-sig")

    print(f"\n분할 완료!")
    print(f"  학습(train): {len(train_df)}개")
    print(f"  검증(val):   {len(val_df)}개")
    print(f"  테스트(test): {len(test_df)}개")
    print(f"\n저장 위치: {SAVE_DIR}/")


# ====================================================
# 메인 실행
# ====================================================
if __name__ == "__main__":
    print("=" * 50)
    print("전처리 시작")
    print("=" * 50)

    # 각 데이터셋 전처리
    khaters_df = process_khaters()
    kold_df = process_kold()
    kmhas_df = process_kmhas()

    # 통합 및 분할
    merge_and_split(khaters_df, kold_df, kmhas_df)

    print("\n" + "=" * 50)
    print("✅ 전처리 완료!")
    print("=" * 50)
