# ====================================================
# src/evaluation/visualize.py
# 실험 결과 시각화 코드
#
# 이 파일이 하는 일:
#   1. results/metrics/ 의 CSV 파일들을 읽어서
#   2. 히트맵 (모델 × 공격유형 × 강도별 F1)
#   3. 막대그래프 (공격 유형별 ΔF1 비교)
#   4. 선 그래프 (강도별 성능 변화)
#   를 results/figures/ 에 저장
#
# 실행 방법:
#   python src/evaluation/visualize.py
# ====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
METRICS_DIR = BASE_DIR / "results" / "metrics"
FIGURES_DIR = BASE_DIR / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── 한글 폰트 설정 ────────────────────────────────────
# 운영체제별 한글 폰트 자동 설정
import platform
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":  # macOS
    plt.rcParams["font.family"] = "AppleGothic"
else:  # Linux (Kaggle/Colab)
    # 나눔폰트 설치 필요: apt-get install fonts-nanum
    try:
        plt.rcParams["font.family"] = "NanumGothic"
    except:
        plt.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# ── 공격 유형 한글 이름 매핑 ──────────────────────────
ATTACK_NAMES = {
    "none":       "원본",
    "phoneme":    "음소치환",
    "visual":     "시각유사",
    "romanize":   "로마자",
    "jamo":       "자모분리",
    "coda":       "받침조작",
    "liaison":    "연음역이용",
    "spacing":    "띄어쓰기",
    "emoji":      "이모지삽입",
    "engtyping":  "영타변환",
    "combined":   "복합공격",
}

# ── 모델 이름 매핑 ────────────────────────────────────
MODEL_NAMES = {
    "klue-bert":    "KLUE-BERT",
    "klue-roberta": "KLUE-RoBERTa",
    "kcbert":       "KCBERT",
}


# ====================================================
# 데이터 불러오기
# ====================================================
def load_all_results() -> pd.DataFrame:
    """
    results/metrics/ 에 있는 모든 모델 결과 CSV를 합쳐서 반환.
    """
    all_dfs = []
    for csv_file in METRICS_DIR.glob("*_results.csv"):
        df = pd.read_csv(csv_file)
        all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(
            f"결과 파일이 없습니다. 먼저 모델 학습을 완료해주세요.\n"
            f"경로: {METRICS_DIR}"
        )

    return pd.concat(all_dfs, ignore_index=True)


# ====================================================
# 그래프 1: 히트맵 (모델 × 공격유형, 강도별)
# ====================================================
def plot_heatmap(df: pd.DataFrame, intensity: float):
    """
    특정 강도에서 모델 × 공격유형 F1 히트맵 생성.

    Args:
        df: 전체 결과 데이터프레임
        intensity: 시각화할 강도 (0.1, 0.2, 0.3)
    """
    # 해당 강도 데이터만 필터링 (원본 포함)
    filtered = df[
        (df["intensity"] == intensity) | (df["attack_type"] == "none")
    ].copy()

    # 피벗 테이블 생성 (행=모델, 열=공격유형)
    pivot = filtered.pivot_table(
        index="model",
        columns="attack_type",
        values="f1",
        aggfunc="mean"
    )

    # 컬럼/인덱스 이름 한글로 변환
    pivot.columns = [ATTACK_NAMES.get(c, c) for c in pivot.columns]
    pivot.index = [MODEL_NAMES.get(i, i) for i in pivot.index]

    # 히트맵 그리기
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        pivot,
        annot=True,          # 각 셀에 수치 표시
        fmt=".3f",           # 소수점 3자리
        cmap="RdYlGn",       # 빨강(낮음) → 노랑 → 초록(높음)
        vmin=0.3,
        vmax=0.9,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "F1 Score"},
    )

    ax.set_title(f"모델별 공격 유형 탐지 성능 (강도 {int(intensity*100)}%)", fontsize=14, pad=15)
    ax.set_xlabel("공격 유형", fontsize=11)
    ax.set_ylabel("모델", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_path = FIGURES_DIR / f"heatmap_f1_intensity_{int(intensity*100)}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"저장: {save_path}")


# ====================================================
# 그래프 2: 막대그래프 (공격 유형별 ΔF1)
# ====================================================
def plot_delta_f1_bar(df: pd.DataFrame):
    """
    공격 유형별 평균 ΔF1 막대그래프.
    ΔF1이 클수록 그 공격에 취약한 것.
    """
    # 원본 제외, 강도 0.2 기준
    filtered = df[
        (df["attack_type"] != "none") &
        (df["intensity"] == 0.2)
    ].copy()

    # 공격 유형별, 모델별 평균 ΔF1
    grouped = filtered.groupby(["attack_type", "model"])["delta_f1"].mean().reset_index()
    grouped["attack_name"] = grouped["attack_type"].map(ATTACK_NAMES)
    grouped["model_name"] = grouped["model"].map(MODEL_NAMES)

    # 공격 유형별 전체 평균으로 정렬 (ΔF1 큰 순)
    order = grouped.groupby("attack_name")["delta_f1"].mean().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(14, 6))

    # 모델별 색상
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    models = grouped["model_name"].unique()

    x = np.arange(len(order))
    width = 0.25

    for i, (model, color) in enumerate(zip(models, colors)):
        model_data = grouped[grouped["model_name"] == model]
        model_data = model_data.set_index("attack_name").reindex(order)
        ax.bar(
            x + i * width,
            model_data["delta_f1"].values,
            width,
            label=model,
            color=color,
            alpha=0.85,
        )

    ax.set_title("공격 유형별 성능 저하 폭 ΔF1 (강도 20%)", fontsize=14, pad=15)
    ax.set_xlabel("공격 유형", fontsize=11)
    ax.set_ylabel("ΔF1 (클수록 취약)", fontsize=11)
    ax.set_xticks(x + width)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()

    save_path = FIGURES_DIR / "bar_delta_f1_by_attack.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"저장: {save_path}")


# ====================================================
# 그래프 3: 선 그래프 (강도별 성능 변화)
# ====================================================
def plot_intensity_line(df: pd.DataFrame):
    """
    공격 강도(10%, 20%, 30%)에 따른 F1 변화 선 그래프.
    """
    filtered = df[df["attack_type"] != "none"].copy()
    filtered["attack_name"] = filtered["attack_type"].map(ATTACK_NAMES)
    filtered["model_name"] = filtered["model"].map(MODEL_NAMES)

    models = filtered["model_name"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(18, 6), sharey=True)

    for ax, model in zip(axes, models):
        model_data = filtered[filtered["model_name"] == model]
        grouped = model_data.groupby(["attack_name", "intensity"])["f1"].mean().reset_index()

        for attack in grouped["attack_name"].unique():
            attack_data = grouped[grouped["attack_name"] == attack]
            ax.plot(
                attack_data["intensity"] * 100,
                attack_data["f1"],
                marker="o",
                label=attack,
                linewidth=1.5,
            )

        ax.set_title(model, fontsize=12)
        ax.set_xlabel("공격 강도 (%)", fontsize=10)
        ax.set_ylabel("F1 Score", fontsize=10)
        ax.set_xticks([10, 20, 30])
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")

    plt.suptitle("공격 강도에 따른 탐지 성능 변화", fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = FIGURES_DIR / "line_f1_by_intensity.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"저장: {save_path}")


# ====================================================
# 그래프 4: 요약 표 (전체 결과 한눈에)
# ====================================================
def plot_summary_table(df: pd.DataFrame):
    """
    모델 × 공격유형 × 강도별 ΔF1 요약 표를 이미지로 저장.
    """
    filtered = df[df["attack_type"] != "none"].copy()
    filtered["attack_name"] = filtered["attack_type"].map(ATTACK_NAMES)
    filtered["model_name"] = filtered["model"].map(MODEL_NAMES)

    pivot = filtered.pivot_table(
        index=["model_name", "intensity"],
        columns="attack_name",
        values="delta_f1",
        aggfunc="mean"
    ).round(3)

    # CSV로도 저장
    save_csv = METRICS_DIR / "summary_delta_f1.csv"
    pivot.to_csv(save_csv, encoding="utf-8-sig")
    print(f"요약 표 저장: {save_csv}")


# ====================================================
# 메인 실행
# ====================================================
if __name__ == "__main__":
    print("결과 파일 불러오는 중...")
    df = load_all_results()
    print(f"총 {len(df)}개 결과 로드 완료\n")

    print("그래프 생성 중...")

    # 강도별 히트맵 3개
    for intensity in [0.1, 0.2, 0.3]:
        plot_heatmap(df, intensity)

    # 공격 유형별 ΔF1 막대그래프
    plot_delta_f1_bar(df)

    # 강도별 성능 변화 선 그래프
    plot_intensity_line(df)

    # 요약 표
    plot_summary_table(df)

    print(f"\n✅ 완료! 저장 위치: {FIGURES_DIR}")
