# ====================================================
# src/evaluation/visualize.py
# 논문 삽입용 실험 결과 시각화 코드
#
# 실행 방법:
#   python src/evaluation/visualize.py
#
# 입력:
#   results/metrics/*_results.csv
#
# 출력:
#   results/figures/fig1_delta_f1_by_attack.{png,pdf}
#   results/figures/fig2_delta_f1_heatmap.{png,pdf}
#   results/figures/fig3_delta_f1_by_intensity.{png,pdf}
#   results/figures/fig4_fn_rate_by_attack.{png,pdf}
#   results/metrics/summary_delta_f1.csv
#   results/metrics/summary_fn_rate.csv
# ====================================================

from __future__ import annotations

import platform
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns


# 현재 파일 위치는 src/evaluation/visualize.py 이므로 parents[2]가 프로젝트 루트이다.
BASE_DIR = Path(__file__).resolve().parents[2]

# METRICS_DIR에는 evaluate_small_model.py가 만든 *_results.csv가 들어 있다.
METRICS_DIR = BASE_DIR / "results" / "metrics"

# FIGURES_DIR에는 논문 삽입용 PNG/PDF 그래프를 저장한다.
FIGURES_DIR = BASE_DIR / "results" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# DPI는 논문/보고서에 넣기 좋은 고해상도 저장을 위한 값이다.
DPI = 300

# 막대그래프와 heatmap에서 기본으로 보여줄 공격 강도이다.
DEFAULT_INTENSITY = 0.2


# 공격 유형 코드명을 한국어 표시 이름으로 바꾸는 표이다.
ATTACK_NAMES = {
    "none": "원본",
    "phoneme": "음소 치환",
    "visual": "시각 유사",
    "romanize": "로마자 혼용",
    "jamo": "자모 분리",
    "coda": "받침 조작",
    "liaison": "연음 역이용",
    "spacing": "띄어쓰기",
    "emoji": "특수문자 삽입",
    "engtyping": "영타 변환",
    "combined": "복합 공격",
}

# Kaggle/Linux 환경에서 한글 폰트를 찾지 못하면 영어 표시 이름으로 대체한다.
ATTACK_NAMES_EN = {
    "none": "Original",
    "phoneme": "Phoneme",
    "visual": "Visual",
    "romanize": "Romanization",
    "jamo": "Jamo split",
    "coda": "Coda",
    "liaison": "Liaison",
    "spacing": "Spacing",
    "emoji": "Symbol",
    "engtyping": "Eng. typing",
    "combined": "Combined",
}

# 결과 CSV의 model 값을 그래프 표시용 모델명으로 바꾸는 표이다.
MODEL_NAMES = {
    "klue-bert": "KLUE-BERT",
    "klue-roberta": "KLUE-RoBERTa",
    "kcbert": "KCBERT",
}

# 그림 제목/축 라벨을 언어별로 관리한다.
# FIGURE_LANG가 "ko"이면 한국어, "en"이면 영어 문구를 사용한다.
TEXT = {
    "ko": {
        "figure1_title": "공격 유형별 F1 감소폭 (강도 {intensity}%)",
        "figure3_title": "공격 강도별 평균 F1 감소폭",
        "figure4_title": "공격 유형별 위음성률 (강도 {intensity}%)",
        "attack_type": "공격 유형",
        "model": "모델",
        "delta_f1": "F1 감소폭 (원본 F1 - 공격 F1)",
        "delta_f1_short": "F1 감소폭",
        "mean_delta_f1": "평균 F1 감소폭",
        "intensity": "공격 강도",
        "fn_rate": "위음성률 FN / (TP + FN)",
    },
    "en": {
        "figure1_title": "F1 Drop by Attack Type (Intensity {intensity}%)",
        "figure3_title": "Mean F1 Drop by Attack Intensity",
        "figure4_title": "False Negative Rate by Attack Type (Intensity {intensity}%)",
        "attack_type": "Attack type",
        "model": "Model",
        "delta_f1": "F1 drop (Original F1 - Attacked F1)",
        "delta_f1_short": "F1 drop",
        "mean_delta_f1": "Mean F1 drop",
        "intensity": "Attack intensity",
        "fn_rate": "False negative rate",
    },
}

# 실제 사용할 그림 언어와 공격 표시 이름.
# setup_plot_style()에서 운영체제/폰트 상황을 보고 값이 바뀔 수 있다.
FIGURE_LANG = "ko"
ACTIVE_ATTACK_NAMES = ATTACK_NAMES

# 그래프에서 모델과 공격 유형을 일관된 순서로 보여주기 위한 기준 목록이다.
MODEL_ORDER = ["klue-bert", "klue-roberta", "kcbert"]
ATTACK_ORDER = [
    "phoneme",
    "visual",
    "romanize",
    "jamo",
    "coda",
    "liaison",
    "spacing",
    "emoji",
    "engtyping",
    "combined",
]


def setup_plot_style() -> None:
    """
    논문 그림에 맞는 공통 스타일을 설정한다.
    """
    # seaborn의 paper context는 글자 크기와 선 두께가 논문 그림에 맞게 작아진다.
    sns.set_theme(style="whitegrid", context="paper")

    global FIGURE_LANG, ACTIVE_ATTACK_NAMES

    # 운영체제별 기본 한글 폰트를 선택한다.
    # Kaggle/Linux에서 NanumGothic이 없으면 영어 그림으로 자동 전환한다.
    if platform.system() == "Windows":
        font_family = "Malgun Gothic"
        FIGURE_LANG = "ko"
    elif platform.system() == "Darwin":
        font_family = "AppleGothic"
        FIGURE_LANG = "ko"
    else:
        try:
            fm.findfont("NanumGothic", fallback_to_default=False)
            font_family = "NanumGothic"
            FIGURE_LANG = "ko"
        except ValueError:
            font_family = "DejaVu Sans"
            FIGURE_LANG = "en"

    # 환경변수 FIGURE_LANG=ko 또는 en을 주면 자동 선택 결과를 덮어쓸 수 있다.
    if os.environ.get("FIGURE_LANG") in {"ko", "en"}:
        FIGURE_LANG = os.environ["FIGURE_LANG"]

    ACTIVE_ATTACK_NAMES = ATTACK_NAMES if FIGURE_LANG == "ko" else ATTACK_NAMES_EN

    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.sans-serif"] = [font_family]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["figure.dpi"] = DPI
    plt.rcParams["savefig.dpi"] = DPI
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 9


def text(key: str, **kwargs) -> str:
    """
    현재 그림 언어에 맞는 문구를 반환한다.
    """
    # kwargs는 제목의 {intensity} 같은 자리표시자를 채우는 값이다.
    return TEXT[FIGURE_LANG][key].format(**kwargs)


def save_figure(fig: plt.Figure, name: str) -> None:
    """
    같은 그림을 PNG와 PDF로 함께 저장한다.
    """
    # PNG는 보고서/발표자료에 바로 넣기 좋고, PDF는 벡터 기반 문서 편집에 유리하다.
    png_path = FIGURES_DIR / f"{name}.png"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"저장: {png_path}")
    print(f"저장: {pdf_path}")


def load_all_results() -> pd.DataFrame:
    """
    최종 결과 CSV만 읽는다. 중간 저장 파일인 *_partial.csv는 제외한다.
    """
    # *_partial.csv는 실험 중간 저장 파일이므로 최종 그림에서는 제외한다.
    csv_files = sorted(
        path
        for path in METRICS_DIR.glob("*_results.csv")
        if "_partial" not in path.stem
    )

    if not csv_files:
        raise FileNotFoundError(
            f"결과 파일이 없습니다. 먼저 모델 학습/평가를 완료해주세요.\n"
            f"경로: {METRICS_DIR}"
        )

    # dfs에는 모델별 결과 CSV를 DataFrame으로 읽은 값들이 들어간다.
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df["source_file"] = csv_file.name
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    return normalize_results(result)


def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    그래프에 필요한 컬럼과 표시용 이름을 정리한다.
    """
    # 그래프 생성에 반드시 필요한 컬럼이다.
    required = {"model", "attack_type", "intensity", "f1", "delta_f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"결과 CSV에 필요한 컬럼이 없습니다: {sorted(missing)}")

    # 원본 DataFrame을 직접 바꾸지 않도록 복사본에서 정리한다.
    df = df.copy()
    df["intensity"] = df["intensity"].astype(float)
    df["delta_f1"] = df["delta_f1"].astype(float)
    df["f1"] = df["f1"].astype(float)

    # 예전 결과 CSV에는 fn_rate가 없을 수 있으므로 없는 경우 NaN으로 채운다.
    if "fn_rate" in df.columns:
        df["fn_rate"] = df["fn_rate"].astype(float)
    else:
        df["fn_rate"] = np.nan

    df["model_name"] = df["model"].map(MODEL_NAMES).fillna(df["model"])
    df["attack_name"] = df["attack_type"].map(ACTIVE_ATTACK_NAMES).fillna(df["attack_type"])

    # 실제 결과에 존재하는 모델/공격만 categorical 순서에 넣는다.
    model_order = [m for m in MODEL_ORDER if m in df["model"].unique()]
    attack_order = [a for a in ATTACK_ORDER if a in df["attack_type"].unique()]

    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df["attack_type"] = pd.Categorical(df["attack_type"], categories=attack_order + ["none"], ordered=True)
    return df.sort_values(["model", "attack_type", "intensity"])


def get_attack_order(df: pd.DataFrame, value_col: str, intensity: float = DEFAULT_INTENSITY) -> list[str]:
    """
    특정 지표 평균값 기준으로 공격 유형을 정렬한다.
    """
    # 원본 none 행은 공격 유형 정렬에서 제외한다.
    filtered = df[
        (df["attack_type"].astype(str) != "none")
        & (df["intensity"] == intensity)
    ].copy()

    if filtered.empty:
        filtered = df[df["attack_type"].astype(str) != "none"].copy()

    # value_col 평균이 큰 공격부터 정렬한다.
    # delta_f1이면 성능 하락이 큰 공격, fn_rate이면 놓친 비율이 큰 공격이 앞에 온다.
    order = (
        filtered.groupby("attack_type", observed=False)[value_col]
        .mean()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    return [attack for attack in order if attack != "none"]


def plot_delta_f1_by_attack(df: pd.DataFrame, intensity: float = DEFAULT_INTENSITY) -> None:
    """
    공격 유형별 F1 감소폭을 모델별 막대그래프로 저장한다.
    """
    # filtered는 특정 강도에서 원본(none)을 제외한 공격 결과만 담는다.
    filtered = df[
        (df["attack_type"].astype(str) != "none")
        & (df["intensity"] == intensity)
    ].copy()

    if filtered.empty:
        print(f"강도 {intensity} 데이터가 없어 fig1을 건너뜁니다.")
        return

    # order는 평균 delta_f1이 큰 공격부터 정렬한 공격 유형 순서이다.
    order = get_attack_order(df, "delta_f1", intensity)
    # attack_label은 그래프 x축에 표시할 사람이 읽기 쉬운 공격명이다.
    filtered["attack_label"] = filtered["attack_type"].astype(str).map(ACTIVE_ATTACK_NAMES)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    sns.barplot(
        data=filtered,
        x="attack_label",
        y="delta_f1",
        hue="model_name",
        order=[ACTIVE_ATTACK_NAMES.get(a, a) for a in order],
        palette="Set2",
        ax=ax,
    )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(text("figure1_title", intensity=int(intensity * 100)))
    ax.set_xlabel(text("attack_type"))
    ax.set_ylabel(text("delta_f1"))
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title=text("model"), frameon=True)
    fig.tight_layout()

    save_figure(fig, "fig1_delta_f1_by_attack")





def plot_delta_f1_by_intensity(df: pd.DataFrame) -> None:
    """
    공격 강도 증가에 따른 평균 F1 감소폭을 선 그래프로 저장한다.
    """
    # 모든 공격 유형을 대상으로 강도별 평균 delta_f1을 본다.
    filtered = df[df["attack_type"].astype(str) != "none"].copy()
    if filtered.empty:
        print("공격 데이터가 없어 fig2를 건너뜁니다.")
        return

    # grouped는 모델별/강도별로 여러 공격 유형의 delta_f1을 평균낸 표이다.
    grouped = (
        filtered.groupby(["model_name", "intensity"], observed=False)["delta_f1"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    sns.lineplot(
        data=grouped,
        x="intensity",
        y="delta_f1",
        hue="model_name",
        marker="o",
        linewidth=2,
        palette="Set2",
        ax=ax,
    )

    # x축 눈금은 0.1, 0.2, 0.3을 10%, 20%, 30%처럼 표시한다.
    intensities = sorted(grouped["intensity"].unique())
    ax.set_xticks(intensities)
    ax.set_xticklabels([f"{int(x * 100)}%" for x in intensities])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(text("figure3_title"))
    ax.set_xlabel(text("intensity"))
    ax.set_ylabel(text("mean_delta_f1"))
    ax.legend(title=text("model"), frameon=True)
    fig.tight_layout()

    save_figure(fig, "fig2_delta_f1_by_intensity")


def plot_fn_rate_by_attack(df: pd.DataFrame, intensity: float = DEFAULT_INTENSITY) -> None:
    """
    공격 유형별 위음성률(FN rate)을 모델별 막대그래프로 저장한다.
    """
    if df["fn_rate"].isna().all():
        print("fn_rate 컬럼이 없어 fig3을 건너뜁니다.")
        return

    # filtered는 특정 강도에서 공격 유형별 위음성률을 비교하기 위한 표이다.
    filtered = df[
        (df["attack_type"].astype(str) != "none")
        & (df["intensity"] == intensity)
    ].copy()

    if filtered.empty:
        print(f"강도 {intensity} 데이터가 없어 fig3을 건너뜁니다.")
        return

    # 위음성률이 큰 공격부터 x축에 배치한다.
    order = get_attack_order(df, "fn_rate", intensity)
    filtered["attack_label"] = filtered["attack_type"].astype(str).map(ACTIVE_ATTACK_NAMES)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    sns.barplot(
        data=filtered,
        x="attack_label",
        y="fn_rate",
        hue="model_name",
        order=[ACTIVE_ATTACK_NAMES.get(a, a) for a in order],
        palette="Set2",
        ax=ax,
    )

    ax.set_title(text("figure4_title", intensity=int(intensity * 100)))
    ax.set_xlabel(text("attack_type"))
    ax.set_ylabel(text("fn_rate"))
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title=text("model"), frameon=True)
    fig.tight_layout()

    save_figure(fig, "fig3_fn_rate_by_attack")


def save_summary_tables(df: pd.DataFrame) -> None:
    """
    논문 표 작성에 바로 쓸 수 있는 요약 CSV를 저장한다.
    """
    # 요약 표에서는 원본 none 행을 제외하고 공격 결과만 사용한다.
    filtered = df[df["attack_type"].astype(str) != "none"].copy()

    # delta_summary는 논문 표에 넣기 쉬운 모델 x 강도 x 공격유형 형태의 F1 감소폭 표이다.
    delta_summary = filtered.pivot_table(
        index=["model_name", "intensity"],
        columns="attack_name",
        values="delta_f1",
        aggfunc="mean",
        observed=False,
    ).round(4)

    delta_path = METRICS_DIR / "summary_delta_f1.csv"
    delta_summary.to_csv(delta_path, encoding="utf-8-sig")
    print(f"요약 표 저장: {delta_path}")

    # fn_rate가 있는 결과 CSV라면 위음성률 요약 표도 함께 저장한다.
    if not filtered["fn_rate"].isna().all():
        fn_summary = filtered.pivot_table(
            index=["model_name", "intensity"],
            columns="attack_name",
            values="fn_rate",
            aggfunc="mean",
            observed=False,
        ).round(4)

        fn_path = METRICS_DIR / "summary_fn_rate.csv"
        fn_summary.to_csv(fn_path, encoding="utf-8-sig")
        print(f"요약 표 저장: {fn_path}")


def main() -> None:
    setup_plot_style()

    print("결과 파일 불러오는 중...")
    df = load_all_results()
    print(f"총 {len(df)}개 결과 로드 완료")

    print("논문용 그래프 생성 중...")
    plot_delta_f1_by_attack(df, DEFAULT_INTENSITY)
    plot_delta_f1_by_intensity(df)
    plot_fn_rate_by_attack(df, DEFAULT_INTENSITY)
    save_summary_tables(df)

    print(f"\n완료! 저장 위치: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
