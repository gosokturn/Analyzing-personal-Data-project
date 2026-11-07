import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

from market_utils import (
    load_raw_data,
    prepare_market_data,
    build_market_features
)

st.set_page_config(
    page_title="세계 금융시장 한눈에 보기",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px;}
    [data-testid="stMetric"] {background: #ffffff; border: 1px solid #e8edf3;
        padding: 16px 18px; border-radius: 14px;}
    div[data-testid="stExpander"] {border: 1px solid #e8edf3; border-radius: 12px;}
    h1, h2, h3 {color: #172b4d;}
    .intro-card {background: #f2f6fb; padding: 22px 24px; border-radius: 16px;
        border: 1px solid #e3ebf5; margin-bottom: 18px;}
    .muted {color: #64748b; font-size: 0.92rem;}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR / "pages"


def home_page():
    st.title("🌍 세계 금융시장 한눈에 보기")
    st.markdown("""
    <div class="intro-card">
      <h3>여러 나라의 주식시장은 어떻게 움직일까요?</h3>
      <div>데이터를 통해 비슷한 시장을 찾고, 시장 사이의 관계와 위험 특성을 알아봅니다.
      전문 투자용 거래 프로그램이 아니라 금융 데이터를 이해하기 위한 학습 서비스입니다.</div>
    </div>
    """, unsafe_allow_html=True)

    try:
        raw = load_raw_data()
        clean, price_col, audit = prepare_market_data(raw, winsorize=False)
        features = build_market_features(clean, price_col, min_observations=60)
    except Exception as e:
        st.error("분석 데이터를 준비하지 못했습니다. 데이터 주소와 원본 CSV의 열 구성을 확인해 주세요.")
        with st.expander("오류 세부 정보"):
            st.exception(e)
        return

    dates = pd.to_datetime(clean["Date"], errors="coerce").dropna()
    market_count = clean["Index"].nunique()
    start_date = dates.min().date() if not dates.empty else None
    end_date = dates.max().date() if not dates.empty else None

    avg_vol = features["annualized_volatility"].mean() if not features.empty else np.nan

    a, b, c, d = st.columns(4)
    a.metric("분석 금융시장", f"{market_count:,}개")
    b.metric("분석 데이터 행", f"{len(clean):,}개")
    c.metric("분석 기간", f"{start_date} ~ {end_date}" if start_date and end_date else "확인 불가")
    d.metric("평균 연환산 변동성", f"{avg_vol:.1%}" if pd.notna(avg_vol) else "계산 불가",
             help="시장별 일별 수익률 표준편차를 연간 기준으로 환산한 값의 평균입니다. 가격이 얼마나 크게 흔들렸는지를 나타냅니다.")

    st.divider()
    st.subheader("데이터에서 살펴볼 수 있는 네 가지 질문")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧩 시장 유형은 어떻게 나뉠까?")
        if len(features) >= 3:
            model_cols = [x for x in [
                "annualized_return", "annualized_volatility", "downside_volatility",
                "max_drawdown", "positive_day_ratio", "average_absolute_return",
                "average_intraday_range"
            ] if x in features.columns and features[x].notna().any()]
            model_data = features[["Index"] + model_cols].dropna(subset=model_cols, how="all")
            if len(model_data) >= 3 and len(model_cols) >= 2:
                X = SimpleImputer(strategy="median").fit_transform(model_data[model_cols])
                X = RobustScaler().fit_transform(X)
                k = min(3, len(model_data)-1)
                labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X)
                st.metric("기본 설정에서 나뉜 그룹", f"{k}개")
                st.write(f"현재 데이터에서 {len(model_data)}개 시장을 선택한 금융 지표의 유사성에 따라 {k}개 그룹으로 나누어 볼 수 있습니다.")
            else:
                st.info("군집화에 필요한 시장별 특징값이 충분하지 않습니다.")
        else:
            st.info("시장별 관측값이 충분하지 않아 군집 요약을 계산할 수 없습니다.")

        st.markdown("#### 📉 변동성이 가장 큰 시장은?")
        if not features.empty and features["annualized_volatility"].notna().any():
            highest = features.loc[features["annualized_volatility"].idxmax()]
            st.write(f"**{highest['Index']}** 시장의 연환산 변동성이 가장 높게 계산되었습니다({highest['annualized_volatility']:.1%}).")
            st.caption("과거 분석 기간의 변동성 비교이며, 미래의 위험을 확정하는 값은 아닙니다.")
        else:
            st.info("변동성을 비교할 수 있는 시장이 없습니다.")

    with col2:
        st.markdown("#### 🔗 서로 비슷하게 움직이는 시장은?")
        pivot = clean.pivot_table(index="Date", columns="Index",
                                  values="daily_return_used", aggfunc="mean")
        valid = pivot.columns[pivot.count() >= 60].tolist()
        corr = pivot[valid].corr(min_periods=60) if len(valid) >= 2 else pd.DataFrame()
        if not corr.empty:
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().dropna()
            if not upper.empty:
                pair = upper.abs().idxmax()
                value = corr.loc[pair[0], pair[1]]
                direction = "같은 방향" if value >= 0 else "반대 방향"
                st.write(f"**{pair[0]}**과 **{pair[1]}**의 상관계수 절댓값이 가장 컸습니다({value:.2f}). 두 시장은 분석 기간에 {direction}으로 움직이는 경향이 상대적으로 강했습니다.")
                st.caption("상관관계는 한 시장이 다른 시장을 움직이게 했다는 뜻은 아닙니다.")
            else:
                st.info("비교 가능한 시장 쌍이 없습니다.")
        else:
            st.info("상관관계를 계산할 만큼 데이터가 겹치는 시장이 부족합니다.")

        st.markdown("#### 🧭 최근 데이터에서 확인할 점")
        if not clean.empty:
            latest = clean["Date"].max()
            latest_rows = clean[clean["Date"] == latest]
            st.write(f"현재 원본에서 확인되는 가장 최근 날짜는 **{pd.Timestamp(latest).date()}**이며, 이 날짜에 {latest_rows['Index'].nunique()}개 시장의 기록이 있습니다.")
            st.caption("시장마다 거래일과 휴장일이 달라 같은 날짜의 기록 수가 다를 수 있습니다.")
        else:
            st.info("표시할 데이터가 없습니다.")

    st.divider()
    st.subheader("어떤 분석을 할 수 있나요?")
    tiles = st.columns(3)
    tiles[0].markdown("**🔍 데이터 탐색**\n\n어떤 시장과 날짜가 데이터에 포함되어 있는지 확인합니다.")
    tiles[1].markdown("**🧹 데이터 전처리**\n\n분석 전에 데이터를 정리하고 수익률로 바꾸는 이유를 살펴봅니다.")
    tiles[2].markdown("**🧠 시장 분석**\n\n비슷한 시장, 시장 사이의 관계, 과거 하락 사례를 분석합니다.")

    st.warning("분석 결과는 과거 자료를 바탕으로 한 학습용 정보입니다. 미래 수익을 보장하거나 손실을 막아 주는 투자 신호가 아닙니다.")


if not PAGES_DIR.exists():
    st.error("pages 폴더를 찾을 수 없습니다.")
    st.stop()

page_specs = [
    ("1_탐색.py", "데이터 탐색", "🔍"),
    ("2_데이터_전처리.py", "데이터 전처리", "🧹"),
    ("3_모델_학습_결과.py", "시장 분석", "🧠")
]
for filename, _, _ in page_specs:
    if not (PAGES_DIR / filename).is_file():
        st.error(f"페이지 파일이 없습니다: pages/{filename}")
        st.stop()

home = st.Page(home_page, title="홈", icon="🏠", default=True)
explore = st.Page(str(PAGES_DIR / "1_탐색.py"), title="데이터 탐색", icon="🔍")
preprocess = st.Page(str(PAGES_DIR / "2_데이터_전처리.py"), title="데이터 전처리", icon="🧹")
model = st.Page(str(PAGES_DIR / "3_모델_학습_결과.py"), title="시장 분석", icon="🧠")

pg = st.navigation(
    {"서비스": [home, explore, preprocess], "심층 분석": [model]},
    position="sidebar"
)
with st.sidebar:
    st.divider()
    st.caption("세계 금융시장 분석 · 학습용")
pg.run()
