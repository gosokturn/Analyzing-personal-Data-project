import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from market_utils import load_raw_data, prepare_market_data, build_market_features

st.title("🧹 데이터 전처리")
st.caption("원본 데이터를 분석에 사용할 수 있는 형태로 정리하고, 각 처리 과정의 이유와 영향을 확인합니다.")

try:
    raw_df = load_raw_data()
except Exception as e:
    st.error("원본 데이터를 불러오지 못했습니다. 데이터 주소와 CSV 상태를 확인해 주세요.")
    with st.expander("오류 세부 정보"):
        st.exception(e)
    st.stop()

st.subheader("먼저 원본 데이터 상태를 확인합니다")
a, b, c = st.columns(3)
a.metric("원본 행 수", f"{len(raw_df):,}")
b.metric("원본 결측 셀", f"{int(raw_df.isna().sum().sum()):,}")
if {"Index", "Date"}.issubset(raw_df.columns):
    dup_count = int(raw_df.duplicated(subset=["Index", "Date"]).sum())
else:
    dup_count = int(raw_df.duplicated().sum())
c.metric("중복 기록 후보", f"{dup_count:,}", help="같은 시장(Index)과 날짜(Date)가 반복된 행의 수입니다.")

with st.expander("원본 데이터 미리 보기"):
    st.dataframe(raw_df.head(30), use_container_width=True, hide_index=True)

with st.expander("열별 결측 데이터 자세히 보기"):
    missing = pd.DataFrame({
        "열": raw_df.columns,
        "결측 개수": [int(raw_df[col].isna().sum()) for col in raw_df.columns],
        "결측 비율": [raw_df[col].isna().mean() for col in raw_df.columns]
    }).sort_values("결측 비율", ascending=False)
    missing["결측 비율"] = missing["결측 비율"].map(lambda x: f"{x:.1%}")
    st.dataframe(missing, use_container_width=True, hide_index=True)

st.divider()
st.subheader("전처리는 왜 필요할까요?")
st.write("컴퓨터가 날짜와 숫자를 올바르게 읽고, 같은 기록이 중복 계산되지 않도록 정리하는 과정입니다. 금융 데이터에서는 큰 변동이 실제 사건일 수도 있으므로 데이터를 무조건 삭제하거나 평균으로 채우지 않는 판단도 중요합니다.")

steps = [
    ("① 날짜 형식 변환", "문자처럼 저장된 날짜를 날짜형으로 읽습니다.", "시간 순서에 따른 전일 대비 수익률과 기간 비교가 가능해집니다.", "날짜가 문자로 남으면 정렬이나 기간 계산이 잘못될 수 있습니다."),
    ("② 숫자형 데이터 변환", "주가·거래량 열을 숫자로 변환합니다.", "평균, 변동률, 그래프 등의 계산을 할 수 있습니다.", "숫자처럼 보이는 문자 값은 통계 계산에서 빠지거나 오류를 낼 수 있습니다."),
    ("③ 결측 가격 확인", "시장명·날짜·사용 종가가 없는 행은 분석에서 제외합니다.", "실제로 관측되지 않은 가격을 임의로 만들지 않습니다.", "평균이나 중앙값으로 가격을 채우면 존재하지 않은 가격과 가짜 수익률이 생길 수 있습니다."),
    ("④ 중복 기록 제거", "같은 시장과 같은 날짜가 반복되면 마지막 기록을 남깁니다.", "특정 날짜가 여러 번 반영되는 문제를 줄입니다.", "중복 행이 남으면 해당 날짜의 영향이 과도하게 커질 수 있습니다."),
    ("⑤ 유효하지 않은 가격 제외", "0 이하의 종가는 수익률 계산에서 제외합니다.", "가격을 분모로 사용하는 수익률 계산의 오류를 방지합니다.", "0 또는 음수 가격으로는 일반적인 주가지수 수익률을 계산할 수 없습니다."),
    ("⑥ 일별 수익률 계산", "오늘 종가와 전날 종가의 변화율을 계산합니다.", "기준점이 서로 다른 지수의 움직임을 비교하기 쉬워집니다.", "가격 숫자만 비교하면 지수별 기준값 차이가 성과 차이처럼 보일 수 있습니다."),
    ("⑦ 이상치 확인", "극단적인 수익률을 오류라고 단정하지 않고 확인합니다.", "실제 급락·급등 사건을 분석에 남길 수 있습니다.", "무조건 삭제하면 시장의 위험이 실제보다 작게 보일 수 있습니다."),
    ("⑧ 변수 표준화", "군집화 전에 RobustScaler로 변수의 크기 차이를 조정합니다.", "수익률·변동성처럼 단위와 범위가 다른 변수를 함께 비교할 수 있습니다.", "숫자 범위가 큰 변수에 K-Means의 거리 계산이 치우칠 수 있습니다.")
]
for title, what, why, risk in steps:
    with st.expander(title, expanded=False):
        st.markdown(f"**무엇을 하나요?** {what}")
        st.markdown(f"**왜 필요한가요?** {why}")
        st.markdown(f"**처리하지 않으면?** {risk}")

st.info("결측 가격은 평균이나 중앙값으로 채우지 않습니다. 또한 극단적인 수익률은 실제 경제 사건일 수 있어 기본 설정에서는 그대로 유지합니다. 표준화는 이 페이지가 아니라 시장 유형 분석의 K-Means 입력 단계에서 적용됩니다.")

st.divider()
st.subheader("극단적인 수익률 처리 방식")
st.write("기본값은 실제 급등·급락을 보존하는 방식입니다. 아래 선택을 켜면 각 시장의 수익률 하위 1%, 상위 1% 경계를 넘는 값을 경계값으로 조정합니다.")
winsorize = st.checkbox(
    "시장별 수익률의 1%·99% 분위수 경계로 극단값 완화",
    value=False,
    key="winsorize_returns",
    help="극단값을 삭제하지 않고 각 시장의 분위수 경계 안으로 조정합니다. 실제 급락·급등의 크기도 줄어들 수 있습니다."
)
if winsorize:
    st.warning("극단값 완화를 적용했습니다. 위험 분석에서는 원래 수익률을 사용한 결과와 비교하는 것이 중요합니다.")
else:
    st.success("극단값을 유지합니다. 실제 시장 위험을 보존하는 기본 설정입니다.")

try:
    clean_df, price_col, audit = prepare_market_data(raw_df, winsorize=winsorize)
except Exception as e:
    st.error("데이터 전처리 중 문제가 발생했습니다.")
    with st.expander("오류 세부 정보"):
        st.exception(e)
    st.stop()

st.divider()
st.subheader("전처리 전후 비교")
x1, x2, x3, x4 = st.columns(4)
x1.metric("처리 전 행", f"{audit['original_rows']:,}")
x2.metric("처리 후 행", f"{audit['final_rows']:,}", delta=f"{audit['final_rows']-audit['original_rows']:,}행")
x3.metric("유효하지 않은 행 제외", f"{audit['invalid_removed']:,}")
x4.metric("중복 행 제외", f"{audit['duplicate_removed']:,}")

audit_table = pd.DataFrame({
    "처리 단계": ["원본", "필수 정보와 유효한 가격 확인 후", "중복 제거 후"],
    "남은 행 수": [audit["original_rows"], audit["after_valid_rows"], audit["final_rows"]]
})
fig_rows = px.bar(audit_table, x="처리 단계", y="남은 행 수", title="전처리 단계별 데이터 행 수")
fig_rows.update_layout(showlegend=False)
st.plotly_chart(fig_rows, use_container_width=True)
st.caption(f"수익률 계산에 사용한 가격 열: {price_col}. 첫 관측일처럼 이전 날짜의 가격이 없는 행은 수익률이 계산되지 않습니다.")

before_missing = int(raw_df.isna().sum().sum())
after_missing = int(clean_df.isna().sum().sum())
m1, m2, m3 = st.columns(3)
m1.metric("원본 결측 셀", f"{before_missing:,}")
m2.metric("전처리 후 결측 셀", f"{after_missing:,}")
m3.metric("계산된 일별 수익률", f"{clean_df['daily_return'].notna().sum():,}개")
st.caption("결측 셀 수는 행 제외와 별개입니다. 분석에 필요한 값이 없는 행을 제외했더라도 High·Low·Volume 등 선택적으로 사용하는 열에는 결측값이 남을 수 있습니다.")

st.divider()
st.subheader("수익률로 바꾸면 무엇이 달라질까요?")
st.write("예를 들어 지수 값이 100에서 102로 변하면 수익률은 약 +2%입니다. 시장마다 지수의 기준 숫자가 다르기 때문에 지수 값 자체보다 변화율을 비교하는 것이 더 적절합니다.")
returns = clean_df.dropna(subset=["daily_return"]).copy()
if not returns.empty:
    raw_returns = returns["daily_return"].replace([np.inf, -np.inf], np.nan).dropna()
    used_returns = returns["daily_return_used"].replace([np.inf, -np.inf], np.nan).dropna()
    q1, q2, q3 = st.columns(3)
    q1.metric("수익률 관측 수", f"{len(raw_returns):,}")
    q2.metric("하위 1% 수익률", f"{raw_returns.quantile(.01):.2%}")
    q3.metric("상위 1% 수익률", f"{raw_returns.quantile(.99):.2%}")
    hist_df = pd.DataFrame({
        "원래 수익률": raw_returns,
        "분석에 사용한 수익률": used_returns
    })
    fig = px.histogram(
        hist_df, x=["원래 수익률", "분석에 사용한 수익률"],
        barmode="overlay", opacity=0.65, nbins=70,
        title="원래 수익률과 분석에 사용한 수익률 분포 비교",
        labels={"value": "일별 수익률", "variable": "구분"}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("극단값 완화를 선택하지 않았다면 두 분포는 동일합니다. 선택한 경우에는 1%·99% 분위수 바깥의 값이 경계값으로 조정됩니다.")
else:
    st.info("수익률을 계산할 수 있는 관측값이 없습니다.")

st.divider()
st.subheader("시장별 특징값 만들기")
st.write("각 시장의 움직임을 여러 숫자로 요약합니다. 연환산 평균 수익률은 일별 평균 수익률에 252를 곱한 근사치이며, 연환산 변동성은 일별 수익률 표준편차에 √252를 곱한 값입니다.")
min_obs = st.slider("시장별 최소 수익률 관측 일수", 30, 252, 60, 10)
features = build_market_features(clean_df, price_col, min_observations=min_obs)
if features.empty:
    st.warning("선택한 최소 관측 일수를 만족하는 시장이 없습니다. 기준을 낮추거나 데이터 기간을 확인해 주세요.")
else:
    st.metric("특징값을 계산한 시장", f"{len(features):,}개")
    st.dataframe(features.round(4), use_container_width=True, hide_index=True)

with st.expander("각 특징값의 의미"):
    st.markdown("""
    - **연환산 평균 수익률:** 하루 평균 수익률을 252거래일 기준으로 환산한 근사치
    - **연환산 변동성:** 수익률이 평균 주변에서 얼마나 크게 흔들렸는지 나타내는 값
    - **하락 변동성:** 하락한 날들의 수익률 표준편차를 연환산한 값
    - **최대 낙폭:** 과거 최고 가격에서 이후 가장 크게 내려간 비율
    - **상승일 비율:** 수익률이 0보다 컸던 거래일의 비율
    - **평균 절대 수익률:** 하루 변동 폭의 평균적인 크기
    - **평균 일중 변동 폭:** 고가와 저가의 차이를 전일 종가로 나눈 값
    """)

st.divider()
st.subheader("정리한 데이터 내려받기")
d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "📥 전처리된 금융 데이터 CSV",
        data=clean_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="global_market_preprocessed.csv",
        mime="text/csv",
        use_container_width=True
    )
with d2:
    if not features.empty:
        st.download_button(
            "📥 시장별 특징 데이터 CSV",
            data=features.to_csv(index=False).encode("utf-8-sig"),
            file_name="global_market_features.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.button("특징 데이터 없음", disabled=True, use_container_width=True)

st.caption("모델 분석 페이지에서도 동일한 전처리 함수와 극단값 완화 설정을 사용합니다.")
