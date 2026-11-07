
import streamlit as st
import pandas as pd
import numpy as np

from market_utils import (
    load_raw_data,
    prepare_market_data,
    build_market_features
)


st.title("🧹 세계 금융시장 데이터 전처리")
st.caption(
    "국가별·시장별 지수를 공정하게 비교하기 위한 전처리 과정"
)


# ==========================================
# 1. 원본 데이터
# ==========================================

try:
    raw_df = load_raw_data()

except Exception as e:
    st.error("원본 데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


st.header("1. 원본 데이터 확인")

c1, c2, c3 = st.columns(3)

c1.metric("원본 행 개수", f"{len(raw_df):,}")
c2.metric("열 개수", f"{len(raw_df.columns):,}")
c3.metric(
    "전체 결측 셀",
    f"{int(raw_df.isna().sum().sum()):,}"
)

with st.expander("원본 데이터 미리 보기"):
    st.dataframe(
        raw_df.head(30),
        use_container_width=True
    )

missing_table = pd.DataFrame({
    "열": raw_df.columns,
    "결측 개수": [
        int(raw_df[col].isna().sum())
        for col in raw_df.columns
    ],
    "결측 비율(%)": [
        round(raw_df[col].isna().mean() * 100, 2)
        for col in raw_df.columns
    ]
})

st.subheader("열별 결측치 현황")

st.dataframe(
    missing_table.sort_values(
        "결측 비율(%)",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 2. 전처리 방법 설명
# ==========================================

st.divider()
st.header("2. 전처리 방법과 선택 이유")

st.markdown("""
### ① 날짜 및 데이터 유형 변환

날짜는 날짜형으로, 주가와 거래량은 숫자형으로 변환한다.

**이유:** 날짜가 문자로 저장되어 있으면 시간 순서에 따른 수익률 계산이
잘못될 수 있다. 숫자형 변환은 통계 계산을 가능하게 한다.

### ② 결측 가격 처리

지수 이름, 날짜, 종가가 없는 행은 분석에서 제외한다.

**이유:** 주가가 없는 날짜의 가격을 중앙값으로 채우면 실제로 관측되지
않은 가격이 만들어진다. 이렇게 만들어진 가격은 가짜 수익률을 발생시킬 수 있다.

### ③ 중복 기록 제거

같은 지수와 같은 날짜에 기록된 행이 여러 개라면 마지막 기록만 남긴다.

**이유:** 동일한 시장의 동일한 날짜가 여러 번 계산되면 특정 날짜의
영향이 과도하게 반영될 수 있다.

### ④ 수익률 변환

종가 자체가 아닌 전일 대비 수익률을 계산한다.

**이유:** 주가지수는 국가마다 숫자의 단위와 기준점이 다르다.
따라서 가격 수준보다 수익률을 비교하는 것이 적절하다.

### ⑤ 이상치 처리

금융 데이터의 큰 변동은 실제 경제 사건일 수 있으므로 무조건 삭제하지 않는다.

**이유:** 실제 급락 데이터를 삭제하면 시장 위험이 실제보다 작게 측정될 수 있다.

### ⑥ 변수 표준화

K-Means 군집화 전에 RobustScaler를 사용한다.

**이유:** 변동성이나 수익률 등 변수마다 숫자의 크기가 다르다.
큰 숫자를 가진 변수가 군집화 결과에 과도하게 영향을 주는 것을 줄인다.
""")

st.info(
    "주의: 결측 가격을 중앙값으로 채우지 않는 것은 금융 시계열의 "
    "실제 움직임을 보존하기 위한 선택이다. 모든 데이터에 같은 결측치 "
    "처리 방법이 정답인 것은 아니다."
)


# ==========================================
# 3. 극단값 완화 선택
# ==========================================

st.divider()
st.header("3. 이상치 및 극단값 처리")

st.markdown("""
수익률이 다른 날짜보다 매우 크거나 작으면 이상치 후보로 볼 수 있다.

그러나 금융시장에서는 경제 위기, 정책 변화, 국제 사건 등으로 인해
실제로 큰 가격 변동이 발생할 수 있다.

따라서 기본적으로 극단값을 유지한다.
""")

winsorize = st.checkbox(
    "시장별 수익률의 1%, 99% 분위수 경계로 극단값 완화",
    value=False,
    key="winsorize_returns"
)

if winsorize:
    st.warning(
        "극단값 완화를 선택했다. 실제 급락과 급등의 크기도 줄어들 수 있으므로 "
        "위험 분석에서는 원본 수익률 결과와 반드시 비교해야 한다."
    )
else:
    st.success(
        "극단값을 그대로 유지한다. 실제 시장 위험을 보존하는 기본 설정이다."
    )


# ==========================================
# 4. 전처리 실행
# ==========================================

st.divider()
st.header("4. 전처리 실행")

try:
    clean_df, price_col, audit = prepare_market_data(
        raw_df,
        winsorize=winsorize
    )

except Exception as e:
    st.error("전처리 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


c1, c2, c3 = st.columns(3)

c1.metric(
    "전처리 전 행",
    f"{audit['original_rows']:,}"
)

c2.metric(
    "전처리 후 행",
    f"{audit['final_rows']:,}"
)

c3.metric(
    "제거된 행",
    f"{audit['original_rows'] - audit['final_rows']:,}"
)

audit_table = pd.DataFrame({
    "단계": [
        "원본 데이터",
        "필수 정보와 유효한 가격 확인 후",
        "중복 기록 제거 후"
    ],
    "행 개수": [
        audit["original_rows"],
        audit["after_valid_rows"],
        audit["final_rows"]
    ]
})

st.dataframe(
    audit_table,
    use_container_width=True,
    hide_index=True
)

st.write(f"사용한 종가 열: **{price_col}**")
st.write(f"중복 제거 행: **{audit['duplicate_removed']:,}개**")
st.write(f"유효하지 않은 행 제거: **{audit['invalid_removed']:,}개**")


# ==========================================
# 5. 수익률 계산 결과
# ==========================================

st.divider()
st.header("5. 수익률 변환 결과")

st.markdown("""
수익률은 현재 날짜의 종가를 전날 종가와 비교해 계산한다.

예를 들어 전날 종가가 100이고 오늘 종가가 102라면 수익률은 약 2%이다.

이렇게 변환하면 서로 다른 숫자 단위의 주가지수를 비교할 수 있다.
""")

return_data = clean_df.dropna(
    subset=["daily_return"]
).copy()

if not return_data.empty:

    st.metric(
        "수익률 계산이 완료된 행",
        f"{len(return_data):,}"
    )

    st.dataframe(
        return_data[
            [
                "Index",
                "Date",
                price_col,
                "daily_return",
                "daily_return_used"
            ]
        ].head(30),
        use_container_width=True
    )

    st.subheader("일별 수익률 분포")

    st.caption(
        "수익률 분포의 꼬리가 길다면 일부 날짜에 큰 변동이 있었을 수 있다."
    )

    import plotly.express as px

    fig = px.histogram(
        return_data,
        x="daily_return",
        nbins=80,
        title="시장 전체의 일별 수익률 분포"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    q01 = return_data["daily_return"].quantile(0.01)
    q99 = return_data["daily_return"].quantile(0.99)

    st.write(f"하위 1% 수익률: **{q01:.4%}**")
    st.write(f"상위 1% 수익률: **{q99:.4%}**")


# ==========================================
# 6. 시장별 특징 데이터
# ==========================================

st.divider()
st.header("6. 시장별 분석용 변수 생성")

st.markdown("""
각 시장의 특징을 숫자로 표현하기 위해 다음 변수를 계산한다.

- 연환산 평균 수익률: 하루 평균 수익률을 연간 거래일 252일 기준으로 환산한 근사치
- 연환산 변동성: 수익률의 표준편차를 연간 기준으로 환산
- 하락 변동성: 하락한 날의 수익률 변동 정도
- 최대 낙폭: 과거 최고점에서 가장 크게 하락한 비율
- 상승일 비율: 수익률이 양수였던 거래일의 비율
- 평균 절대 수익률: 하루 변동의 평균적인 크기
- 평균 일중 변동 폭: 고가와 저가의 차이를 전일 종가로 나눈 값
""")

min_obs = st.slider(
    "시장별 최소 수익률 관측 일수",
    min_value=30,
    max_value=252,
    value=60,
    step=10
)

features = build_market_features(
    clean_df,
    price_col,
    min_observations=min_obs
)

if features.empty:
    st.warning(
        "조건을 만족하는 시장이 없습니다. "
        "최소 관측 일수를 낮추거나 원본 데이터의 기간을 확인해 주세요."
    )
else:
    st.metric(
        "분석 가능한 시장 수",
        f"{len(features):,}"
    )

    st.dataframe(
        features.round(4),
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# 7. 다운로드
# ==========================================

st.divider()
st.header("7. 전처리 데이터 다운로드")

st.download_button(
    label="📥 전처리된 금융 데이터 CSV 다운로드",
    data=clean_df.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="global_market_preprocessed.csv",
    mime="text/csv"
)

if not features.empty:
    st.download_button(
        label="📥 시장별 특징 데이터 CSV 다운로드",
        data=features.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="global_market_features.csv",
        mime="text/csv"
    )

st.caption(
    "모델 학습 페이지에서도 동일한 전처리 함수를 사용한다."
)
