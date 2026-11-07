import streamlit as st
import pandas as pd
import plotly.express as px


DATA_URL = (
    "https://raw.githubusercontent.com/"
    "gosokturn/Analyzing-personal-Data-project/"
    "refs/heads/main/data/indexData.csv"
)


@st.cache_data
def load_data(url):
    return pd.read_csv(url, encoding="utf-8")


st.title("📊 기초 데이터 분석")
st.caption("원본 금융 지수 데이터의 구조와 통계적 특성을 탐색합니다.")

try:
    df = load_data(DATA_URL)
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# 1. 데이터 기본 정보

st.header("1. 데이터 기본 정보")

col1, col2, col3 = st.columns(3)

col1.metric("행 개수", f"{len(df):,}")
col2.metric("열 개수", f"{len(df.columns):,}")
col3.metric(
    "결측 셀 개수",
    f"{int(df.isna().sum().sum()):,}"
)

st.subheader("데이터 미리 보기")
st.dataframe(df.head(20), use_container_width=True)

with st.expander("열 이름, 데이터 유형, 결측치 확인"):

    schema = pd.DataFrame({
        "열 이름": df.columns,
        "데이터 유형": [
            str(df[col].dtype) for col in df.columns
        ],
        "결측치 개수": [
            int(df[col].isna().sum()) for col in df.columns
        ],
        "결측 비율(%)": [
            round(df[col].isna().mean() * 100, 2)
            for col in df.columns
        ]
    })

    st.dataframe(
        schema,
        use_container_width=True,
        hide_index=True
    )


# 2. 수치형 데이터 통계

st.header("2. 수치형 데이터 통계")

numeric_df = df.select_dtypes(include="number")

if not numeric_df.empty:

    st.dataframe(
        numeric_df.describe().T.round(3),
        use_container_width=True
    )

else:
    st.info("수치형 열이 없습니다.")


# 3. 숫자형 변수 분포

st.header("3. 숫자형 변수 분포")

if not numeric_df.empty:

    selected_col = st.selectbox(
        "분포를 확인할 열",
        options=numeric_df.columns.tolist()
    )

    chart_data = pd.to_numeric(
        df[selected_col],
        errors="coerce"
    ).dropna()

    col1, col2 = st.columns(2)

    with col1:

        fig_hist = px.histogram(
            chart_data,
            x=selected_col,
            title=f"{selected_col}의 분포",
            nbins=40
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )

    with col2:

        fig_box = px.box(
            chart_data,
            y=selected_col,
            title=f"{selected_col}의 상자 그림",
            points="outliers"
        )

        st.plotly_chart(
            fig_box,
            use_container_width=True
        )

    st.write(f"유효한 데이터 개수: {chart_data.count():,}")
    st.write(f"평균: {chart_data.mean():,.4f}")
    st.write(f"중앙값: {chart_data.median():,.4f}")
    st.write(f"표준편차: {chart_data.std():,.4f}")


# 4. 지수별 데이터 비교

st.header("4. 지수별 데이터 비교")

if "Index" in df.columns:

    index_counts = (
        df["Index"]
        .value_counts(dropna=False)
        .rename_axis("지수")
        .reset_index(name="행 개수")
    )

    fig_index = px.bar(
        index_counts,
        x="지수",
        y="행 개수",
        title="지수별 데이터 개수"
    )

    st.plotly_chart(
        fig_index,
        use_container_width=True
    )

    st.dataframe(
        index_counts,
        use_container_width=True,
        hide_index=True
    )

    available_indexes = (
        df["Index"].dropna().unique().tolist()
    )

    if available_indexes:

        selected_index = st.selectbox(
            "상세하게 확인할 지수",
            options=available_indexes
        )

        selected_data = df[
            df["Index"] == selected_index
        ].copy()

        st.subheader(f"{selected_index} 데이터")

        if (
            "Date" in selected_data.columns
            and "Close" in selected_data.columns
        ):

            selected_data["Date"] = pd.to_datetime(
                selected_data["Date"],
                errors="coerce"
            )

            selected_data["Close"] = pd.to_numeric(
                selected_data["Close"],
                errors="coerce"
            )

            selected_data = selected_data.dropna(
                subset=["Date", "Close"]
            ).sort_values("Date")

            if not selected_data.empty:

                fig_line = px.line(
                    selected_data,
                    x="Date",
                    y="Close",
                    title=f"{selected_index} 종가 변화"
                )

                st.plotly_chart(
                    fig_line,
                    use_container_width=True
                )

        with st.expander("선택한 지수의 데이터 미리 보기"):

            st.dataframe(
                selected_data.head(100),
                use_container_width=True
            )

else:
    st.info(
        "Index 열이 없어 지수별 비교를 표시할 수 없습니다."
    )


# 5. 원본 데이터 다운로드

st.header("5. 원본 데이터 다운로드")

st.download_button(
    label="📥 원본 CSV 다운로드",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="indexData_original.csv",
    mime="text/csv"
)

st.caption(
    "이 페이지는 원본 데이터 탐색용입니다. "
    "실제 전처리는 '데이터 전처리' 페이지에서 진행하세요."
)
