import io
import pandas as pd
import plotly.express as px
import streamlit as st

# GitHub Raw CSV 주소를 미리 지정합니다.
DATA_URL = "https://raw.githubusercontent.com/gosokturn/Analyzing-personal-Data-project/refs/heads/main/data/indexData.csv"

st.set_page_config(
    page_title="주가지수 데이터 기초통계 분석",
    page_icon="📊",
    layout="wide",
)

st.title("📊 주가지수 데이터 기초통계 분석")
st.write(
    "GitHub에 저장된 주가지수 CSV 데이터를 자동으로 불러와 "
    "데이터의 구조와 기초통계를 살펴보는 교육용 앱입니다."
)

@st.cache_data(show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    """URL에서 CSV를 읽어 옵니다."""
    return pd.read_csv(url)

with st.sidebar:
    st.header("분석 설정")
    st.caption("데이터 주소는 앱 코드에 미리 설정되어 있습니다.")
    st.code(DATA_URL, language=None)
    reload_data = st.button("🔄 데이터 다시 불러오기", use_container_width=True)
    if reload_data:
        load_data.clear()

try:
    with st.spinner("GitHub에서 데이터를 불러오는 중입니다..."):
        df = load_data(DATA_URL)
except Exception as exc:
    st.error("데이터를 불러오지 못했습니다.")
    st.write("확인할 사항:")
    st.markdown(
        "- GitHub Raw 주소가 올바른지 확인합니다.\n"
        "- 저장소와 CSV 파일이 공개되어 있는지 확인합니다.\n"
        "- 파일이 정상적인 CSV 형식인지 확인합니다."
    )
    st.caption(f"오류 정보: {exc}")
    st.stop()

if df.empty:
    st.warning("CSV 파일에 분석할 데이터가 없습니다.")
    st.stop()

# 기본적인 공통 지표
row_count, col_count = df.shape
missing_cells = int(df.isna().sum().sum())
missing_percent = missing_cells / (row_count * col_count) * 100 if row_count * col_count else 0
duplicate_rows = int(df.duplicated().sum())

st.subheader("1. 데이터 한눈에 보기")
c1, c2, c3, c4 = st.columns(4)
c1.metric("행 수", f"{row_count:,}")
c2.metric("열 수", f"{col_count:,}")
c3.metric("결측값 수", f"{missing_cells:,}", f"{missing_percent:.2f}%")
c4.metric("중복 행 수", f"{duplicate_rows:,}")

with st.expander("데이터 미리 보기", expanded=True):
    st.dataframe(df.head(10), use_container_width=True)

st.subheader("2. 열별 데이터 구조")
schema = pd.DataFrame({
    "열 이름": df.columns,
    "자료형": [str(df[col].dtype) for col in df.columns],
    "고유값 수": [int(df[col].nunique(dropna=True)) for col in df.columns],
    "결측값 수": [int(df[col].isna().sum()) for col in df.columns],
    "결측 비율(%)": [
        round(float(df[col].isna().mean() * 100), 2) for col in df.columns
    ],
})
st.dataframe(schema, use_container_width=True, hide_index=True)

st.subheader("3. 수치형 데이터 기초통계")
numeric_cols = df.select_dtypes(include="number").columns.tolist()

if numeric_cols:
    numeric_stats = df[numeric_cols].describe().T
    numeric_stats = numeric_stats.rename(columns={
        "count": "개수",
        "mean": "평균",
        "std": "표준편차",
        "min": "최솟값",
        "25%": "1사분위수",
        "50%": "중앙값",
        "75%": "3사분위수",
        "max": "최댓값",
    })
    numeric_stats["결측값 수"] = df[numeric_cols].isna().sum()
    st.dataframe(numeric_stats, use_container_width=True)

    st.markdown("#### 수치형 열 분포 살펴보기")
    selected_numeric = st.selectbox("분포를 확인할 열", numeric_cols)
    chart_data = pd.to_numeric(df[selected_numeric], errors="coerce").dropna()

    if not chart_data.empty:
        left, right = st.columns(2)
        with left:
            fig_hist = px.histogram(
                x=chart_data,
                nbins=40,
                title=f"{selected_numeric} 분포",
                labels={"x": selected_numeric, "y": "빈도"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        with right:
            fig_box = px.box(
                x=chart_data,
                title=f"{selected_numeric} 상자그림",
                labels={"x": selected_numeric},
            )
            st.plotly_chart(fig_box, use_container_width=True)
else:
    st.info("수치형 열이 없어 수치형 기초통계를 표시할 수 없습니다.")

st.subheader("4. 범주형 데이터 기초통계")
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

if categorical_cols:
    cat_rows = []
    for col in categorical_cols:
        series = df[col].dropna()
        if series.empty:
            mode_value = "값 없음"
            mode_count = 0
        else:
            counts = series.astype(str).value_counts()
            mode_value = str(counts.index[0])
            mode_count = int(counts.iloc[0])
        cat_rows.append({
            "열 이름": col,
            "고유값 수": int(df[col].nunique(dropna=True)),
            "최빈값": mode_value,
            "최빈값 빈도": mode_count,
            "결측값 수": int(df[col].isna().sum()),
        })
    st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)
else:
    st.info("범주형 열이 없어 범주형 기초통계를 표시할 수 없습니다.")

st.subheader("5. 기초통계표 다운로드")
csv_buffer = io.StringIO()
if numeric_cols:
    numeric_stats.to_csv(csv_buffer, encoding="utf-8-sig")
    st.download_button(
        "⬇️ 수치형 기초통계 CSV 다운로드",
        data=csv_buffer.getvalue().encode("utf-8-sig"),
        file_name="numeric_summary.csv",
        mime="text/csv",
    )
else:
    st.caption("수치형 열이 없어 다운로드할 기초통계표가 없습니다.")

st.divider()
st.subheader("해석할 때 주의할 점")
st.markdown(
    "- **평균과 중앙값**을 비교하면 값의 분포가 한쪽으로 치우쳤는지 살펴보는 데 도움이 됩니다.\n"
    "- **표준편차**가 크면 데이터 값이 평균 주변에 넓게 퍼져 있을 가능성이 있습니다.\n"
    "- **결측값**과 **중복 행**은 분석 전에 원인을 확인해야 합니다.\n"
    "- 주가지수 데이터의 평균이나 표준편차만으로 급락 위험을 판단할 수는 없습니다. "
    "시장별 수익률, 변동성, 최대 낙폭 등을 별도로 계산해야 합니다."
)
st.caption("교육용 데이터 분석 도구입니다. 통계 결과는 데이터의 품질과 구성에 따라 달라질 수 있습니다.")
