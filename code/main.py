import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="금융 지수 데이터 분석",
    page_icon="📈",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "gosokturn/Analyzing-personal-Data-project/"
    "refs/heads/main/data/indexData.csv"
)


@st.cache_data
def load_data(url):
    return pd.read_csv(url, encoding="utf-8")


st.title("📈 금융 지수 데이터 분석")
st.write(
    "금융 지수 데이터를 탐색하고, "
    "별도의 페이지에서 데이터 전처리를 수행할 수 있는 앱입니다."
)

st.info(
    "왼쪽 사이드바에서 '기초 데이터 분석' 또는 "
    "'데이터 전처리' 페이지를 선택하세요."
)

try:
    df = load_data(DATA_URL)
except Exception as e:
    st.error("GitHub에서 CSV 데이터를 불러오지 못했습니다.")
    st.write("데이터 주소와 저장소의 공개 설정을 확인해 주세요.")
    st.exception(e)
    st.stop()

col1, col2, col3 = st.columns(3)

col1.metric("행 개수", f"{len(df):,}")
col2.metric("열 개수", f"{len(df.columns):,}")
col3.metric(
    "결측 셀 개수",
    f"{int(df.isna().sum().sum()):,}"
)

st.subheader("원본 데이터 미리 보기")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("열 정보")

schema = pd.DataFrame({
    "열 이름": df.columns,
    "데이터 유형": [
        str(df[col].dtype) for col in df.columns
    ],
    "결측치 개수": [
        int(df[col].isna().sum()) for col in df.columns
    ]
})

st.dataframe(
    schema,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "전처리 페이지에서 적용한 변경은 "
    "이 메인 화면의 원본 데이터에 영향을 주지 않습니다."
)
