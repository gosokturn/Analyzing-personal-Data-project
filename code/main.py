
import streamlit as st
import pandas as pd
from pathlib import Path


# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="금융 지수 데이터 분석",
    page_icon="📈",
    layout="wide"
)


# ==========================================
# 2. 파일 경로 설정
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR / "pages"

EXPLORE_FILE = PAGES_DIR / "1_탐색.py"
PREPROCESS_FILE = PAGES_DIR / "2_데이터_전처리.py"


# ==========================================
# 3. 페이지 파일 확인
# ==========================================

if not PAGES_DIR.exists():
    st.error("pages 폴더를 찾을 수 없습니다.")
    st.write(f"확인한 경로: {PAGES_DIR}")
    st.stop()

if not EXPLORE_FILE.is_file():
    st.error("1_탐색.py 파일을 찾을 수 없습니다.")
    st.write(f"확인한 경로: {EXPLORE_FILE}")
    st.stop()

if not PREPROCESS_FILE.is_file():
    st.error("2_데이터_전처리.py 파일을 찾을 수 없습니다.")
    st.write(f"확인한 경로: {PREPROCESS_FILE}")
    st.stop()


# ==========================================
# 4. 데이터 주소
# ==========================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "gosokturn/Analyzing-personal-Data-project/"
    "refs/heads/main/data/indexData.csv"
)


# ==========================================
# 5. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data(url):
    return pd.read_csv(url, encoding="utf-8")


# ==========================================
# 6. 홈 화면
# ==========================================

def home_page():

    st.title("📈 금융 지수 데이터 분석")

    st.write(
        "금융 지수 데이터를 탐색하고, "
        "별도의 페이지에서 데이터 전처리를 수행할 수 있는 앱입니다."
    )

    st.info(
        "왼쪽 사이드바에서 원하는 페이지를 선택하세요."
    )

    # 데이터 불러오기
    try:
        df = load_data(DATA_URL)

    except Exception as e:
        st.error("GitHub에서 CSV 데이터를 불러오지 못했습니다.")
        st.write("데이터 주소와 저장소의 공개 설정을 확인해 주세요.")
        st.exception(e)
        return

    # 데이터 요약
    st.subheader("📊 원본 데이터 요약")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "행 개수",
        f"{len(df):,}"
    )

    col2.metric(
        "열 개수",
        f"{len(df.columns):,}"
    )

    col3.metric(
        "결측 셀 개수",
        f"{int(df.isna().sum().sum()):,}"
    )

    # 원본 데이터 미리 보기
    st.divider()

    st.subheader("📋 원본 데이터 미리 보기")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # 열 정보
    st.divider()

    st.subheader("🔎 열 정보")

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

    st.divider()

    st.caption(
        "전처리 페이지에서 적용한 변경은 "
        "이 메인 화면의 원본 데이터에 영향을 주지 않습니다."
    )


# ==========================================
# 7. 페이지 등록
# ==========================================

home = st.Page(
    home_page,
    title="홈",
    icon="🏠",
    default=True
)

explore = st.Page(
    str(EXPLORE_FILE),
    title="데이터 탐색",
    icon="🔍",
    url_path="explore"
)

preprocess = st.Page(
    str(PREPROCESS_FILE),
    title="데이터 전처리",
    icon="🧹",
    url_path="preprocessing"
)


# ==========================================
# 8. 사이드바 및 페이지 이동
# ==========================================

pg = st.navigation(
    [
        home,
        explore,
        preprocess
    ],
    position="sidebar"
)


# 사이드바 하단 안내
with st.sidebar:

    st.divider()

    st.caption("📈 금융 지수 데이터 분석 프로젝트")


# ==========================================
# 9. 선택한 페이지 실행
# ==========================================

pg.run()
