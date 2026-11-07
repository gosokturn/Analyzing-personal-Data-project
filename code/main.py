
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="세계 금융시장 분석",
    page_icon="🌍",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR / "pages"


def home_page():
    from market_utils import load_raw_data

    st.title("🌍 세계 금융시장 분석")
    st.write(
        "국가별·시장별 주가지수 데이터를 분석하여 "
        "시장 유형과 위험 특성을 파악하는 프로젝트입니다."
    )

    st.info(
        "왼쪽 메뉴에서 데이터 탐색, 데이터 전처리, "
        "모델 학습 결과를 확인할 수 있습니다."
    )

    try:
        df = load_raw_data()
    except Exception as e:
        st.error("원본 CSV를 불러오지 못했습니다.")
        st.exception(e)
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("전체 데이터 행", f"{len(df):,}")
    c2.metric("데이터 열", f"{len(df.columns):,}")
    c3.metric(
        "결측 셀",
        f"{int(df.isna().sum().sum()):,}"
    )

    st.subheader("원본 데이터 미리 보기")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("데이터 열 정보")
    schema = pd.DataFrame({
        "열 이름": df.columns,
        "데이터 유형": [
            str(df[col].dtype) for col in df.columns
        ],
        "결측치": [
            int(df[col].isna().sum()) for col in df.columns
        ]
    })
    st.dataframe(
        schema,
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader("분석 목표")
    st.markdown("""
    1. **시장 그룹화:** 비슷한 수익률과 위험 특성을 가진 시장을 묶는다.
    2. **상관관계 분석:** 서로 비슷한 방향으로 움직이는 시장을 찾는다.
    3. **하락 위험 분석:** 과거 데이터에서 큰 하락이 나타나는 특성을 살펴본다.
    4. **경제 자료 비교:** 국가별 거시경제 자료를 추가해 시장 특성과 비교한다.
    """)

    st.warning(
        "이 프로젝트는 통계와 머신러닝을 학습하기 위한 분석 도구이며, "
        "미래 수익이나 손실 방지를 보장하지 않습니다."
    )


if not PAGES_DIR.exists():
    st.error("pages 폴더를 찾을 수 없습니다.")
    st.stop()

required_pages = [
    "1_탐색.py",
    "2_데이터_전처리.py",
    "3_모델_학습_결과.py"
]

for filename in required_pages:
    if not (PAGES_DIR / filename).is_file():
        st.error(f"페이지 파일이 없습니다: pages/{filename}")
        st.stop()

home = st.Page(
    home_page,
    title="홈",
    icon="🏠",
    default=True
)

explore = st.Page(
    str(PAGES_DIR / "1_탐색.py"),
    title="데이터 탐색",
    icon="🔍"
)

preprocess = st.Page(
    str(PAGES_DIR / "2_데이터_전처리.py"),
    title="데이터 전처리",
    icon="🧹"
)

model = st.Page(
    str(PAGES_DIR / "3_모델_학습_결과.py"),
    title="모델 학습 결과",
    icon="🧠"
)

pg = st.navigation(
    {
        "데이터 분석": [home, explore, preprocess],
        "머신러닝": [model]
    },
    position="sidebar"
)

with st.sidebar:
    st.divider()
    st.caption("세계 금융시장 분석 프로젝트")

pg.run()
