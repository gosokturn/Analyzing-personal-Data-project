import streamlit as st
import pandas as pd
import plotly.express as px
from urllib.error import URLError

st.set_page_config(
    page_title="주가지수 데이터 기초통계 분석",
    page_icon="📊",
    layout="wide",
)

st.title("📊 주가지수 데이터 기초통계 분석")
st.caption("문제 A: 과거 주가지수 데이터로 시장 급락 위험을 살펴보기 위한 기초 데이터 점검")

st.markdown(
    """
이 앱은 GitHub의 **Raw CSV 주소**를 불러와 데이터의 크기, 결측치, 자료형,
중앙값을 포함한 기초통계량을 확인합니다. 주가지수 급락 위험 분석에 앞서
데이터의 품질과 구조를 점검하는 데 사용할 수 있습니다.
"""
)

DEFAULT_URL = ""
with st.sidebar:
    st.header("데이터 불러오기")
    raw_url = st.text_input(
        "GitHub Raw CSV 주소",
        value=DEFAULT_URL,
        placeholder="https://raw.githubusercontent.com/사용자/저장소/브랜치/파일.csv",
        help="GitHub에서 CSV 파일을 연 뒤 Raw를 눌러 나온 주소를 붙여 넣으세요.",
    )
    load_button = st.button("데이터 불러오기", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**주소 확인 방법**")
    st.markdown("1. GitHub 저장소에서 CSV 파일 열기\n2. `Raw` 버튼 누르기\n3. 브라우저 주소를 복사해 위 칸에 붙여 넣기")
    st.caption("공개적으로 접근 가능한 CSV 주소가 필요합니다.")

if "data_url" not in st.session_state:
    st.session_state.data_url = ""
if load_button:
    st.session_state.data_url = raw_url.strip()

@st.cache_data(show_spinner="CSV 데이터를 불러오는 중입니다...")
def load_csv(url: str) -> pd.DataFrame:
    if not url:
        raise ValueError("먼저 GitHub Raw CSV 주소를 입력하세요.")
    # GitHub Raw CSV는 보통 UTF-8 인코딩이다. 실패할 경우를 고려해 인코딩을 순서대로 시도한다.
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(url, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return pd.read_csv(url, low_memory=False)

if not st.session_state.data_url:
    st.info("왼쪽에 GitHub Raw CSV 주소를 입력한 다음 **데이터 불러오기**를 누르세요.")
    st.markdown("### 이 앱에서 확인할 수 있는 정보")
    c1, c2, c3 = st.columns(3)
    c1.metric("데이터 크기", "행 · 열 개수")
    c2.metric("데이터 품질", "결측치 · 중복 행")
    c3.metric("기초통계량", "평균 · 중앙값 · 사분위수")
    st.stop()

try:
    df = load_csv(st.session_state.data_url)
except Exception as exc:
    st.error("데이터를 불러오지 못했습니다.")
    st.write("주소가 Raw CSV 주소인지, 파일이 공개되어 있는지 확인하세요.")
    st.caption(f"오류 유형: {type(exc).__name__}")
    with st.expander("오류 세부 정보"):
        st.code(str(exc))
    st.stop()

if df.empty:
    st.warning("CSV 파일에 데이터 행이 없습니다.")
    st.stop()

# 기본 정리: 열 이름의 앞뒤 공백 제거
df.columns = [str(col).strip() for col in df.columns]

n_rows, n_cols = df.shape
missing_cells = int(df.isna().sum().sum())
total_cells = n_rows * n_cols
missing_pct = (missing_cells / total_cells * 100) if total_cells else 0
duplicate_rows = int(df.duplicated().sum())

st.success(f"데이터를 불러왔습니다. 행 {n_rows:,}개, 열 {n_cols:,}개입니다.")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("행 개수", f"{n_rows:,}")
m2.metric("열 개수", f"{n_cols:,}")
m3.metric("전체 결측 셀", f"{missing_cells:,}")
m4.metric("결측 비율", f"{missing_pct:.2f}%")
m5.metric("중복 행", f"{duplicate_rows:,}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["데이터 미리보기", "결측치·자료형", "기초통계량", "분포 살펴보기"]
)

with tab1:
    st.subheader("데이터 미리보기")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        preview_rows = st.selectbox("미리 볼 행 수", [5, 10, 20, 50, 100], index=1)
    with col_b:
        st.write(f"전체 데이터 크기: **{n_rows:,}행 × {n_cols:,}열**")
    st.dataframe(df.head(preview_rows), use_container_width=True, hide_index=True)

    with st.expander("열 이름 목록"):
        st.write(list(df.columns))

    st.subheader("중복 행 확인")
    st.write(f"모든 열의 값이 완전히 같은 중복 행은 **{duplicate_rows:,}개**입니다.")
    if duplicate_rows:
        st.dataframe(df[df.duplicated(keep=False)].head(100), use_container_width=True)

with tab2:
    st.subheader("열별 결측치와 자료형")
    missing_summary = pd.DataFrame({
        "열 이름": df.columns,
        "자료형": [str(dtype) for dtype in df.dtypes],
        "결측치 개수": [int(df[col].isna().sum()) for col in df.columns],
        "결측치 비율(%)": [
            round(float(df[col].isna().mean() * 100), 2) for col in df.columns
        ],
        "고유값 개수": [int(df[col].nunique(dropna=True)) for col in df.columns],
    }).sort_values("결측치 개수", ascending=False)
    st.dataframe(missing_summary, use_container_width=True, hide_index=True)

    fig_missing = px.bar(
        missing_summary.sort_values("결측치 개수", ascending=True),
        x="결측치 개수",
        y="열 이름",
        orientation="h",
        text="결측치 개수",
        title="열별 결측치 개수",
    )
    fig_missing.update_layout(yaxis_title="", xaxis_title="결측치 개수")
    st.plotly_chart(fig_missing, use_container_width=True)

    all_missing = missing_summary[missing_summary["결측치 개수"] > 0]
    if all_missing.empty:
        st.success("결측치가 없습니다.")
    else:
        st.warning(
            f"결측치가 있는 열은 {len(all_missing)}개입니다. "
            "분석 전에 결측치가 발생한 이유와 처리 방법을 검토하세요."
        )

with tab3:
    st.subheader("기초통계량")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        stats = df[numeric_cols].describe().T
        stats["중앙값"] = df[numeric_cols].median()
        stats["결측치 개수"] = df[numeric_cols].isna().sum()
        stats = stats.rename(columns={
            "count": "유효 데이터 수",
            "mean": "평균",
            "std": "표준편차",
            "min": "최솟값",
            "25%": "제1사분위수(Q1)",
            "50%": "중앙값(50%)",
            "75%": "제3사분위수(Q3)",
            "max": "최댓값",
        })
        preferred = [
            "유효 데이터 수", "평균", "중앙값", "중앙값(50%)", "표준편차",
            "최솟값", "제1사분위수(Q1)", "제3사분위수(Q3)", "최댓값", "결측치 개수"
        ]
        stats = stats[[col for col in preferred if col in stats.columns]]
        st.dataframe(stats.round(4), use_container_width=True)

        st.download_button(
            "기초통계량 CSV 다운로드",
            data=stats.to_csv(index=True, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="기초통계량.csv",
            mime="text/csv",
        )
    else:
        st.info("숫자형 열이 없어 수치형 기초통계량을 계산할 수 없습니다.")

    st.subheader("범주형 열 요약")
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    if categorical_cols:
        cat_rows = []
        for col in categorical_cols:
            modes = df[col].mode(dropna=True)
            cat_rows.append({
                "열 이름": col,
                "고유값 개수": int(df[col].nunique(dropna=True)),
                "최빈값": str(modes.iloc[0]) if not modes.empty else "없음",
                "최빈값 빈도": int((df[col] == modes.iloc[0]).sum()) if not modes.empty else 0,
                "결측치 개수": int(df[col].isna().sum()),
            })
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)
    else:
        st.info("범주형 열이 없습니다.")

with tab4:
    st.subheader("숫자형 변수 분포")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.info("분포 그래프를 그릴 수 있는 숫자형 열이 없습니다.")
    else:
        selected_col = st.selectbox("분석할 숫자형 열", numeric_cols)
        chart_df = df[[selected_col]].dropna()
        if chart_df.empty:
            st.warning("선택한 열에 유효한 숫자 데이터가 없습니다.")
        else:
            g1, g2 = st.columns(2)
            with g1:
                fig_hist = px.histogram(
                    chart_df, x=selected_col, nbins=40,
                    title=f"{selected_col}: 값의 분포",
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            with g2:
                fig_box = px.box(
                    chart_df, y=selected_col,
                    title=f"{selected_col}: 상자그림",
                )
                st.plotly_chart(fig_box, use_container_width=True)

            values = chart_df[selected_col]
            q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("평균", f"{values.mean():,.4f}")
            s2.metric("중앙값", f"{median:,.4f}")
            s3.metric("표준편차", f"{values.std():,.4f}" if len(values) > 1 else "계산 불가")
            s4.metric("IQR (Q3-Q1)", f"{q3-q1:,.4f}")

st.divider()
st.subheader("주식시장 급락 위험 분석을 위한 참고")
st.markdown(
    """
- **기초통계만으로 급락 위험을 확정할 수는 없습니다.** 평균과 중앙값은 전체적인 분포를 요약할 뿐입니다.
- 실제 급락 위험을 탐구하려면 날짜별 종가로 일별 수익률을 계산하고, 최고점 대비 하락률(낙폭)을 추가로 분석하는 것이 좋습니다.
- 서로 다른 주가지수는 기준값과 관측 기간이 다를 수 있으므로, 지수의 원래 숫자 크기만으로 시장을 비교하지 마세요.
"""
)
st.caption("분석 결과는 입력한 CSV 파일을 기준으로 계산됩니다. 데이터의 출처와 이용 조건은 별도로 확인해야 합니다.")
