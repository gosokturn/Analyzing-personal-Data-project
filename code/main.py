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

# ============================================================
# 7. 전처리 근거 분석 및 전처리 전후 비교
# 기존 main.py의 맨 아래에 추가
# ============================================================

import numpy as np

st.divider()
st.header("7. 데이터 전처리 근거 분석")

st.write(
    "원본 데이터의 구조와 분포를 확인하고, 결측치와 이상치 후보를 구분합니다. "
    "전처리 기준과 전후 통계량을 비교하여 처리 근거를 확인합니다."
)

# ------------------------------------------------------------
# 7-1. 원본 데이터 준비
# ------------------------------------------------------------

# 기존 앱에서 불러온 원본 데이터의 복사본을 사용한다.
audit_raw = original.copy()

# 열 이름 앞뒤 공백 제거
audit_raw.columns = [str(col).strip() for col in audit_raw.columns]

# 열 이름을 대소문자와 관계없이 찾기 위한 함수
def find_column(data, name):
    for col in data.columns:
        if str(col).strip().lower() == name.lower():
            return col
    return None


date_col = find_column(audit_raw, "Date")
index_col = find_column(audit_raw, "Index")
close_col = find_column(audit_raw, "Close")
volume_col = find_column(audit_raw, "Volume")

# ------------------------------------------------------------
# 7-2. 원본 데이터의 행·열 수와 결측치
# ------------------------------------------------------------

st.subheader("7-1. 데이터를 처음 열었을 때 발견한 점")

raw_rows, raw_cols = audit_raw.shape
raw_missing = int(audit_raw.isna().sum().sum())
raw_missing_rate = (
    raw_missing / (raw_rows * raw_cols) * 100
    if raw_rows * raw_cols > 0 else 0
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("원본 행 수", f"{raw_rows:,}행")
m2.metric("원본 열 수", f"{raw_cols:,}열")
m3.metric("결측치 개수", f"{raw_missing:,}개")
m4.metric("결측치 비율", f"{raw_missing_rate:.2f}%")

st.write("### 열별 결측치 현황")

missing_table = pd.DataFrame({
    "열 이름": audit_raw.columns,
    "결측치 개수": [
        int(audit_raw[col].isna().sum())
        for col in audit_raw.columns
    ],
    "결측치 비율(%)": [
        round(float(audit_raw[col].isna().mean() * 100), 2)
        for col in audit_raw.columns
    ],
})

missing_table = missing_table.sort_values(
    "결측치 개수",
    ascending=False
)

st.dataframe(
    missing_table,
    use_container_width=True,
    hide_index=True
)

st.write("### 원본 데이터 미리 보기")
st.dataframe(
    audit_raw.head(10),
    use_container_width=True
)

# ------------------------------------------------------------
# 7-3. 결측치와 이상치 후보의 차이
# ------------------------------------------------------------

st.subheader("7-2. 결측치와 이상치 후보 구분하기")

st.markdown("""
**결측치**는 값이 비어 있거나 기록되지 않은 경우이다.
예를 들어 `Close` 열에 종가가 기록되지 않았다면 결측치이다.

**이상치 후보**는 값이 존재하지만 해당 데이터의 일반적인 분포에서
매우 멀리 떨어져 있는 값이다.
예를 들어 `Volume` 값이 다른 날짜에 비해 매우 크다면 이상치 후보가 될 수 있다.

두 개념은 다르다. 결측치는 값의 부재이고, 이상치는 값이 존재하지만
분포상 극단적인 경우이다. 이상치 후보가 실제 오류라는 뜻은 아니다.
""")

# 수치형 열을 숫자로 변환할 수 있는지 확인한다.
# 원본 자체는 변경하지 않는다.
audit_numeric = audit_raw.copy()

numeric_candidates = []

for col in audit_numeric.columns:
    converted = pd.to_numeric(audit_numeric[col], errors="coerce")

    # 숫자로 변환되는 값이 하나 이상 있는 열을 후보로 선택
    if converted.notna().any():
        numeric_candidates.append(col)

if numeric_candidates:
    selected_col = st.selectbox(
        "분포를 확인할 열을 선택하세요.",
        numeric_candidates,
        key="audit_distribution_column"
    )

    selected_values = pd.to_numeric(
        audit_numeric[selected_col],
        errors="coerce"
    ).dropna()

    if not selected_values.empty:
        q1 = selected_values.quantile(0.25)
        q3 = selected_values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (
            (selected_values < lower_bound)
            | (selected_values > upper_bound)
        )

        outlier_count = int(outlier_mask.sum())

        left, right = st.columns(2)

        with left:
            hist_fig = px.histogram(
                x=selected_values,
                nbins=50,
                title=f"{selected_col} 히스토그램",
                labels={
                    "x": selected_col,
                    "y": "빈도"
                }
            )
            st.plotly_chart(
                hist_fig,
                use_container_width=True
            )

        with right:
            box_fig = px.box(
                x=selected_values,
                title=f"{selected_col} 상자그림",
                labels={"x": selected_col}
            )
            st.plotly_chart(
                box_fig,
                use_container_width=True
            )

        st.write("### 선택한 열의 통계 결과")

        s1, s2, s3, s4 = st.columns(4)

        s1.metric("유효한 값 개수", f"{len(selected_values):,}")
        s2.metric("평균", f"{selected_values.mean():,.4f}")
        s3.metric("중앙값", f"{selected_values.median():,.4f}")
        s4.metric("IQR 기준 이상치 후보", f"{outlier_count:,}")

        st.write(
            f"1사분위수(Q1): {q1:,.4f} / "
            f"3사분위수(Q3): {q3:,.4f}"
        )

        st.write(
            f"이상치 후보 판단 범위: "
            f"{lower_bound:,.4f} 미만 또는 "
            f"{upper_bound:,.4f} 초과"
        )

        st.info(
            "이상치 후보는 IQR(사분위수 범위) 기준으로 계산했습니다. "
            "이 값은 자동 삭제하지 않습니다. "
            "주가지수의 급격한 가격 변화나 거래량 증가는 실제 시장 상황일 수 있기 때문입니다."
        )

else:
    st.warning("분포를 확인할 수 있는 수치형 열이 없습니다.")

# ------------------------------------------------------------
# 7-4. 편향 가능성 확인
# ------------------------------------------------------------

st.subheader("7-3. 데이터 편향 가능성 확인")

st.markdown("""
**편향**은 데이터가 전체 상황을 충분히 대표하지 못해
분석 결과가 특정 방향으로 치우치는 현상이다.

주가지수 데이터에서는 다음과 같은 편향 가능성을 확인할 수 있다.

- 지수마다 데이터 행 수가 크게 다를 수 있다.
- 특정 연도의 데이터가 다른 기간보다 적을 수 있다.
- 특정 지수나 기간에 대한 결과가 전체 시장의 특성처럼 해석될 수 있다.
""")

if index_col:
    st.write("### 지수별 데이터 개수")

    index_counts = (
        audit_raw[index_col]
        .value_counts(dropna=False)
        .rename_axis("지수")
        .reset_index(name="데이터 개수")
    )

    st.dataframe(
        index_counts,
        use_container_width=True,
        hide_index=True
    )

    index_fig = px.bar(
        index_counts,
        x="지수",
        y="데이터 개수",
        title="지수별 데이터 개수 비교"
    )

    st.plotly_chart(
        index_fig,
        use_container_width=True
    )

    st.caption(
        "지수별 데이터 개수가 다르다는 사실만으로 편향이 확정되는 것은 아닙니다. "
        "지수별 관측 기간과 데이터 수집 방식을 함께 확인해야 합니다."
    )

if date_col:
    year_data = audit_raw.copy()
    year_data[date_col] = pd.to_datetime(
        year_data[date_col],
        errors="coerce"
    )

    year_data = year_data.dropna(subset=[date_col])
    year_data["연도"] = year_data[date_col].dt.year

    yearly_counts = (
        year_data["연도"]
        .value_counts()
        .sort_index()
        .rename_axis("연도")
        .reset_index(name="데이터 개수")
    )

    st.write("### 연도별 데이터 개수")

    year_fig = px.bar(
        yearly_counts,
        x="연도",
        y="데이터 개수",
        title="연도별 관측 데이터 개수"
    )

    st.plotly_chart(
        year_fig,
        use_container_width=True
    )

    st.caption(
        "연도별 데이터 수가 크게 다르면 기간별 비교에 영향을 줄 수 있습니다. "
        "거래일 수 차이 때문인지 실제 누락 때문인지 구분해야 합니다."
    )

# ------------------------------------------------------------
# 7-5. 전처리 기준 적용
# ------------------------------------------------------------

st.subheader("7-4. 결측치와 중복 데이터 전처리")

st.markdown("""
이번 전처리에서는 다음 기준을 적용한다.

1. 날짜가 있는 경우 날짜를 날짜형으로 변환한다.
2. 종가가 있는 경우 숫자형으로 변환한다.
3. 날짜가 없거나 해석할 수 없으면 날짜순 분석에 사용할 수 없으므로 제외한다.
4. 종가가 없거나 숫자로 변환할 수 없으면 종가 분석에서 제외한다.
5. 지수명과 날짜가 모두 있는 경우, 같은 지수의 같은 날짜가 중복된 행은 첫 번째 행만 남긴다.
6. IQR 기준 이상치 후보는 자동 삭제하지 않는다. 실제 시장 변동일 가능성을 확인해야 하기 때문이다.
7. 다른 열의 결측치는 무조건 채우지 않는다. 해당 열이 필요한 분석을 할 때 별도로 처리한다.
""")

audit_clean = audit_raw.copy()

# 날짜를 날짜형으로 변환
if date_col:
    audit_clean[date_col] = pd.to_datetime(
        audit_clean[date_col],
        errors="coerce"
    )

# 종가를 숫자형으로 변환
if close_col:
    audit_clean[close_col] = pd.to_numeric(
        audit_clean[close_col],
        errors="coerce"
    )

rows_before_filter = len(audit_clean)

# 날짜가 있으면 유효한 날짜가 있는 행만 남긴다.
if date_col:
    audit_clean = audit_clean.dropna(subset=[date_col])

# 종가가 있으면 종가가 유효한 행만 남긴다.
if close_col:
    audit_clean = audit_clean.dropna(subset=[close_col])

rows_after_missing = len(audit_clean)
removed_missing_rows = rows_before_filter - rows_after_missing

# 지수명과 날짜가 함께 있을 때만 복합키 중복을 확인한다.
removed_duplicate_rows = 0

if index_col and date_col:
    duplicate_mask = audit_clean.duplicated(
        subset=[index_col, date_col],
        keep="first"
    )

    removed_duplicate_rows = int(duplicate_mask.sum())

    audit_clean = audit_clean.loc[
        ~duplicate_mask
    ].copy()

# 날짜순 정렬
if date_col:
    sort_cols = []

    if index_col:
        sort_cols.append(index_col)

    sort_cols.append(date_col)

    audit_clean = audit_clean.sort_values(
        sort_cols
    ).reset_index(drop=True)

rows_after = len(audit_clean)

st.write("### 전처리 처리 내역")

processing_log = pd.DataFrame([
    {
        "처리 항목": "날짜 변환",
        "처리 기준": "Date 열을 날짜형으로 변환",
        "결과": "날짜순 정렬과 기간별 분석을 위해 적용"
    },
    {
        "처리 항목": "종가 변환",
        "처리 기준": "Close 열을 숫자형으로 변환",
        "결과": "종가 통계와 수익률 계산을 위해 적용"
    },
    {
        "처리 항목": "핵심 열 결측치 처리",
        "처리 기준": "날짜 또는 종가가 유효하지 않은 행 제외",
        "결과": f"{removed_missing_rows:,}행 제외"
    },
    {
        "처리 항목": "중복 데이터 처리",
        "처리 기준": "지수명과 날짜가 같은 행은 첫 행만 유지",
        "결과": f"{removed_duplicate_rows:,}행 제외"
    },
    {
        "처리 항목": "이상치 후보 처리",
        "처리 기준": "IQR 기준으로 후보를 확인하되 자동 삭제하지 않음",
        "결과": "실제 시장 변동일 가능성이 있어 원본 관측값 유지"
    }
])

st.dataframe(
    processing_log,
    use_container_width=True,
    hide_index=True
)

# ------------------------------------------------------------
# 7-6. 전처리 전후 행 수와 평균 비교
# ------------------------------------------------------------

st.subheader("7-5. 전처리 전후 비교")

comparison_rows = [
    {
        "비교 항목": "행 수",
        "전처리 전": len(audit_raw),
        "전처리 후": len(audit_clean)
    },
    {
        "비교 항목": "열 수",
        "전처리 전": audit_raw.shape[1],
        "전처리 후": audit_clean.shape[1]
    },
    {
        "비교 항목": "전체 결측 셀 수",
        "전처리 전": int(audit_raw.isna().sum().sum()),
        "전처리 후": int(audit_clean.isna().sum().sum())
    },
    {
        "비교 항목": "완전 중복 행 수",
        "전처리 전": int(audit_raw.duplicated().sum()),
        "전처리 후": int(audit_clean.duplicated().sum())
    }
]

comparison_table = pd.DataFrame(comparison_rows)

st.dataframe(
    comparison_table,
    use_container_width=True,
    hide_index=True
)

if close_col:
    # 원본의 종가도 숫자로 변환해 비교한다.
    raw_close = pd.to_numeric(
        audit_raw[close_col],
        errors="coerce"
    ).dropna()

    clean_close = pd.to_numeric(
        audit_clean[close_col],
        errors="coerce"
    ).dropna()

    avg_comparison = pd.DataFrame([
        {
            "통계량": "종가 평균",
            "전처리 전": (
                round(float(raw_close.mean()), 4)
                if not raw_close.empty else np.nan
            ),
            "전처리 후": (
                round(float(clean_close.mean()), 4)
                if not clean_close.empty else np.nan
            )
        },
        {
            "통계량": "종가 중앙값",
            "전처리 전": (
                round(float(raw_close.median()), 4)
                if not raw_close.empty else np.nan
            ),
            "전처리 후": (
                round(float(clean_close.median()), 4)
                if not clean_close.empty else np.nan
            )
        },
        {
            "통계량": "종가 표준편차",
            "전처리 전": (
                round(float(raw_close.std()), 4)
                if len(raw_close) > 1 else np.nan
            ),
            "전처리 후": (
                round(float(clean_close.std()), 4)
                if len(clean_close) > 1 else np.nan
            )
        }
    ])

    st.write("### 종가 통계량 비교")

    st.dataframe(
        avg_comparison,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "전처리 전후 평균 차이는 결측값 및 중복 데이터 제거의 영향을 포함할 수 있습니다. "
        "이상치 후보는 삭제하지 않았으므로 이상치 제거 효과를 나타내는 비교는 아닙니다."
    )

# ------------------------------------------------------------
# 7-7. 전처리 결과 다운로드
# ------------------------------------------------------------

st.subheader("7-6. 전처리 결과 다운로드")

st.download_button(
    label="전처리 데이터 CSV 다운로드",
    data=audit_clean.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="preprocessed_indexData.csv",
    mime="text/csv",
    key="download_preprocessed_indexData"
)

st.success(
    f"전처리 완료: 원본 {len(audit_raw):,}행에서 "
    f"전처리 후 {len(audit_clean):,}행으로 변경되었습니다."
)

st.caption(
    "주의: 이 코드는 원본 데이터 파일을 덮어쓰지 않습니다. "
    "전처리 결과는 별도 데이터프레임으로 생성되며, "
    "IQR 이상치 후보는 자동 삭제하지 않습니다."
)
