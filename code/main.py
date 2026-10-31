import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------------------------------
# 1. 페이지 설정 및 데이터 불러오기
# --------------------------------------------------

st.set_page_config(
    page_title="데이터 전처리",
    page_icon="🧹",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "gosokturn/Analyzing-personal-Data-project/"
    "refs/heads/main/data/indexData.csv"
)


@st.cache_data
def load_data(url):
    data = pd.read_csv(url, encoding="utf-8")
    return data


st.title("🧹 데이터 전처리 실습실")
st.caption("원본 데이터의 문제를 확인하고, 처리 방법을 선택해 전후 결과를 비교합니다.")

try:
    raw_df = load_data(DATA_URL)
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write("GitHub CSV 주소와 파일 공개 여부를 확인해 주세요.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# 2. 전처리 전 원본 데이터 준비
# --------------------------------------------------

before_df = raw_df.copy()

# 날짜 열 변환
if "Date" in before_df.columns:
    before_df["Date"] = pd.to_datetime(
        before_df["Date"],
        errors="coerce"
    )

# 금융 데이터에서 숫자로 다뤄야 하는 열
financial_numeric_cols = [
    "Open", "High", "Low", "Close", "Adj Close", "Volume"
]

for col in financial_numeric_cols:
    if col in before_df.columns:
        before_df[col] = pd.to_numeric(
            before_df[col],
            errors="coerce"
        )

# 숫자형 열 목록
numeric_cols = before_df.select_dtypes(
    include=np.number
).columns.tolist()

# 날짜와 지수 이름 열은 이상치 탐색 대상에서 제외
outlier_candidates = [
    col for col in numeric_cols
    if col.lower() not in ["id", "index"]
]


# --------------------------------------------------
# 3. 원본 데이터 상태 확인
# --------------------------------------------------

st.header("1. 전처리 전 데이터 확인")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("행 개수", f"{len(before_df):,}")

with col2:
    st.metric("열 개수", f"{len(before_df.columns):,}")

with col3:
    st.metric(
        "결측 셀 개수",
        f"{int(before_df.isna().sum().sum()):,}"
    )

with col4:
    st.metric(
        "중복 행 개수",
        f"{int(before_df.duplicated().sum()):,}"
    )

with st.expander("원본 데이터 미리 보기"):
    st.dataframe(before_df.head(20), use_container_width=True)

with st.expander("열별 결측치 확인"):
    missing_table = pd.DataFrame({
        "열 이름": before_df.columns,
        "결측치 개수": [
            int(before_df[col].isna().sum())
            for col in before_df.columns
        ],
        "결측 비율(%)": [
            round(before_df[col].isna().mean() * 100, 2)
            for col in before_df.columns
        ]
    })

    st.dataframe(
        missing_table.sort_values(
            "결측치 개수",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# 4. 결측치와 이상치 구분
# --------------------------------------------------

st.header("2. 결측치와 이상치 확인")

st.markdown("""
- **결측치**: 데이터가 비어 있거나 값이 기록되지 않은 경우이다.
- **이상치**: 값이 기록되어 있지만, 다른 관측값과 비교해 유난히 크거나 작은 경우이다.
- **중요한 차이**: 결측치는 값이 없는 문제이고, 이상치는 값의 분포와 타당성을 살펴보는 문제이다.
""")

st.info(
    "주가 데이터에서 거래량이나 가격이 유난히 크다는 이유만으로 오류라고 단정할 수는 없습니다. "
    "실제 시장 상황을 반영한 값일 수 있으므로, 이상치 후보를 확인한 뒤 처리 방법을 선택합니다."
)

if outlier_candidates:
    selected_outlier_col = st.selectbox(
        "이상치를 확인할 숫자형 열",
        options=outlier_candidates,
        index=(
            outlier_candidates.index("Close")
            if "Close" in outlier_candidates
            else 0
        )
    )

    valid_values = before_df[selected_outlier_col].dropna()

    if len(valid_values) >= 4:
        q1 = valid_values.quantile(0.25)
        q3 = valid_values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (
            (before_df[selected_outlier_col] < lower_bound)
            | (before_df[selected_outlier_col] > upper_bound)
        )

        outlier_mask = outlier_mask.fillna(False)
        outlier_df = before_df.loc[outlier_mask].copy()

        st.subheader(f"'{selected_outlier_col}' 이상치 후보")

        c1, c2, c3 = st.columns(3)

        c1.metric("이상치 후보 개수", f"{len(outlier_df):,}")
        c2.metric("IQR 하한", f"{lower_bound:,.3f}")
        c3.metric("IQR 상한", f"{upper_bound:,.3f}")

        st.caption(
            "IQR 방식은 제1사분위수(Q1)와 제3사분위수(Q3)의 차이를 이용합니다. "
            "Q1 - 1.5×IQR보다 작거나 Q3 + 1.5×IQR보다 큰 값을 이상치 후보로 표시합니다."
        )

        fig = px.box(
            before_df,
            y=selected_outlier_col,
            points="outliers",
            title=f"{selected_outlier_col} 분포와 이상치 후보"
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("이상치 후보 데이터 확인"):
            if outlier_df.empty:
                st.write("이 기준에 해당하는 이상치 후보가 없습니다.")
            else:
                st.dataframe(
                    outlier_df.head(100),
                    use_container_width=True
                )
                st.caption(
                    "최대 100개 행을 표시합니다. 후보 개수와 실제 표시 행 수는 다를 수 있습니다."
                )

    else:
        st.warning(
            "유효한 숫자 데이터가 충분하지 않아 IQR 방식으로 이상치를 확인하기 어렵습니다."
        )

else:
    selected_outlier_col = None
    st.warning("이상치 탐색에 사용할 숫자형 열이 없습니다.")


# --------------------------------------------------
# 5. 데이터 편향을 고려한 점검
# --------------------------------------------------

st.header("3. 데이터 편향 가능성 점검")

st.markdown("""
데이터 편향은 수집된 데이터가 전체 대상을 충분히 대표하지 못하거나,
특정 대상 또는 기간에 관측값이 치우쳐 나타나는 문제이다.
""")

if "Index" in before_df.columns:
    index_counts = (
        before_df["Index"]
        .value_counts(dropna=False)
        .rename_axis("지수")
        .reset_index(name="행 개수")
    )

    st.subheader("지수별 데이터 개수")

    fig_index = px.bar(
        index_counts,
        x="지수",
        y="행 개수",
        title="지수별 관측 행 개수"
    )
    st.plotly_chart(fig_index, use_container_width=True)

    st.dataframe(
        index_counts,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "지수별 행 개수가 다르면 지수마다 분석에 사용할 수 있는 데이터 양이 달라질 수 있습니다. "
        "다만 행 개수가 다르다는 사실만으로 편향이 확정되는 것은 아닙니다."
    )

if "Date" in before_df.columns:
    dated = before_df.dropna(subset=["Date"]).copy()

    if not dated.empty:
        dated["연도"] = dated["Date"].dt.year

        year_counts = (
            dated["연도"]
            .value_counts()
            .sort_index()
            .rename_axis("연도")
            .reset_index(name="행 개수")
        )

        st.subheader("연도별 데이터 개수")

        fig_year = px.bar(
            year_counts,
            x="연도",
            y="행 개수",
            title="연도별 관측 행 개수"
        )
        st.plotly_chart(fig_year, use_container_width=True)

        st.caption(
            "특정 연도의 데이터가 부족하거나 분석 기간이 제한되어 있다면 "
            "전체 시장의 장기적 특성을 대표하지 못할 가능성이 있습니다."
        )


# --------------------------------------------------
# 6. 전처리 방법 선택
# --------------------------------------------------

st.header("4. 전처리 방법 선택")

st.markdown(
    "아래 설정을 바꾸면 선택한 처리 방법에 따라 전처리 결과가 다시 계산됩니다."
)

st.subheader("① 결측치 처리")

missing_method = st.radio(
    "결측치를 어떻게 처리할까요?",
    options=[
        "그대로 유지",
        "결측치가 있는 행 삭제",
        "숫자형은 중앙값, 문자형은 최빈값으로 대체"
    ],
    index=0,
    help=(
        "유지하면 결측치가 남습니다. "
        "행 삭제는 데이터 손실을 유발할 수 있으며, "
        "대체는 실제 관측값이 아닌 추정값을 넣는 방법입니다."
    )
)

st.subheader("② 중복 데이터 처리")

remove_duplicates = st.checkbox(
    "중복 행 제거",
    value=True,
    help="완전히 같은 행이 반복된 경우 첫 번째 행만 남깁니다."
)

if {"Index", "Date"}.issubset(before_df.columns):
    remove_same_date_duplicates = st.checkbox(
        "같은 지수와 날짜의 중복 기록 제거",
        value=False,
        help=(
            "같은 지수와 날짜가 반복되는 행을 제거합니다. "
            "원본에 여러 종류의 기록이 같은 날짜에 존재할 수 있는지 먼저 확인하세요."
        )
    )
else:
    remove_same_date_duplicates = False

st.subheader("③ 이상치 처리")

outlier_method = st.radio(
    "이상치 후보를 어떻게 처리할까요?",
    options=[
        "그대로 유지",
        "IQR 경계값으로 조정",
        "이상치 후보가 있는 행 삭제"
    ],
    index=0,
    help=(
        "금융 데이터의 극단적인 값은 실제 시장 상황일 수 있으므로 "
        "기본값은 유지입니다. 경계값 조정과 행 삭제는 분석 결과를 바꿀 수 있습니다."
    )
)

if not outlier_candidates:
    outlier_method = "그대로 유지"
    st.info("이상치 처리를 선택할 숫자형 열이 없어 이상치 처리는 수행하지 않습니다.")

st.warning(
    "전처리 방법에는 정답 하나만 있는 것이 아닙니다. "
    "분석 목적과 데이터의 의미를 고려해 선택해야 합니다."
)


# --------------------------------------------------
# 7. 선택한 전처리 실행
# --------------------------------------------------

processed_df = before_df.copy()
processing_log = []

# 날짜와 숫자형 데이터 변환
if "Date" in processed_df.columns:
    invalid_dates_before = int(processed_df["Date"].isna().sum())
    processing_log.append(
        f"날짜 변환 후 결측 날짜: {invalid_dates_before:,}개"
    )

# 결측치 처리
if missing_method == "결측치가 있는 행 삭제":
    rows_before = len(processed_df)
    processed_df = processed_df.dropna().copy()
    removed = rows_before - len(processed_df)

    processing_log.append(
        f"결측치가 포함된 행 삭제: {removed:,}행 제거"
    )

elif missing_method == "숫자형은 중앙값, 문자형은 최빈값으로 대체":
    for col in processed_df.columns:
        if pd.api.types.is_numeric_dtype(processed_df[col]):
            median_value = processed_df[col].median()

            if pd.notna(median_value):
                missing_count = int(processed_df[col].isna().sum())
                processed_df[col] = processed_df[col].fillna(median_value)

                if missing_count > 0:
                    processing_log.append(
                        f"{col}: 결측치 {missing_count:,}개를 중앙값으로 대체"
                    )

        else:
            modes = processed_df[col].mode(dropna=True)

            if not modes.empty:
                missing_count = int(processed_df[col].isna().sum())
                processed_df[col] = processed_df[col].fillna(modes.iloc[0])

                if missing_count > 0:
                    processing_log.append(
                        f"{col}: 결측치 {missing_count:,}개를 최빈값으로 대체"
                    )

# 완전히 동일한 행 제거
if remove_duplicates:
    rows_before = len(processed_df)
    processed_df = processed_df.drop_duplicates().copy()
    removed = rows_before - len(processed_df)

    processing_log.append(
        f"완전히 동일한 중복 행 제거: {removed:,}행 제거"
    )

# 같은 지수와 날짜의 중복 기록 제거
if remove_same_date_duplicates:
    rows_before = len(processed_df)

    processed_df = processed_df.drop_duplicates(
        subset=["Index", "Date"],
        keep="first"
    ).copy()

    removed = rows_before - len(processed_df)

    processing_log.append(
        f"지수와 날짜 기준 중복 기록 제거: {removed:,}행 제거"
    )

# 이상치 처리
if (
    selected_outlier_col is not None
    and selected_outlier_col in processed_df.columns
    and outlier_method != "그대로 유지"
):
    values = processed_df[selected_outlier_col].dropna()

    if len(values) >= 4:
        q1_after = values.quantile(0.25)
        q3_after = values.quantile(0.75)
        iqr_after = q3_after - q1_after

        lower_after = q1_after - 1.5 * iqr_after
        upper_after = q3_after + 1.5 * iqr_after

        if outlier_method == "IQR 경계값으로 조정":
            processed_df[selected_outlier_col] = (
                processed_df[selected_outlier_col]
                .clip(lower=lower_after, upper=upper_after)
            )

            processing_log.append(
                f"{selected_outlier_col}: IQR 경계값 "
                f"({lower_after:,.3f}~{upper_after:,.3f})으로 조정"
            )

        elif outlier_method == "이상치 후보가 있는 행 삭제":
            outlier_mask_after = (
                (processed_df[selected_outlier_col] < lower_after)
                | (processed_df[selected_outlier_col] > upper_after)
            )

            outlier_mask_after = outlier_mask_after.fillna(False)

            rows_before = len(processed_df)
            processed_df = processed_df.loc[
                ~outlier_mask_after
            ].copy()

            removed = rows_before - len(processed_df)

            processing_log.append(
                f"{selected_outlier_col}의 IQR 이상치 후보 행 삭제: "
                f"{removed:,}행 제거"
            )

    else:
        processing_log.append(
            f"{selected_outlier_col}: 데이터가 부족해 이상치 처리를 건너뜀"
        )

# 날짜순 정렬
sort_cols = []

if "Index" in processed_df.columns:
    sort_cols.append("Index")

if "Date" in processed_df.columns:
    sort_cols.append("Date")

if sort_cols:
    processed_df = processed_df.sort_values(
        by=sort_cols,
        na_position="last"
    ).reset_index(drop=True)

    processing_log.append(
        f"데이터를 {', '.join(sort_cols)} 기준으로 정렬"
    )


# --------------------------------------------------
# 8. 전처리 전후 결과 비교
# --------------------------------------------------

st.header("5. 전처리 결과 비교")

before_rows = len(before_df)
after_rows = len(processed_df)

before_missing = int(before_df.isna().sum().sum())
after_missing = int(processed_df.isna().sum().sum())

before_duplicates = int(before_df.duplicated().sum())
after_duplicates = int(processed_df.duplicated().sum())

result1, result2, result3, result4 = st.columns(4)

result1.metric(
    "행 개수",
    f"{after_rows:,}",
    delta=f"{after_rows - before_rows:+,}"
)

result2.metric(
    "결측 셀 개수",
    f"{after_missing:,}",
    delta=f"{after_missing - before_missing:+,}",
    delta_color="inverse"
)

result3.metric(
    "완전 중복 행 개수",
    f"{after_duplicates:,}",
    delta=f"{after_duplicates - before_duplicates:+,}",
    delta_color="inverse"
)

result4.metric(
    "열 개수",
    f"{len(processed_df.columns):,}",
    delta=f"{len(processed_df.columns) - len(before_df.columns):+,}"
)

comparison_table = pd.DataFrame({
    "항목": [
        "행 개수",
        "열 개수",
        "결측 셀 개수",
        "완전 중복 행 개수"
    ],
    "전처리 전": [
        before_rows,
        len(before_df.columns),
        before_missing,
        before_duplicates
    ],
    "전처리 후": [
        after_rows,
        len(processed_df.columns),
        after_missing,
        after_duplicates
    ]
})

st.subheader("전처리 전후 요약")

st.dataframe(
    comparison_table,
    use_container_width=True,
    hide_index=True
)

st.subheader("실제로 적용한 처리")

if processing_log:
    for item in processing_log:
        st.write(f"• {item}")
else:
    st.write("선택한 설정에 따라 추가로 적용된 처리 내용이 없습니다.")


# --------------------------------------------------
# 9. 주요 숫자형 변수의 전후 통계 비교
# --------------------------------------------------

st.subheader("숫자형 변수 통계 비교")

if numeric_cols:
    stat_col = st.selectbox(
        "비교할 숫자형 열",
        options=[
            col for col in numeric_cols
            if col in processed_df.columns
        ],
        index=(
            [
                col for col in numeric_cols
                if col in processed_df.columns
            ].index("Close")
            if "Close" in numeric_cols
            and "Close" in processed_df.columns
            else 0
        )
    )

    stats_before = before_df[stat_col].describe()
    stats_after = processed_df[stat_col].describe()

    stats_table = pd.DataFrame({
        "통계량": [
            "유효한 값 개수",
            "평균",
            "중앙값",
            "표준편차",
            "최솟값",
            "최댓값"
        ],
        "전처리 전": [
            before_df[stat_col].count(),
            before_df[stat_col].mean(),
            before_df[stat_col].median(),
            before_df[stat_col].std(),
            before_df[stat_col].min(),
            before_df[stat_col].max()
        ],
        "전처리 후": [
            processed_df[stat_col].count(),
            processed_df[stat_col].mean(),
            processed_df[stat_col].median(),
            processed_df[stat_col].std(),
            processed_df[stat_col].min(),
            processed_df[stat_col].max()
        ]
    })

    st.dataframe(
        stats_table.round(4),
        use_container_width=True,
        hide_index=True
    )

    chart_df = pd.DataFrame({
        "값": pd.concat([
            before_df[stat_col].dropna().reset_index(drop=True),
            processed_df[stat_col].dropna().reset_index(drop=True)
        ], ignore_index=True),
        "구분": (
            ["전처리 전"] * before_df[stat_col].notna().sum()
            + ["전처리 후"] * processed_df[stat_col].notna().sum()
        )
    })

    fig_compare = px.box(
        chart_df,
        x="구분",
        y="값",
        color="구분",
        title=f"{stat_col} 전처리 전후 분포 비교",
        points=False
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True
    )

    st.caption(
        "평균이나 분포가 달라졌다면 전처리 방법이 데이터의 특성에 영향을 주었을 가능성이 있습니다. "
        "변화가 크다는 이유만으로 전처리가 잘못되었다고 단정할 수는 없습니다."
    )

else:
    st.info("전후 통계를 비교할 숫자형 열이 없습니다.")


# --------------------------------------------------
# 10. 처리한 데이터 미리 보기 및 다운로드
# --------------------------------------------------

st.header("6. 전처리된 데이터 확인 및 저장")

st.write(
    f"전처리된 데이터: **{len(processed_df):,}행 × "
    f"{len(processed_df.columns):,}열**"
)

st.dataframe(
    processed_df.head(50),
    use_container_width=True
)

csv_data = processed_df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    label="📥 전처리된 CSV 다운로드",
    data=csv_data,
    file_name="preprocessed_indexData.csv",
    mime="text/csv"
)

st.caption(
    "다운로드 파일은 현재 선택한 전처리 설정을 적용한 결과입니다. "
    "원본 데이터는 GitHub에 있는 CSV를 기준으로 유지됩니다."
)
