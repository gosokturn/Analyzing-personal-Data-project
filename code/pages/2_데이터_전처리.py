import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
    return pd.read_csv(url, encoding="utf-8")


st.title("🧹 데이터 전처리 실습실")
st.caption(
    "원본 데이터의 문제를 확인하고, "
    "전처리 방법을 선택해 결과를 비교합니다."
)

try:
    raw_df = load_data(DATA_URL)
except Exception as e:
    st.error("GitHub에서 CSV 데이터를 불러오지 못했습니다.")
    st.write("데이터 주소와 저장소의 공개 설정을 확인해 주세요.")
    st.exception(e)
    st.stop()


# ----------------------------------------
# 1. 원본 데이터 준비
# ----------------------------------------

before_df = raw_df.copy()

if "Date" in before_df.columns:
    before_df["Date"] = pd.to_datetime(
        before_df["Date"],
        errors="coerce"
    )

for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
    if col in before_df.columns:
        before_df[col] = pd.to_numeric(
            before_df[col],
            errors="coerce"
        )

numeric_cols = before_df.select_dtypes(
    include=np.number
).columns.tolist()

outlier_candidates = [
    col for col in numeric_cols
    if col.lower() not in ["id", "index"]
]


# ----------------------------------------
# 2. 원본 데이터 상태
# ----------------------------------------

st.header("1. 원본 데이터 상태")

col1, col2, col3, col4 = st.columns(4)

col1.metric("행 개수", f"{len(before_df):,}")
col2.metric("열 개수", f"{len(before_df.columns):,}")
col3.metric(
    "결측 셀 개수",
    f"{int(before_df.isna().sum().sum()):,}"
)
col4.metric(
    "완전 중복 행",
    f"{int(before_df.duplicated().sum()):,}"
)

with st.expander("원본 데이터 미리 보기"):
    st.dataframe(
        before_df.head(20),
        use_container_width=True
    )

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

st.subheader("열별 결측치")

st.dataframe(
    missing_table.sort_values(
        "결측치 개수",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# ----------------------------------------
# 3. 결측치와 이상치 확인
# ----------------------------------------

st.header("2. 결측치와 이상치 확인")

st.markdown("""
**결측치**는 값이 비어 있거나 기록되지 않은 경우이다.

**이상치**는 값이 존재하지만 다른 관측값과 비교했을 때
유난히 크거나 작은 경우이다.

결측치는 값이 없는 문제이고, 이상치는 값의 분포와 타당성을
확인해야 하는 문제라는 차이가 있다.
""")

st.info(
    "금융 데이터의 큰 가격 변동이나 거래량은 실제 시장 상황일 수 있습니다. "
    "이상치 후보라고 해서 반드시 잘못된 데이터는 아닙니다."
)

selected_outlier_col = None

if outlier_candidates:

    default_idx = (
        outlier_candidates.index("Close")
        if "Close" in outlier_candidates
        else 0
    )

    selected_outlier_col = st.selectbox(
        "이상치 후보를 확인할 숫자형 열",
        options=outlier_candidates,
        index=default_idx
    )

    values = before_df[selected_outlier_col].dropna()

    if len(values) >= 4:

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_mask = (
            (before_df[selected_outlier_col] < lower)
            | (before_df[selected_outlier_col] > upper)
        ).fillna(False)

        outlier_df = before_df.loc[outlier_mask].copy()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "이상치 후보 개수",
            f"{len(outlier_df):,}"
        )

        c2.metric(
            "IQR 하한",
            f"{lower:,.3f}"
        )

        c3.metric(
            "IQR 상한",
            f"{upper:,.3f}"
        )

        st.caption(
            "IQR = 제3사분위수(Q3) - 제1사분위수(Q1). "
            "Q1 - 1.5×IQR보다 작거나 Q3 + 1.5×IQR보다 큰 값을 "
            "이상치 후보로 표시합니다."
        )

        fig_box = px.box(
            before_df,
            y=selected_outlier_col,
            points="outliers",
            title=f"{selected_outlier_col} 분포와 이상치 후보"
        )

        st.plotly_chart(
            fig_box,
            use_container_width=True
        )

        with st.expander("이상치 후보 데이터 확인"):

            if outlier_df.empty:
                st.write("이 기준에 해당하는 이상치 후보가 없습니다.")
            else:
                st.dataframe(
                    outlier_df.head(100),
                    use_container_width=True
                )

                st.caption(
                    "최대 100개 행을 표시합니다."
                )

    else:
        st.warning(
            "유효한 숫자 데이터가 부족해 IQR 분석을 수행할 수 없습니다."
        )

else:
    st.info("이상치 탐색에 사용할 숫자형 열이 없습니다.")


# ----------------------------------------
# 4. 데이터 편향 가능성 확인
# ----------------------------------------

st.header("3. 데이터 편향 가능성 점검")

st.markdown("""
데이터 편향은 수집된 데이터가 전체 대상을 충분히 대표하지 못하거나,
특정 대상 또는 기간에 관측값이 치우쳐 나타나는 문제이다.

지수별·연도별 데이터 개수를 비교하여 데이터가 특정 지수나 기간에
집중되어 있는지 확인한다.
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

    st.plotly_chart(
        fig_index,
        use_container_width=True
    )

    st.dataframe(
        index_counts,
        use_container_width=True,
        hide_index=True
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

        st.plotly_chart(
            fig_year,
            use_container_width=True
        )

st.caption(
    "지수별·연도별 행 개수가 다르다는 사실만으로 편향이 확정되는 것은 아닙니다. "
    "데이터 수집 기간과 분석 목적을 함께 고려해야 합니다."
)


# ----------------------------------------
# 5. 전처리 방법 선택
# ----------------------------------------

st.header("4. 전처리 방법 선택")

st.subheader("① 결측치 처리")

missing_method = st.radio(
    "결측치를 어떻게 처리할까요?",
    options=[
        "그대로 유지",
        "결측치가 있는 행 삭제",
        "숫자형은 중앙값, 문자형은 최빈값으로 대체"
    ],
    index=0
)

st.caption(
    "행 삭제는 데이터 손실을 유발할 수 있습니다. "
    "대체 방법은 실제 관측값이 아닌 추정값을 넣는 방법입니다."
)

st.subheader("② 중복 데이터 처리")

remove_duplicates = st.checkbox(
    "완전히 동일한 중복 행 제거",
    value=True
)

remove_same_date_duplicates = False

if {"Index", "Date"}.issubset(before_df.columns):

    remove_same_date_duplicates = st.checkbox(
        "같은 지수와 날짜의 중복 기록 제거",
        value=False,
        help=(
            "같은 지수와 날짜가 반복되는 행을 제거합니다. "
            "중복 기록이 실제로 불필요한지 확인한 뒤 선택하세요."
        )
    )

st.subheader("③ 이상치 처리")

outlier_method = st.radio(
    "이상치 후보를 어떻게 처리할까요?",
    options=[
        "그대로 유지",
        "IQR 경계값으로 조정",
        "이상치 후보가 있는 행 삭제"
    ],
    index=0
)

st.warning(
    "이상치 후보는 반드시 오류가 아닙니다. "
    "금융 데이터에서는 실제 시장 변화를 반영할 수 있으므로 "
    "기본 설정은 유지입니다."
)


# ----------------------------------------
# 6. 선택한 전처리 실행
# ----------------------------------------

processed_df = before_df.copy()
processing_log = []

# 결측치 처리

if missing_method == "결측치가 있는 행 삭제":

    old_len = len(processed_df)

    processed_df = processed_df.dropna().copy()

    removed = old_len - len(processed_df)

    processing_log.append(
        f"결측치가 포함된 행 {removed:,}개 삭제"
    )

elif missing_method == "숫자형은 중앙값, 문자형은 최빈값으로 대체":

    for col in processed_df.columns:

        missing_count = int(processed_df[col].isna().sum())

        if missing_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(processed_df[col]):

            fill_value = processed_df[col].median()
            method_name = "중앙값"

        else:

            modes = processed_df[col].mode(dropna=True)

            if modes.empty:
                continue

            fill_value = modes.iloc[0]
            method_name = "최빈값"

        if pd.notna(fill_value):

            processed_df[col] = processed_df[col].fillna(
                fill_value
            )

            processing_log.append(
                f"{col}: 결측치 {missing_count:,}개를 "
                f"{method_name}으로 대체"
            )


# 완전히 동일한 행 제거

if remove_duplicates:

    old_len = len(processed_df)

    processed_df = processed_df.drop_duplicates().copy()

    removed = old_len - len(processed_df)

    processing_log.append(
        f"완전히 동일한 중복 행 {removed:,}개 제거"
    )


# 같은 지수와 날짜의 중복 기록 제거

if remove_same_date_duplicates:

    old_len = len(processed_df)

    processed_df = processed_df.drop_duplicates(
        subset=["Index", "Date"],
        keep="first"
    ).copy()

    removed = old_len - len(processed_df)

    processing_log.append(
        f"지수와 날짜 기준 중복 기록 {removed:,}개 제거"
    )


# 이상치 처리

if (
    selected_outlier_col is not None
    and selected_outlier_col in processed_df.columns
    and outlier_method != "그대로 유지"
):

    values = processed_df[selected_outlier_col].dropna()

    if len(values) >= 4:

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        if outlier_method == "IQR 경계값으로 조정":

            processed_df[selected_outlier_col] = (
                processed_df[selected_outlier_col]
                .clip(lower=lower, upper=upper)
            )

            processing_log.append(
                f"{selected_outlier_col}: "
                "이상치 후보를 IQR 경계값으로 조정"
            )

        elif outlier_method == "이상치 후보가 있는 행 삭제":

            mask = (
                (processed_df[selected_outlier_col] < lower)
                | (processed_df[selected_outlier_col] > upper)
            ).fillna(False)

            old_len = len(processed_df)

            processed_df = processed_df.loc[~mask].copy()

            removed = old_len - len(processed_df)

            processing_log.append(
                f"{selected_outlier_col}: "
                f"이상치 후보 행 {removed:,}개 삭제"
            )

    else:

        processing_log.append(
            f"{selected_outlier_col}: 데이터가 부족해 "
            "이상치 처리를 건너뜀"
        )


# 날짜 및 지수 기준 정렬

sort_cols = [
    col for col in ["Index", "Date"]
    if col in processed_df.columns
]

if sort_cols:

    processed_df = processed_df.sort_values(
        by=sort_cols,
        na_position="last"
    ).reset_index(drop=True)

    processing_log.append(
        f"{', '.join(sort_cols)} 기준으로 데이터 정렬"
    )


# ----------------------------------------
# 7. 전처리 전후 비교
# ----------------------------------------

st.header("5. 전처리 전후 결과 비교")

before_rows = len(before_df)
after_rows = len(processed_df)

before_missing = int(before_df.isna().sum().sum())
after_missing = int(processed_df.isna().sum().sum())

before_duplicates = int(before_df.duplicated().sum())
after_duplicates = int(processed_df.duplicated().sum())

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "행 개수",
    f"{after_rows:,}",
    delta=f"{after_rows - before_rows:+,}"
)

c2.metric(
    "결측 셀 개수",
    f"{after_missing:,}",
    delta=f"{after_missing - before_missing:+,}",
    delta_color="inverse"
)

c3.metric(
    "완전 중복 행",
    f"{after_duplicates:,}",
    delta=f"{after_duplicates - before_duplicates:+,}",
    delta_color="inverse"
)

c4.metric(
    "열 개수",
    f"{len(processed_df.columns):,}"
)

comparison = pd.DataFrame({
    "항목": [
        "행 개수",
        "열 개수",
        "결측 셀 개수",
        "완전 중복 행 개수"
    ],
    "전처리 전": [
        len(before_df),
        len(before_df.columns),
        before_missing,
        before_duplicates
    ],
    "전처리 후": [
        len(processed_df),
        len(processed_df.columns),
        after_missing,
        after_duplicates
    ]
})

st.subheader("전처리 전후 요약")

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

st.subheader("실제로 적용한 처리")

if processing_log:

    for item in processing_log:
        st.write(f"• {item}")

else:

    st.write(
        "선택한 설정에 따라 추가로 적용된 처리 내용이 없습니다."
    )


# ----------------------------------------
# 8. 주요 통계량 비교
# ----------------------------------------

st.subheader("숫자형 변수 통계 비교")

available_numeric = [
    col for col in numeric_cols
    if col in processed_df.columns
]

if available_numeric:

    default_idx = (
        available_numeric.index("Close")
        if "Close" in available_numeric
        else 0
    )

    stat_col = st.selectbox(
        "비교할 숫자형 열",
        options=available_numeric,
        index=default_idx
    )

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

    plot_data = pd.concat([
        pd.DataFrame({
            "구분": "전처리 전",
            "값": before_df[stat_col].dropna()
        }),
        pd.DataFrame({
            "구분": "전처리 후",
            "값": processed_df[stat_col].dropna()
        })
    ], ignore_index=True)

    fig_compare = px.box(
        plot_data,
        x="구분",
        y="값",
        color="구분",
        title=f"{stat_col} 전처리 전후 분포 비교"
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True
    )

    st.caption(
        "평균이나 분포가 달라졌다면 전처리 방법이 데이터에 "
        "영향을 주었을 가능성이 있습니다. 변화가 크다고 해서 "
        "무조건 잘못된 전처리라는 뜻은 아닙니다."
    )

else:

    st.info("비교할 숫자형 열이 없습니다.")


# ----------------------------------------
# 9. 전처리 결과 미리 보기 및 다운로드
# ----------------------------------------

st.header("6. 전처리 결과 확인 및 저장")

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
    "다운로드 파일은 현재 선택한 전처리 방법을 적용한 결과입니다. "
    "원본 GitHub CSV는 변경되지 않습니다."
)
