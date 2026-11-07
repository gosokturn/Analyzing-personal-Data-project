
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from market_utils import (
    load_raw_data,
    prepare_market_data,
    build_market_features
)


st.title("🧠 세계 금융시장 모델 학습 결과")
st.caption(
    "시장 군집화, 상관관계, 하락 위험 분류를 종합적으로 분석합니다."
)


# ==========================================
# 1. 데이터 준비
# ==========================================

try:
    raw_df = load_raw_data()

except Exception as e:
    st.error("원본 데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


winsorize = st.session_state.get(
    "winsorize_returns",
    False
)

try:
    clean_df, price_col, audit = prepare_market_data(
        raw_df,
        winsorize=winsorize
    )

except Exception as e:
    st.error("데이터 전처리에 실패했습니다.")
    st.exception(e)
    st.stop()


st.info(
    "현재 전처리 설정: "
    + (
        "시장별 수익률 극단값 완화 적용"
        if winsorize
        else "수익률 극단값 유지"
    )
)


min_obs = st.slider(
    "시장별 최소 관측 일수",
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

if len(features) < 3:
    st.error(
        "군집화를 실행하려면 최소 3개 이상의 시장이 필요합니다."
    )
    st.stop()


# ==========================================
# 2. K-Means 군집화
# ==========================================

st.divider()
st.header("1. K-Means 군집화")

st.markdown("""
**목표:** 비슷한 수익률과 위험 특성을 가진 금융시장을 같은 그룹으로 묶는다.

K-Means는 각 시장의 특징을 좌표로 표현하고,
서로 가까운 시장끼리 K개의 군집으로 나누는 알고리즘이다.

군집은 국가의 선진국·개발도상국 여부를 자동으로 판정하는 것이 아니라,
입력한 금융시장 지표가 비슷한 시장끼리 묶은 결과이다.
""")

feature_labels = {
    "annualized_return": "연환산 평균 수익률",
    "annualized_volatility": "연환산 변동성",
    "downside_volatility": "하락 변동성",
    "max_drawdown": "최대 낙폭",
    "positive_day_ratio": "상승일 비율",
    "average_absolute_return": "평균 절대 수익률",
    "average_intraday_range": "평균 일중 변동 폭"
}

feature_columns = [
    col for col in feature_labels
    if col in features.columns
    and features[col].notna().any()
]

if len(feature_columns) < 2:
    st.error("군집화에 사용할 수 있는 특징 변수가 부족합니다.")
    st.stop()


st.subheader("군집화에 사용할 변수")

selected_features = st.multiselect(
    "분석 변수 선택",
    options=feature_columns,
    default=feature_columns
)

if len(selected_features) < 2:
    st.warning("두 개 이상의 변수를 선택해 주세요.")
    st.stop()


st.write(
    "선택한 변수: "
    + ", ".join(
        feature_labels[col] for col in selected_features
    )
)

st.markdown("""
**왜 표준화할까?**

수익률은 소수 단위로 나타나지만, 변동성이나 다른 지표는 값의 범위가
다를 수 있다. K-Means는 거리로 유사도를 계산하므로,
숫자 크기가 큰 변수가 결과를 지배할 수 있다.

RobustScaler는 중앙값과 사분위 범위를 이용해 변수를 변환한다.
극단적인 시장 데이터가 존재할 수 있는 금융 분석에서 고려할 만한 방법이다.
""")


# 모든 값이 비어 있는 행은 제거
model_data = features[
    ["Index"] + selected_features
].copy()

model_data = model_data.dropna(
    subset=selected_features,
    how="all"
).reset_index(drop=True)

if len(model_data) < 3:
    st.error("군집화에 필요한 시장 수가 부족합니다.")
    st.stop()


# 결측 특징값을 중앙값으로 대체
imputer = SimpleImputer(strategy="median")

X_imputed = imputer.fit_transform(
    model_data[selected_features]
)

# RobustScaler로 표준화
scaler = RobustScaler()

X_scaled = scaler.fit_transform(X_imputed)


max_k = min(6, len(model_data) - 1)

k = st.slider(
    "군집 수 K 선택",
    min_value=2,
    max_value=max_k,
    value=min(3, max_k)
)

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=20
)

labels = kmeans.fit_predict(X_scaled)

cluster_results = model_data.copy()
cluster_results["Cluster"] = [
    f"그룹 {label + 1}" for label in labels
]


# 실루엣 점수
if len(set(labels)) > 1 and len(set(labels)) < len(labels):
    sil = silhouette_score(
        X_scaled,
        labels
    )
else:
    sil = np.nan


c1, c2, c3 = st.columns(3)

c1.metric("분석 시장 수", f"{len(cluster_results):,}")
c2.metric("군집 수", str(k))
c3.metric(
    "실루엣 점수",
    f"{sil:.3f}" if pd.notna(sil) else "계산 불가"
)

st.caption(
    "실루엣 점수는 군집이 얼마나 잘 구분되는지 참고하는 지표이다. "
    "값이 높다고 미래 예측력이 높다는 뜻은 아니다."
)


# 군집별 산점도
if (
    "annualized_volatility" in cluster_results.columns
    and "annualized_return" in cluster_results.columns
):

    fig_cluster = px.scatter(
        cluster_results,
        x="annualized_volatility",
        y="annualized_return",
        color="Cluster",
        hover_name="Index",
        title="시장별 수익률과 변동성 군집",
        labels={
            "annualized_volatility": "연환산 변동성",
            "annualized_return": "연환산 평균 수익률"
        }
    )

    st.plotly_chart(
        fig_cluster,
        use_container_width=True
    )


# 군집별 평균 특징
st.subheader("군집별 공통점")

cluster_summary = (
    cluster_results
    .groupby("Cluster")[selected_features]
    .mean()
)

cluster_summary = cluster_summary.rename(
    columns=feature_labels
)

st.dataframe(
    cluster_summary.round(4),
    use_container_width=True
)


# 군집별 시장 목록
st.subheader("각 군집에 포함된 시장")

for cluster_name, group in cluster_results.groupby("Cluster"):

    with st.expander(
        f"{cluster_name} ({len(group)}개 시장)"
    ):

        st.write(
            ", ".join(group["Index"].astype(str).tolist())
        )

        st.dataframe(
            group,
            use_container_width=True,
            hide_index=True
        )


# 다운로드
st.download_button(
    "📥 군집화 결과 CSV 다운로드",
    data=cluster_results.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="market_clustering_results.csv",
    mime="text/csv"
)


# ==========================================
# 3. 상관관계 분석
# ==========================================

st.divider()
st.header("2. 국가별·시장별 상관관계 분석")

st.markdown("""
**목표:** 여러 시장의 수익률이 서로 어떤 방향으로 움직이는지 분석한다.

- +1에 가까움: 두 시장의 수익률이 같은 방향으로 움직이는 경향이 강하다.
- 0에 가까움: 선형적인 관계가 약하다.
- -1에 가까움: 두 시장의 수익률이 반대 방향으로 움직이는 경향이 강하다.

상관관계는 인과관계를 의미하지 않는다.
""")

returns_pivot = clean_df.pivot_table(
    index="Date",
    columns="Index",
    values="daily_return_used",
    aggfunc="mean"
)

# 관측값이 충분한 시장만 사용
valid_markets = returns_pivot.count()
valid_markets = valid_markets[
    valid_markets >= 60
].index.tolist()

returns_pivot = returns_pivot[
    valid_markets
]

if len(valid_markets) >= 2:

    corr_matrix = returns_pivot.corr(
        min_periods=60
    )

    corr_matrix = corr_matrix.dropna(
        axis=0,
        how="all"
    ).dropna(
        axis=1,
        how="all"
    )

    if len(corr_matrix) >= 2:

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            title="시장별 일별 수익률 상관관계"
        )

        st.plotly_chart(
            fig_corr,
            use_container_width=True
        )

        st.subheader("특정 시장과 다른 시장의 관계")

        selected_market = st.selectbox(
            "기준 시장 선택",
            options=corr_matrix.columns.tolist()
        )

        selected_corr = (
            corr_matrix[selected_market]
            .drop(selected_market)
            .sort_values(ascending=False)
            .rename("상관계수")
            .reset_index()
            .rename(columns={"Index": "비교 시장"})
        )

        st.dataframe(
            selected_corr,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "상관계수는 같은 기간의 일별 수익률을 기준으로 계산한다. "
            "서로 다른 거래일과 휴장일은 결측값으로 남는다."
        )

    else:
        st.warning("상관관계를 계산할 수 있는 시장이 부족합니다.")

else:
    st.warning(
        "60개 이상의 수익률 관측값을 가진 시장이 2개 이상 필요합니다."
    )


# ==========================================
# 4. 하락 위험 분류
# ==========================================

st.divider()
st.header("3. 로지스틱 회귀: 다음 거래일 하락 위험 분류")

st.markdown("""
**목표:** 현재까지의 수익률과 최근 변동성을 이용해
다음 거래일에 일정 수준 이상의 하락이 발생할지 분류한다.

분류 기준은 사용자가 선택한다. 예를 들어 -2%를 기준으로 선택하면
다음 거래일 수익률이 -2% 이하인 경우를 하락 위험 사례로 표시한다.

이 모델은 과거 자료의 패턴을 학습하는 교육용 실험이며,
실제 시장에서 하락을 확실히 예측하지는 못한다.
""")

crash_threshold = st.slider(
    "급격한 하락 기준",
    min_value=-10.0,
    max_value=-1.0,
    value=-2.0,
    step=0.5
) / 100


# 시장별 시간순 데이터 생성
classification_df = clean_df[
    ["Index", "Date", "daily_return_used"]
].copy()

classification_df = classification_df.sort_values(
    ["Index", "Date"]
)

grouped_returns = classification_df.groupby(
    "Index"
)["daily_return_used"]

# 현재 날짜까지의 정보만 입력 변수로 사용
classification_df["return_today"] = (
    classification_df["daily_return_used"]
)

classification_df["return_5d"] = (
    grouped_returns
    .transform(
        lambda s: s.rolling(5, min_periods=5).sum()
    )
)

classification_df["volatility_20d"] = (
    grouped_returns
    .transform(
        lambda s: s.rolling(20, min_periods=20).std()
    )
)

classification_df["mean_return_20d"] = (
    grouped_returns
    .transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
)

# 다음 거래일 수익률
classification_df["next_return"] = (
    grouped_returns.shift(-1)
)

classification_df["target"] = (
    classification_df["next_return"]
    <= crash_threshold
).astype(int)

model_features = [
    "return_today",
    "return_5d",
    "volatility_20d",
    "mean_return_20d"
]

classification_df = classification_df.dropna(
    subset=model_features + ["next_return"]
).copy()

if len(classification_df) < 100:
    st.warning(
        "분류 모델을 학습하기 위한 데이터가 충분하지 않습니다."
    )

else:

    # 날짜를 기준으로 학습/테스트 분리
    unique_dates = sorted(
        classification_df["Date"].unique()
    )

    split_position = int(len(unique_dates) * 0.8)

    if split_position <= 0 or split_position >= len(unique_dates):
        st.warning("학습용과 테스트용 날짜를 나누기 어렵습니다.")

    else:

        split_date = unique_dates[split_position]

        train_df = classification_df[
            classification_df["Date"] < split_date
        ]

        test_df = classification_df[
            classification_df["Date"] >= split_date
        ]

        X_train = train_df[model_features]
        y_train = train_df["target"]

        X_test = test_df[model_features]
        y_test = test_df["target"]

        if y_train.nunique() < 2:
            st.warning(
                "학습 데이터에 하락과 비하락 사례가 모두 있어야 "
                "로지스틱 회귀를 학습할 수 있습니다."
            )

        elif len(test_df) == 0:
            st.warning("테스트 데이터가 없습니다.")

        else:

            classifier = Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42
                    )
                )
            ])

            classifier.fit(
                X_train,
                y_train
            )

            predictions = classifier.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                zero_division=0
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "정확도",
                f"{accuracy:.3f}"
            )

            c2.metric(
                "정밀도",
                f"{precision:.3f}"
            )

            c3.metric(
                "재현율",
                f"{recall:.3f}"
            )

            st.write(
                f"학습 데이터: {len(train_df):,}개"
            )

            st.write(
                f"테스트 데이터: {len(test_df):,}개"
            )

            st.write(
                f"학습/테스트 분리 날짜: {pd.Timestamp(split_date).date()}"
            )

            st.caption(
                "정확도는 전체 예측 중 맞힌 비율, 정밀도는 하락이라고 "
                "예측한 사례 중 실제 하락의 비율, 재현율은 실제 하락 "
                "사례 중 찾아낸 비율이다."
            )

            matrix = confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1]
            )

            matrix_df = pd.DataFrame(
                matrix,
                index=["실제 비하락", "실제 하락"],
                columns=["비하락 예측", "하락 예측"]
            )

            st.subheader("분류 결과")

            st.dataframe(
                matrix_df,
                use_container_width=True
            )

            result_df = test_df[
                ["Index", "Date", "next_return", "target"]
            ].copy()

            result_df["predicted"] = predictions

            result_df["실제 결과"] = result_df["target"].map({
                0: "비하락",
                1: "하락"
            })

            result_df["모델 예측"] = result_df["predicted"].map({
                0: "비하락",
                1: "하락"
            })

            st.dataframe(
                result_df.head(100),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 하락 위험 분류 결과 다운로드",
                data=result_df.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="market_downside_classification.csv",
                mime="text/csv"
            )


# ==========================================
# 5. 국가별 거시경제 자료 추가 분석
# ==========================================

st.divider()
st.header("4. 국가별 경제 특성과 군집 비교")

st.markdown("""
현재 주가지수 데이터만으로는 GDP, 물가상승률, 정책금리,
실업률 등의 국가 경제 특성을 알 수 없다.

이러한 경제 지표를 별도의 CSV로 추가하면
금융시장 군집과 국가 경제 특성의 관계를 비교할 수 있다.
""")

metadata_template = pd.DataFrame({
    "Index": features["Index"],
    "Country": "",
    "Region": "",
    "GDP_per_capita": np.nan,
    "Inflation": np.nan,
    "Policy_rate": np.nan,
    "Unemployment": np.nan
})

st.download_button(
    "📥 국가별 경제 자료 입력용 양식 다운로드",
    data=metadata_template.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="market_economic_metadata_template.csv",
    mime="text/csv"
)

st.caption(
    "각 지수에 해당하는 국가와 경제 지표를 직접 확인해 입력해야 한다. "
    "한 국가에 여러 지수가 있거나 다국적 지수인 경우에는 "
    "국가 대응 관계를 신중히 정해야 한다."
)

uploaded_metadata = st.file_uploader(
    "국가별 경제 자료 CSV 업로드",
    type=["csv"]
)

if uploaded_metadata is not None:

    try:
        metadata = pd.read_csv(
            uploaded_metadata,
            encoding="utf-8-sig"
        )

        if "Index" not in metadata.columns:
            st.error("경제 자료에 Index 열이 필요합니다.")

        else:

            combined = cluster_results.merge(
                metadata,
                on="Index",
                how="left"
            )

            st.subheader("군집과 경제 자료 결합 결과")

            st.dataframe(
                combined,
                use_container_width=True,
                hide_index=True
            )

            if "Country" in combined.columns:

                country_counts = (
                    combined.groupby(
                        ["Cluster", "Country"],
                        dropna=False
                    )
                    .size()
                    .reset_index(name="시장 수")
                )

                st.subheader("군집별 국가 분포")

                fig_country = px.bar(
                    country_counts,
                    x="Cluster",
                    y="시장 수",
                    color="Country",
                    title="군집별 국가 분포"
                )

                st.plotly_chart(
                    fig_country,
                    use_container_width=True
                )

            macro_columns = [
                "GDP_per_capita",
                "Inflation",
                "Policy_rate",
                "Unemployment"
            ]

            available_macro = [
                col for col in macro_columns
                if col in combined.columns
            ]

            for col in available_macro:
                combined[col] = pd.to_numeric(
                    combined[col],
                    errors="coerce"
                )

            correlation_columns = [
                col for col in [
                    "annualized_return",
                    "annualized_volatility",
                    "max_drawdown"
                ] + available_macro
                if col in combined.columns
                and pd.api.types.is_numeric_dtype(combined[col])
                and combined[col].notna().sum() >= 3
            ]

            if len(correlation_columns) >= 2:

                macro_corr = combined[
                    correlation_columns
                ].corr()

                fig_macro = px.imshow(
                    macro_corr,
                    text_auto=".2f",
                    zmin=-1,
                    zmax=1,
                    color_continuous_scale="RdBu_r",
                    title="시장 특성과 경제 지표의 상관관계"
                )

                st.plotly_chart(
                    fig_macro,
                    use_container_width=True
                )

                st.caption(
                    "이 상관관계는 관측된 시장별 자료 사이의 관계를 나타낸다. "
                    "경제 지표의 측정 연도와 주가지수 분석 기간이 다르면 "
                    "해석에 주의해야 한다."
                )

            st.download_button(
                "📥 경제 자료 결합 결과 다운로드",
                data=combined.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="market_economic_cluster_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error("경제 자료를 읽는 중 오류가 발생했습니다.")
        st.exception(e)


# ==========================================
# 6. 최종 해석
# ==========================================

st.divider()
st.header("5. 모델 결과 해석 시 주의사항")

st.markdown("""
1. 같은 군집에 속한다고 해서 국가의 경제 구조까지 같다는 의미는 아니다.
2. 상관관계가 높다고 한 시장이 다른 시장의 원인이라는 뜻은 아니다.
3. 군집화는 입력 변수와 분석 기간을 바꾸면 결과도 달라질 수 있다.
4. 하락 분류 모델의 성능은 미래 시장에서 그대로 유지되지 않을 수 있다.
5. 거래 비용, 환율, 세금, 거래 가능 시간 등의 요소는 이 모델에 포함되지 않았다.

따라서 모델 결과는 금융시장의 특징을 이해하고 위험을 연구하는 자료로 활용해야 한다.
""")
