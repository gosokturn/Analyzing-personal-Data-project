import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from market_utils import load_raw_data, prepare_market_data, build_market_features

st.title("🧠 시장 유형과 관계 분석")
st.caption("비슷한 시장을 묶고, 시장 사이의 움직임과 과거 하락 사례를 데이터로 살펴봅니다.")

try:
    raw_df = load_raw_data()
    winsorize = st.session_state.get("winsorize_returns", False)
    clean_df, price_col, audit = prepare_market_data(raw_df, winsorize=winsorize)
except Exception as e:
    st.error("분석 데이터를 준비하지 못했습니다. 데이터 원본과 전처리 설정을 확인해 주세요.")
    with st.expander("오류 세부 정보"):
        st.exception(e)
    st.stop()

st.caption("현재 전처리 설정: " + ("시장별 수익률 극단값 완화 적용" if winsorize else "극단값 유지"))

min_obs = st.slider("시장별 최소 수익률 관측 일수", 30, 252, 60, 10,
                    help="이 기준보다 적은 수익률 기록을 가진 시장은 시장 유형 분석에서 제외됩니다.")
features = build_market_features(clean_df, price_col, min_observations=min_obs)

if len(features) < 3:
    st.error("군집화를 실행하려면 조건을 만족하는 시장이 최소 3개 필요합니다.")
    st.stop()

feature_labels = {
    "annualized_return": "연환산 평균 수익률",
    "annualized_volatility": "연환산 변동성",
    "downside_volatility": "하락 변동성",
    "max_drawdown": "최대 낙폭",
    "positive_day_ratio": "상승일 비율",
    "average_absolute_return": "평균 절대 수익률",
    "average_intraday_range": "평균 일중 변동 폭"
}

# ─────────────────────────────────────
# 시장 유형 분석
# ─────────────────────────────────────
st.divider()
st.header("1. 비슷한 움직임을 보이는 시장 찾기")
st.write("각 시장의 수익률과 위험 관련 특징을 비교해 비슷한 시장끼리 묶습니다. 이 그룹은 국가의 경제 수준이 아니라 선택한 금융 지표의 유사성을 나타냅니다.")

feature_columns = [col for col in feature_labels if col in features.columns and features[col].notna().any()]
if len(feature_columns) < 2:
    st.warning("군집화에 사용할 수 있는 특징 변수가 부족합니다.")
    st.stop()

default_features = feature_columns
with st.expander("군집화에 사용할 분석 기준 선택", expanded=False):
    selected_features = st.multiselect(
        "비교할 특징값 (2개 이상)",
        options=feature_columns,
        default=default_features,
        format_func=lambda x: feature_labels[x]
    )
    st.caption("선택한 특징값만 K-Means의 거리 계산에 사용됩니다. 변수 선택을 바꾸면 그룹 구성도 달라질 수 있습니다.")

if len(selected_features) < 2:
    st.warning("군집화를 위해 특징값을 두 개 이상 선택해 주세요.")
    st.stop()

model_data = features[["Index"] + selected_features].copy()
model_data = model_data.dropna(subset=selected_features, how="all").reset_index(drop=True)
if len(model_data) < 3:
    st.warning("현재 선택한 특징값으로 분석할 수 있는 시장이 3개 미만입니다.")
    st.stop()

# 기존 모델 로직 유지: 중앙값 대체 → RobustScaler → KMeans
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(model_data[selected_features])
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_imputed)

max_k = min(6, len(model_data) - 1)
k = st.slider("시장 그룹 수", 2, max_k, min(3, max_k), 1,
              help="몇 개의 그룹으로 나눌지 정합니다. 값에 따라 시장 분류가 달라집니다.")
kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
labels = kmeans.fit_predict(X_scaled)

cluster_results = model_data.copy()
cluster_results["Cluster"] = [f"그룹 {label + 1}" for label in labels]
if len(set(labels)) > 1 and len(set(labels)) < len(labels):
    sil = silhouette_score(X_scaled, labels)
else:
    sil = np.nan

c1, c2, c3 = st.columns(3)
c1.metric("분석한 시장", f"{len(cluster_results):,}개")
c2.metric("나뉜 그룹", f"{k}개")
c3.metric("그룹 구분 참고 점수", f"{sil:.3f}" if pd.notna(sil) else "계산 불가",
          help="실루엣 점수는 같은 그룹 안의 시장이 서로 가깝고 다른 그룹과 떨어져 있는 정도를 요약합니다. 미래 예측력이나 투자 수익을 뜻하지 않습니다.")
st.caption("실루엣 점수는 군집이 얼마나 잘 구분되는지 참고하는 지표입니다. 높다고 해서 그룹이 경제적으로 더 의미 있거나 미래를 잘 예측한다는 뜻은 아닙니다.")

full_cluster_results = features.merge(cluster_results[["Index", "Cluster"]], on="Index", how="inner")

st.subheader("그룹별 특징")
for cluster_name in sorted(full_cluster_results["Cluster"].unique(), key=lambda x: int(x.split()[-1])):
    group = full_cluster_results[full_cluster_results["Cluster"] == cluster_name]
    avg_return = group["annualized_return"].mean()
    avg_vol = group["annualized_volatility"].mean()
    avg_dd = group["max_drawdown"].mean()
    st.markdown(f"#### {cluster_name} · {len(group)}개 시장")
    m1, m2, m3 = st.columns(3)
    m1.metric("평균 연환산 수익률", f"{avg_return:.1%}" if pd.notna(avg_return) else "계산 불가")
    m2.metric("평균 연환산 변동성", f"{avg_vol:.1%}" if pd.notna(avg_vol) else "계산 불가")
    m3.metric("평균 최대 낙폭", f"{avg_dd:.1%}" if pd.notna(avg_dd) else "계산 불가")
    descriptions = []
    if pd.notna(avg_vol):
        median_vol = full_cluster_results["annualized_volatility"].median()
        descriptions.append("전체 시장의 중앙값보다 변동성이 큰 편입니다." if avg_vol > median_vol else "전체 시장의 중앙값보다 변동성이 작은 편입니다.")
    if pd.notna(avg_return):
        median_return = full_cluster_results["annualized_return"].median()
        descriptions.append("평균 수익률도 전체 시장 중앙값보다 높게 나타났습니다." if avg_return > median_return else "평균 수익률은 전체 시장 중앙값보다 낮거나 비슷하게 나타났습니다.")
    if descriptions:
        st.write(" ".join(descriptions))
    st.caption("그룹의 평균값은 이 그룹에 포함된 시장별 특징값을 단순 평균한 값입니다.")
    with st.expander(f"{cluster_name}에 포함된 시장 보기"):
        st.write(", ".join(group["Index"].astype(str).tolist()))
        st.dataframe(group[["Index"] + [x for x in feature_labels if x in group.columns]].round(4),
                     use_container_width=True, hide_index=True)

if {"annualized_volatility", "annualized_return"}.issubset(cluster_results.columns):
    fig_scatter = px.scatter(
        cluster_results, x="annualized_volatility", y="annualized_return",
        color="Cluster", hover_name="Index", hover_data=selected_features,
        title="시장별 평균 수익률과 변동성",
        labels={"annualized_volatility": "연환산 변동성", "annualized_return": "연환산 평균 수익률", "Cluster": "시장 그룹"}
    )
    fig_scatter.update_layout(margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption("오른쪽으로 갈수록 변동성이 크고, 위쪽일수록 분석 기간의 연환산 평균 수익률이 높습니다.")

summary_cols = [x for x in selected_features if x in full_cluster_results.columns]
cluster_summary = full_cluster_results.groupby("Cluster")[summary_cols].mean().rename(columns=feature_labels)
if not cluster_summary.empty:
    fig_summary = px.bar(cluster_summary.reset_index().melt(id_vars="Cluster", var_name="특징값", value_name="그룹 평균"),
                         x="특징값", y="그룹 평균", color="Cluster", barmode="group",
                         title="그룹별 선택 특징값 비교")
    st.plotly_chart(fig_summary, use_container_width=True)

st.download_button(
    "📥 시장 그룹 분류 결과 CSV 다운로드",
    data=cluster_results.to_csv(index=False).encode("utf-8-sig"),
    file_name="market_clustering_results.csv",
    mime="text/csv"
)

# ─────────────────────────────────────
# 시장 관계 분석
# ─────────────────────────────────────
st.divider()
st.header("2. 세계 주식시장은 서로 얼마나 비슷하게 움직일까?")
st.write("두 시장의 일별 수익률이 같은 방향 또는 반대 방향으로 움직이는 경향이 있는지 상관계수로 살펴봅니다.")

with st.expander("상관계수 읽는 법"):
    st.markdown("""
    - **+1에 가까움:** 두 시장의 수익률이 같은 방향으로 움직이는 경향이 강합니다.
    - **0에 가까움:** 두 시장 사이의 선형적인 관계가 약합니다.
    - **-1에 가까움:** 두 시장의 수익률이 반대 방향으로 움직이는 경향이 강합니다.

    상관관계는 비례관계나 인과관계와 다릅니다. 상관계수가 높더라도 한 시장이 다른 시장을 움직였다고 단정할 수 없습니다.
    """)

returns_pivot = clean_df.pivot_table(index="Date", columns="Index", values="daily_return_used", aggfunc="mean")
valid_markets = returns_pivot.count()
valid_markets = valid_markets[valid_markets >= 60].index.tolist()
returns_pivot = returns_pivot[valid_markets]

if len(valid_markets) >= 2:
    corr_matrix = returns_pivot.corr(min_periods=60)
    corr_matrix = corr_matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if len(corr_matrix) >= 2:
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
            color_continuous_scale="RdBu_r",
            title="시장별 일별 수익률 상관관계",
            labels={"color": "상관계수"}
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("색과 숫자는 두 시장의 일별 수익률 관계를 나타냅니다. 흰색에 가까운 값은 0 부근, 양수와 음수는 각각 같은 방향과 반대 방향의 경향을 뜻합니다.")

        selected_market = st.selectbox("자세히 살펴볼 기준 시장", corr_matrix.columns.tolist())
        selected_corr = corr_matrix[selected_market].drop(selected_market).sort_values(ascending=False)
        corr_table = selected_corr.rename("상관계수").reset_index().rename(columns={"Index": "비교 시장"})
        left, right = st.columns(2)
        with left:
            st.markdown("**같은 방향으로 움직이는 경향이 큰 시장**")
            st.dataframe(corr_table.head(5).round(3), use_container_width=True, hide_index=True)
        with right:
            st.markdown("**반대 방향으로 움직이는 경향이 큰 시장**")
            st.dataframe(corr_table.tail(5).sort_values("상관계수").round(3), use_container_width=True, hide_index=True)
        fig_one = px.bar(corr_table.sort_values("상관계수"), x="상관계수", y="비교 시장", orientation="h",
                         title=f"{selected_market}과 다른 시장의 상관계수",
                         color="상관계수", color_continuous_scale="RdBu_r", range_color=[-1, 1])
        st.plotly_chart(fig_one, use_container_width=True)
        st.caption("상관계수는 같은 날짜에 관측된 수익률을 기준으로 계산합니다. 시장마다 거래일과 휴장일이 달라 실제 비교에 사용된 날짜 수가 다를 수 있습니다.")
    else:
        st.warning("상관관계를 계산할 수 있는 시장이 충분하지 않습니다.")
else:
    st.warning("60개 이상의 수익률 기록을 가진 시장이 2개 이상 필요합니다.")

# ─────────────────────────────────────
# 하락 위험 분류
# ─────────────────────────────────────
st.divider()
st.header("3. 과거 데이터에서 큰 하락 사례 살펴보기")
st.write("로지스틱 회귀 모델이 과거 수익률 흐름을 바탕으로 다음 거래일의 큰 하락 사례를 얼마나 잘 분류했는지 확인합니다.")
st.warning("이 모델은 과거 자료를 이용한 교육용 분류 실험입니다. 미래의 하락을 확실히 예측하거나 투자 손실을 방지하지 않습니다.")

crash_threshold = st.slider("큰 하락으로 분류할 다음 거래일 수익률 기준", -10.0, -1.0, -2.0, 0.5) / 100
st.caption(f"다음 거래일 수익률이 {crash_threshold:.1%} 이하이면 '하락 사례'로 분류합니다.")

classification_df = clean_df[["Index", "Date", "daily_return_used"]].copy()
classification_df = classification_df.sort_values(["Index", "Date"])
grouped_returns = classification_df.groupby("Index")["daily_return_used"]

# 기존 모델 입력 변수 계산 방식 유지
classification_df["return_today"] = classification_df["daily_return_used"]
classification_df["return_5d"] = grouped_returns.transform(lambda s: s.rolling(5, min_periods=5).sum())
classification_df["volatility_20d"] = grouped_returns.transform(lambda s: s.rolling(20, min_periods=20).std())
classification_df["mean_return_20d"] = grouped_returns.transform(lambda s: s.rolling(20, min_periods=20).mean())
classification_df["next_return"] = grouped_returns.shift(-1)
classification_df["target"] = (classification_df["next_return"] <= crash_threshold).astype(int)

model_features = ["return_today", "return_5d", "volatility_20d", "mean_return_20d"]
classification_df = classification_df.dropna(subset=model_features + ["next_return"]).copy()

if len(classification_df) < 100:
    st.warning("분류 모델을 학습하기 위한 데이터가 충분하지 않습니다.")
else:
    unique_dates = sorted(classification_df["Date"].unique())
    split_position = int(len(unique_dates) * 0.8)
    if split_position <= 0 or split_position >= len(unique_dates):
        st.warning("학습용과 테스트용 날짜를 나누기 어렵습니다.")
    else:
        split_date = unique_dates[split_position]
        train_df = classification_df[classification_df["Date"] < split_date]
        test_df = classification_df[classification_df["Date"] >= split_date]
        X_train, y_train = train_df[model_features], train_df["target"]
        X_test, y_test = test_df[model_features], test_df["target"]

        if y_train.nunique() < 2:
            st.warning("학습 데이터에 하락과 비하락 사례가 모두 있어야 로지스틱 회귀를 학습할 수 있습니다.")
        elif len(test_df) == 0:
            st.warning("테스트 데이터가 없습니다.")
        else:
            classifier = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
            ])
            classifier.fit(X_train, y_train)
            predictions = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)

            st.subheader("모델 성능을 쉽게 읽어보기")
            a, b, c = st.columns(3)
            a.metric("정확도", f"{accuracy:.1%}", help="전체 테스트 사례 중 실제 결과를 맞힌 비율입니다.")
            b.metric("정밀도", f"{precision:.1%}", help="하락이라고 예측한 사례 중 실제로 하락한 비율입니다.")
            c.metric("재현율", f"{recall:.1%}", help="실제로 하락한 사례 중 모델이 찾아낸 비율입니다.")
            with st.expander("세 지표의 차이 알아보기"):
                st.markdown("""
                - **정확도:** 전체 예측 중 맞힌 비율
                - **정밀도:** 하락이라고 예측한 것 중 실제 하락의 비율
                - **재현율:** 실제 하락 사례 중 모델이 찾아낸 비율

                하락 사례를 놓치지 않는 것이 중요할 때는 재현율도 살펴봐야 합니다. 반대로 하락이라고 경고한 사례가 얼마나 정확했는지는 정밀도로 확인합니다.
                """)

            split_dt = pd.Timestamp(split_date).date()
            st.write(f"학습 기간: **{pd.Timestamp(train_df['Date'].min()).date()} ~ {pd.Timestamp(train_df['Date'].max()).date()}** · {len(train_df):,}건")
            st.write(f"테스트 기간: **{split_dt} ~ {pd.Timestamp(test_df['Date'].max()).date()}** · {len(test_df):,}건")
            st.caption("날짜를 기준으로 과거 데이터를 학습용, 이후 데이터를 테스트용으로 나누었습니다. 무작위로 섞지 않아 시간 순서를 유지합니다.")

            matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
            matrix_df = pd.DataFrame(matrix, index=["실제 비하락", "실제 하락"], columns=["비하락 예측", "하락 예측"])
            fig_matrix = px.imshow(matrix_df, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                                   title="실제 결과와 모델 예측 비교", labels={"x": "모델 예측", "y": "실제 결과", "color": "사례 수"})
            st.plotly_chart(fig_matrix, use_container_width=True)
            st.caption("대각선 칸은 맞힌 사례이고, 대각선 바깥 칸은 틀린 사례입니다.")

            false_negative = int(matrix[1, 0])
            actual_crashes = int(matrix[1, :].sum())
            actual_noncrashes = int(matrix[0, :].sum())
            q1, q2, q3 = st.columns(3)
            q1.metric("실제 하락 사례", f"{actual_crashes:,}건")
            q2.metric("찾아낸 하락 사례", f"{int(matrix[1,1]):,}건")
            q3.metric("놓친 하락 사례", f"{false_negative:,}건", help="실제로 하락했지만 모델이 비하락으로 예측한 사례입니다.")
            if actual_crashes:
                st.info(f"실제 하락 사례 {actual_crashes:,}건 중 {false_negative:,}건을 놓쳤습니다. 재현율은 이 비율을 반영합니다.")
            else:
                st.info("테스트 기간에 실제 하락 사례가 없어 하락 사례를 얼마나 찾아냈는지 평가할 수 없습니다.")

            result_df = test_df[["Index", "Date", "next_return", "target"]].copy()
            result_df["predicted"] = predictions
            result_df["실제 결과"] = result_df["target"].map({0: "비하락", 1: "하락"})
            result_df["모델 예측"] = result_df["predicted"].map({0: "비하락", 1: "하락"})
            with st.expander("테스트 사례별 결과 표 보기"):
                st.dataframe(result_df.head(300), use_container_width=True, hide_index=True)
            st.download_button(
                "📥 하락 위험 분류 결과 CSV 다운로드",
                data=result_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="market_downside_classification.csv",
                mime="text/csv"
            )

# ─────────────────────────────────────
# 국가별 경제 자료 업로드
# ─────────────────────────────────────
st.divider()
st.header("4. 금융시장과 국가 경제 지표 함께 살펴보기")
st.write("주가지수만으로는 GDP, 물가상승률, 정책금리, 실업률을 알 수 없습니다. 별도의 경제 자료를 추가하면 금융시장 특징과 경제 지표 사이의 관계를 탐색할 수 있습니다.")

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
    "📥 국가별 경제 자료 입력 양식 다운로드",
    data=metadata_template.to_csv(index=False).encode("utf-8-sig"),
    file_name="market_economic_metadata_template.csv",
    mime="text/csv"
)
st.caption("지수 이름이 곧 국가 이름이라는 보장은 없습니다. 한 국가에 여러 지수가 있거나 여러 국가를 포함하는 지수도 있으므로 Country 열은 출처를 확인해 입력해야 합니다.")

uploaded_metadata = st.file_uploader("작성한 국가별 경제 자료 CSV 업로드", type=["csv"])
if uploaded_metadata is not None:
    try:
        metadata = pd.read_csv(uploaded_metadata, encoding="utf-8-sig")
        if "Index" not in metadata.columns:
            st.error("업로드한 파일에 Index 열이 필요합니다.")
        else:
            combined = full_cluster_results.merge(metadata, on="Index", how="left")
            st.subheader("시장 분석 결과와 경제 자료 결합")
            st.dataframe(combined, use_container_width=True, hide_index=True)

            if "Country" in combined.columns:
                country_counts = combined.groupby(["Cluster", "Country"], dropna=False).size().reset_index(name="시장 수")
                fig_country = px.bar(country_counts, x="Cluster", y="시장 수", color="Country",
                                     title="그룹별 국가 정보 분포")
                st.plotly_chart(fig_country, use_container_width=True)

            macro_columns = ["GDP_per_capita", "Inflation", "Policy_rate", "Unemployment"]
            available_macro = [col for col in macro_columns if col in combined.columns]
            for col in available_macro:
                combined[col] = pd.to_numeric(combined[col], errors="coerce")
            correlation_columns = [
                col for col in ["annualized_return", "annualized_volatility", "max_drawdown"] + available_macro
                if col in combined.columns and pd.api.types.is_numeric_dtype(combined[col]) and combined[col].notna().sum() >= 3
            ]
            if len(correlation_columns) >= 2:
                macro_corr = combined[correlation_columns].corr()
                fig_macro = px.imshow(macro_corr, text_auto=".2f", zmin=-1, zmax=1,
                                      color_continuous_scale="RdBu_r",
                                      title="시장 특징과 업로드한 경제 지표의 상관관계")
                st.plotly_chart(fig_macro, use_container_width=True)
                st.caption("시장별 관측값 사이의 상관관계입니다. 경제 지표의 기준 연도와 주가지수 분석 기간이 다르면 직접 비교에 한계가 있습니다.")
            else:
                st.info("상관관계를 계산하려면 숫자로 읽을 수 있는 경제 지표가 두 개 이상 필요합니다.")
            st.download_button(
                "📥 경제 자료 결합 결과 CSV 다운로드",
                data=combined.to_csv(index=False).encode("utf-8-sig"),
                file_name="market_economic_cluster_results.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error("경제 자료 CSV를 읽는 중 문제가 발생했습니다. 열 이름과 숫자 형식을 확인해 주세요.")
        with st.expander("오류 세부 정보"):
            st.exception(e)

st.divider()
with st.expander("분석 결과를 해석할 때 기억할 점"):
    st.markdown("""
    1. 같은 그룹에 속한다고 국가의 경제 구조까지 같다는 뜻은 아닙니다.
    2. 상관관계가 높다고 한 시장이 다른 시장의 원인이라는 뜻은 아닙니다.
    3. 군집 결과는 입력 변수, 분석 기간, 그룹 수에 따라 달라질 수 있습니다.
    4. 하락 분류 모델의 과거 성능이 미래에도 유지된다고 보장할 수 없습니다.
    5. 거래 비용, 환율, 세금, 거래 가능 시간 등은 이 분석에 포함되어 있지 않습니다.
    """)
