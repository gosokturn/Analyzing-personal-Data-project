import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from market_utils import load_raw_data, prepare_market_data, build_market_features

st.set_page_config(
    page_title="세계 금융시장 분석",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1250px;}
[data-testid="stMetric"] {background:#fff; border:1px solid #e8edf3;
    padding:14px 16px; border-radius:13px;}
div[data-testid="stExpander"] {border:1px solid #e8edf3; border-radius:12px;}
h1,h2,h3 {color:#172b4d;}
.info-card {background:#f2f6fb; padding:18px 20px; border-radius:14px;
    border:1px solid #e3ebf5; margin-bottom:12px;}
</style>
""", unsafe_allow_html=True)

st.title("🌍 세계 금융시장 분석")
st.caption("분석 질문부터 데이터, 기준선, 모델 평가까지 한 흐름으로 확인합니다.")

@st.cache_data(show_spinner="세계 금융시장 데이터를 불러오는 중...")
def get_data():
    raw = load_raw_data()
    clean, price_col, audit = prepare_market_data(raw, winsorize=False)
    features = build_market_features(clean, price_col, min_observations=60)
    return raw, clean, price_col, audit, features

try:
    raw_df, clean_df, price_col, audit, market_features = get_data()
except Exception as e:
    st.error("데이터를 불러오거나 분석용 데이터로 변환하지 못했습니다.")
    with st.expander("오류 내용"):
        st.exception(e)
    st.stop()

# 분류 실험 데이터: 다음 거래일의 큰 하락 여부를 예측
def make_classification_data(data, threshold):
    df = data[["Index", "Date", "daily_return_used"]].copy()
    df = df.sort_values(["Index", "Date"]).reset_index(drop=True)
    grouped = df.groupby("Index")["daily_return_used"]
    df["return_today"] = df["daily_return_used"]
    df["return_5d"] = grouped.transform(lambda s: s.rolling(5, min_periods=5).sum())
    df["volatility_20d"] = grouped.transform(lambda s: s.rolling(20, min_periods=20).std())
    df["mean_return_20d"] = grouped.transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["next_return"] = grouped.shift(-1)
    df["target"] = (df["next_return"] <= threshold).astype(int)
    features = ["return_today", "return_5d", "volatility_20d", "mean_return_20d"]
    return df.dropna(subset=features + ["next_return"]).copy(), features

tab_question, tab_data, tab_baseline, tab_eval = st.tabs([
    "① 질문", "② 데이터 분석", "③ 기준선", "④ 모델 평가"
])

with tab_question:
    st.header("① 어떤 질문에 답하려는가?")
    st.markdown("""
    <div class="info-card">
    <b>핵심 질문</b><br>
    세계 금융시장의 과거 움직임을 바탕으로 시장 유형을 구분하고,
    특정 시장에서 다음 거래일에 큰 하락이 나타날 가능성을 과거 데이터로 분류할 수 있을까?
    </div>
    """, unsafe_allow_html=True)

    st.subheader("문제에 따라 모델 종류를 구분")
    q1, q2 = st.columns(2)
    with q1:
        st.markdown("#### 🧩 시장 유형 찾기")
        st.write("시장별 수익률·변동성·최대 낙폭 등의 특징이 비슷한 시장끼리 묶는 문제이다.")
        st.metric("적합한 학습 방식", "군집화")
        st.write("정답 그룹이 미리 주어지지 않으므로 비지도학습을 사용한다.")
    with q2:
        st.markdown("#### 📉 다음 거래일 하락 사례 분류")
        st.write("최근 수익률 흐름을 이용해 다음 거래일이 설정한 하락 기준에 해당하는지 분류한다.")
        st.metric("적합한 학습 방식", "분류")
        st.write("하락/비하락이라는 두 범주를 예측하므로 지도학습을 사용한다.")

    st.info("회귀는 연속적인 수치(예: 다음 날 수익률 자체)를 예측할 때 적합하다. 현재 앱의 하락 위험 질문은 하락/비하락 범주를 구분하므로 분류가 적합하다.")
    st.subheader("이번 모델 평가의 범위")
    st.write("모델 평가 탭에서는 하락 위험 분류를 중심으로 기준선과 후보 모델 3종을 같은 시간 분할·변수 조건에서 비교한다. 군집화는 별도로 여러 군집 알고리즘의 군집 품질 지표를 비교한다.")
    st.warning("이 프로젝트는 금융 데이터 분석 학습용이다. 과거 자료에서 얻은 결과가 미래 수익을 보장하거나 손실을 예방하지 않는다.")

with tab_data:
    st.header("② 데이터 분석")
    st.write("모델에 넣기 전 데이터의 규모, 기간, 변수와 예측 대상의 분포를 확인한다.")
    dates = pd.to_datetime(clean_df["Date"], errors="coerce").dropna()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("원본 데이터 행", f"{len(raw_df):,}")
    c2.metric("전처리 후 행", f"{len(clean_df):,}")
    c3.metric("분석 시장 수", f"{clean_df['Index'].nunique():,}")
    c4.metric("결측 셀 수", f"{int(clean_df.isna().sum().sum()):,}")
    if not dates.empty:
        st.write(f"**분석 기간:** {dates.min().date()} ~ {dates.max().date()}")
    st.write(f"**가격 변수:** `{price_col}`")
    st.write("**군집화 입력 변수:** 연환산 평균 수익률, 연환산 변동성, 하락 변동성, 최대 낙폭, 상승일 비율 등 계산 가능한 시장별 특징값")
    st.write("**분류 입력 변수:** 당일 수익률, 최근 5거래일 누적 수익률, 최근 20거래일 변동성, 최근 20거래일 평균 수익률")
    with st.expander("시장별 특징 데이터 미리 보기"):
        st.dataframe(market_features, use_container_width=True, hide_index=True)
    with st.expander("전처리 데이터 미리 보기"):
        st.dataframe(clean_df.head(30), use_container_width=True, hide_index=True)

    threshold_pct = st.slider("큰 하락 기준: 다음 거래일 수익률(%)", -10.0, -1.0, -2.0, 0.5, key="data_threshold")
    class_df, class_features = make_classification_data(clean_df, threshold_pct / 100)
    if not class_df.empty:
        counts = class_df["target"].value_counts().reindex([0, 1], fill_value=0)
        total = int(counts.sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("분류 실험 행 수", f"{total:,}")
        c2.metric("비하락 사례", f"{int(counts[0]):,} ({counts[0]/total:.1%})" if total else "0")
        c3.metric("하락 사례", f"{int(counts[1]):,} ({counts[1]/total:.1%})" if total else "0")
        st.caption("하락 사례 비율이 매우 낮으면 단순히 비하락만 예측해도 정확도가 높아질 수 있다. 따라서 정확도만 보지 않고 정밀도·재현율·F1도 함께 비교한다.")
    else:
        st.warning("분류 모델을 만들기에 유효한 데이터가 부족하다.")

with tab_baseline:
    st.header("③ 기준선(Baseline)")
    st.write("기준선은 복잡한 모델의 성능이 실제로 나아졌는지 판단하기 위한 출발점이다.")
    st.subheader("하락 위험 분류의 기준선")
    st.markdown("""
    **다수 범주 예측(DummyClassifier, most_frequent)**

    학습 데이터에서 더 많이 나타난 범주 하나만 계속 예측한다.
    예를 들어 비하락이 대부분이면 모든 테스트 사례를 비하락이라고 예측한다.
    입력 변수의 패턴을 학습하지 않는 단순한 기준 모델이다.
    """)
    st.subheader("군집화의 기준선")
    st.markdown("""
    **K-Means (기본 비교 모델)**

    선택한 시장 특징값을 표준화한 뒤 K=3으로 나눈 결과를 기준으로 삼는다.
    다른 군집 알고리즘도 동일한 시장·특징 변수·그룹 수를 사용해 실루엣 점수 등을 비교한다.
    """)
    st.info("기준선은 '성능이 좋아야 하는 모델'이 아니라, 후보 모델이 넘어야 할 최소 비교선이다. 기준선보다 나아지지 않았다면 복잡한 모델을 선택할 근거가 약하다.")
    st.subheader("공정한 비교를 위한 고정 조건")
    st.markdown("""
    - 동일한 원본 데이터와 전처리 결과 사용
    - 분류 모델은 동일한 입력 변수 4개 사용
    - 날짜 순서를 유지한 동일한 학습/테스트 분할 사용
    - 분류 기준(다음 거래일 수익률 임계값)과 평가 지표 동일
    - 난수 시드와 주요 모델 설정 고정
    """)

with tab_eval:
    st.header("④ 모델 평가")
    st.write("분류 모델은 기준선과 후보 3종을 비교하고, 군집화 모델은 동일한 입력값으로 군집 품질을 비교한다.")
    threshold_pct = st.slider("큰 하락 기준: 다음 거래일 수익률(%)", -10.0, -1.0, -2.0, 0.5, key="eval_threshold")
    class_df, class_features = make_classification_data(clean_df, threshold_pct / 100)

    st.subheader("A. 하락 위험 분류 모델 비교")
    if len(class_df) < 100 or class_df["target"].nunique() < 2:
        st.warning("분류 비교를 하려면 유효한 데이터 100건 이상과 하락/비하락 두 범주가 필요하다.")
    else:
        unique_dates = np.array(sorted(class_df["Date"].unique()))
        split_pos = int(len(unique_dates) * 0.8)
        if split_pos <= 0 or split_pos >= len(unique_dates):
            st.warning("날짜 기준 학습/테스트 분할이 불가능하다.")
        else:
            split_date = unique_dates[split_pos]
            train = class_df[class_df["Date"] < split_date]
            test = class_df[class_df["Date"] >= split_date]
            X_train, y_train = train[class_features], train["target"]
            X_test, y_test = test[class_features], test["target"]

            if len(test) == 0 or y_train.nunique() < 2:
                st.warning("학습 데이터에 두 범주가 모두 있거나 테스트 데이터가 있어야 한다.")
            else:
                models = {
                    "기준선: 다수 범주 예측": DummyClassifier(strategy="most_frequent"),
                    "로지스틱 회귀": Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
                    ]),
                    "랜덤 포레스트": Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                         random_state=42, n_jobs=-1, min_samples_leaf=2))
                    ]),
                    "히스토그램 기반 부스팅": Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("model", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08,
                                                                 random_state=42))
                    ])
                }
                rows, prediction_store = [], {}
                for name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test)
                        proba = None
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(X_test)[:, 1]
                        auc = roc_auc_score(y_test, proba) if proba is not None and y_test.nunique() == 2 else np.nan
                        rows.append({
                            "모델": name,
                            "정확도": accuracy_score(y_test, pred),
                            "정밀도": precision_score(y_test, pred, zero_division=0),
                            "재현율": recall_score(y_test, pred, zero_division=0),
                            "F1": f1_score(y_test, pred, zero_division=0),
                            "ROC-AUC": auc
                        })
                        prediction_store[name] = (model, pred)
                    except Exception as err:
                        st.warning(f"{name} 학습 중 오류가 발생했다: {err}")

                if rows:
                    result = pd.DataFrame(rows)
                    baseline_name = "기준선: 다수 범주 예측"
                    base_row = result[result["모델"] == baseline_name]
                    candidate_rows = result[result["모델"] != baseline_name]
                    if not candidate_rows.empty:
                        best_name = candidate_rows.sort_values(["F1", "재현율"], ascending=False).iloc[0]["모델"]
                        best_row = result[result["모델"] == best_name].iloc[0]
                    else:
                        best_name, best_row = None, None

                    st.dataframe(
                        result.style.format({c: "{:.3f}" for c in ["정확도", "정밀도", "재현율", "F1", "ROC-AUC"]}),
                        use_container_width=True, hide_index=True
                    )
                    st.caption("모든 분류 모델은 동일한 날짜 기준 학습/테스트 구간과 동일한 입력 변수를 사용했다. 지표는 테스트 데이터에서 계산했다.")
                    if best_name and not base_row.empty:
                        base = base_row.iloc[0]
                        st.subheader("결과 해석")
                        st.write(f"F1을 우선하고 재현율을 보조 기준으로 비교했을 때 후보 중 **{best_name}**의 점수가 가장 높았다.")
                        delta_f1 = float(best_row["F1"] - base["F1"])
                        delta_recall = float(best_row["재현율"] - base["재현율"])
                        st.write(f"- 기준선 대비 F1 변화: **{delta_f1:+.3f}**")
                        st.write(f"- 기준선 대비 재현율 변화: **{delta_recall:+.3f}**")
                        if delta_f1 > 0 and delta_recall > 0:
                            st.success("선정 후보는 기준선보다 F1과 재현율이 모두 높다. 다만 다른 기간에서도 유지되는지 추가 검증이 필요하다.")
                        elif delta_f1 > 0:
                            st.info("F1은 기준선보다 높지만 재현율 개선은 확인되지 않았다. 하락 사례를 놓치는 정도도 함께 고려해야 한다.")
                        else:
                            st.warning("선정 후보가 F1 기준으로 기준선을 넘지 못했다. 현재 결과만으로 복잡한 모델을 선택할 근거가 충분하지 않다.")

                        st.markdown("**선정 후보의 해석**")
                        st.write(f"정확도 {best_row['정확도']:.3f}, 정밀도 {best_row['정밀도']:.3f}, 재현율 {best_row['재현율']:.3f}, F1 {best_row['F1']:.3f}이다.")
                        st.write("재현율은 실제 하락 사례 중 찾아낸 비율이며, 정밀도는 하락이라고 예측한 사례 중 실제 하락인 비율이다. 하락 사례가 적은 데이터에서는 정확도만으로 모델을 판단하지 않는다.")

                        chosen_model, chosen_pred = prediction_store[best_name]
                        cm = confusion_matrix(y_test, chosen_pred, labels=[0, 1])
                        cm_df = pd.DataFrame(cm, index=["실제 비하락", "실제 하락"], columns=["비하락 예측", "하락 예측"])
                        fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                                           title=f"{best_name}: 혼동행렬",
                                           labels={"x": "예측", "y": "실제", "color": "사례 수"})
                        st.plotly_chart(fig_cm, use_container_width=True)
                        st.write(f"학습 기간: {pd.Timestamp(train['Date'].min()).date()} ~ {pd.Timestamp(train['Date'].max()).date()} ({len(train):,}건)")
                        st.write(f"테스트 기간: {pd.Timestamp(test['Date'].min()).date()} ~ {pd.Timestamp(test['Date'].max()).date()} ({len(test):,}건)")
                        st.download_button("📥 분류 모델 비교 결과 CSV", result.to_csv(index=False).encode("utf-8-sig"),
                                           "classification_model_comparison.csv", "text/csv")
                    else:
                        st.warning("기준선 또는 비교 후보 모델 결과가 충분하지 않다.")
                else:
                    st.error("모델 비교 결과를 계산하지 못했다.")

    st.divider()
    st.subheader("B. 시장 유형 군집화 모델 비교")
    st.write("군집화는 정답 라벨이 없으므로 정확도 대신 실루엣 점수, Davies-Bouldin 지수, Calinski-Harabasz 지수를 참고한다.")
    feat_cols = [c for c in [
        "annualized_return", "annualized_volatility", "downside_volatility",
        "max_drawdown", "positive_day_ratio", "average_absolute_return",
        "average_intraday_range"
    ] if c in market_features.columns and market_features[c].notna().any()]
    if len(feat_cols) < 2 or len(market_features) < 4:
        st.info("군집화 모델을 비교할 시장 특징값이 충분하지 않다.")
    else:
        chosen_feats = st.multiselect("군집화 비교에 사용할 동일한 특징 변수", feat_cols, default=feat_cols,
                                      key="cluster_features")
        k = st.slider("군집 수 K", 2, min(6, len(market_features)-1),
                      min(3, min(6, len(market_features)-1)), key="cluster_k")
        if len(chosen_feats) < 2:
            st.warning("군집화 비교에는 특징 변수를 2개 이상 선택해야 한다.")
        else:
            cluster_data = market_features[["Index"] + chosen_feats].dropna(subset=chosen_feats, how="all").copy()
            if len(cluster_data) <= k:
                st.warning("선택한 조건에서 군집 수보다 시장 수가 많아야 한다.")
            else:
                imputer = SimpleImputer(strategy="median")
                X = imputer.fit_transform(cluster_data[chosen_feats])
                X_scaled = RobustScaler().fit_transform(X)
                cluster_models = {
                    "기준선: K-Means": KMeans(n_clusters=k, random_state=42, n_init=20),
                    "계층적 군집화": AgglomerativeClustering(n_clusters=k),
                    "가우시안 혼합 모델": GaussianMixture(n_components=k, random_state=42, n_init=5)
                }
                cluster_rows = []
                for name, model in cluster_models.items():
                    try:
                        if name == "가우시안 혼합 모델":
                            labels = model.fit_predict(X_scaled)
                        else:
                            labels = model.fit_predict(X_scaled)
                        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
                            continue
                        cluster_rows.append({
                            "모델": name,
                            "실루엣 점수(높을수록 좋음)": silhouette_score(X_scaled, labels),
                            "Davies-Bouldin(낮을수록 좋음)": davies_bouldin_score(X_scaled, labels),
                            "Calinski-Harabasz(높을수록 좋음)": calinski_harabasz_score(X_scaled, labels)
                        })
                    except Exception as err:
                        st.warning(f"{name} 계산 중 오류: {err}")
                if cluster_rows:
                    cluster_result = pd.DataFrame(cluster_rows)
                    st.dataframe(cluster_result.round(3), use_container_width=True, hide_index=True)
                    best_cluster = cluster_result.sort_values(
                        ["실루엣 점수(높을수록 좋음)", "Davies-Bouldin(낮을수록 좋음)"],
                        ascending=[False, True]
                    ).iloc[0]
                    st.write(f"실루엣 점수를 우선하고 Davies-Bouldin 지수를 보조 기준으로 보면 **{best_cluster['모델']}**이 이번 설정에서 가장 높은 평가를 받았다.")
                    st.caption("군집 품질 지표는 선택한 변수와 K에 따라 달라진다. 점수가 높다고 경제적 의미가 자동으로 보장되거나 미래 주가를 예측하는 것은 아니다.")
                    st.download_button("📥 군집화 모델 비교 결과 CSV", cluster_result.to_csv(index=False).encode("utf-8-sig"),
                                       "clustering_model_comparison.csv", "text/csv")
                else:
                    st.warning("유효한 군집 결과를 계산하지 못했다. K 또는 변수 구성을 바꾸어 확인해 보자.")

    st.divider()
    st.subheader("평가 결과를 보고서에 옮길 때")
    st.write("표에 표시된 실제 수치와 기준선 대비 차이를 사용해 작성한다. 계산되지 않은 값이나 확인하지 않은 모델 성능을 임의로 기재하지 않는다.")
    st.markdown("""
    - **질문:** 다음 거래일의 큰 하락 여부를 과거 금융시장 데이터로 분류할 수 있는가?
    - **기준선:** 학습 데이터에서 가장 많은 범주를 테스트 데이터 전체에 예측하는 다수 범주 기준선이다.
    - **비교:** 로지스틱 회귀, 랜덤 포레스트, 히스토그램 기반 부스팅을 동일한 날짜 분할과 입력 변수로 비교한다.
    - **선정:** F1을 우선하고 재현율을 보조 기준으로 살펴보되, 기준선보다 개선되지 않으면 우수한 모델이라고 결론 내리지 않는다.
    """)

st.divider()
st.caption("분석 결과는 과거 데이터에 대한 교육용 실험이다. 거래 비용, 환율, 세금, 국가별 휴장일과 실제 투자 조건을 모두 반영하지 않으며 투자 판단의 근거로 사용해서는 안 된다.")
