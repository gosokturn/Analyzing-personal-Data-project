import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
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
    page_title="세계 금융시장 한눈에 보기",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1250px;}
[data-testid="stMetric"] {background:#fff; border:1px solid #e8edf3;
    padding:15px 17px; border-radius:14px;}
div[data-testid="stExpander"] {border:1px solid #e8edf3; border-radius:12px;}
h1,h2,h3 {color:#172b4d;}
.intro-card {background:#f2f6fb; padding:22px 24px; border-radius:16px;
    border:1px solid #e3ebf5; margin-bottom:18px;}
.info-card {background:#f2f6fb; padding:18px 20px; border-radius:14px;
    border:1px solid #e3ebf5; margin-bottom:12px;}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR / "pages"

@st.cache_data(show_spinner="세계 금융시장 데이터를 불러오는 중...")
def get_data():
    raw = load_raw_data()
    clean, price_col, audit = prepare_market_data(raw, winsorize=False)
    features = build_market_features(clean, price_col, min_observations=60)
    return raw, clean, price_col, audit, features

try:
    raw_df, clean_df, price_col, audit, market_features = get_data()
except Exception as e:
    data_error = e
    raw_df, clean_df, price_col, audit, market_features = None, None, None, None, None


def home_page():
    st.title("🌍 세계 금융시장 한눈에 보기")
    st.markdown("""
    <div class="intro-card">
      <h3>여러 나라의 주식시장은 어떻게 움직일까요?</h3>
      <div>데이터를 통해 비슷한 시장을 찾고, 시장 사이의 관계와 위험 특성을 알아봅니다.
      전문 투자용 거래 프로그램이 아니라 금융 데이터를 이해하기 위한 학습 서비스입니다.</div>
    </div>
    """, unsafe_allow_html=True)

    if raw_df is None:
        st.error("분석 데이터를 준비하지 못했습니다. 데이터 주소와 원본 CSV의 열 구성을 확인해 주세요.")
        st.exception(data_error)
        return

    dates = pd.to_datetime(clean_df["Date"], errors="coerce").dropna()
    market_count = clean_df["Index"].nunique()
    start_date = dates.min().date() if not dates.empty else None
    end_date = dates.max().date() if not dates.empty else None
    avg_vol = market_features["annualized_volatility"].mean() if not market_features.empty else np.nan

    a, b, c, d = st.columns(4)
    a.metric("분석 금융시장", f"{market_count:,}개")
    b.metric("분석 데이터 행", f"{len(clean_df):,}개")
    c.metric("분석 기간", f"{start_date} ~ {end_date}" if start_date and end_date else "확인 불가")
    d.metric("평균 연환산 변동성", f"{avg_vol:.1%}" if pd.notna(avg_vol) else "계산 불가")

    st.divider()
    st.subheader("데이터에서 살펴볼 수 있는 네 가지 질문")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧩 시장 유형은 어떻게 나뉠까?")
        if len(market_features) >= 3:
            st.write(f"현재 데이터에서 {len(market_features)}개 시장을 수익률과 위험 지표의 유사성에 따라 묶어 볼 수 있습니다.")
        else:
            st.info("군집화에 필요한 시장별 특징값이 충분하지 않습니다.")
        st.markdown("#### 📉 변동성이 가장 큰 시장은?")
        if not market_features.empty and market_features["annualized_volatility"].notna().any():
            highest = market_features.loc[market_features["annualized_volatility"].idxmax()]
            st.write(f"**{highest['Index']}** 시장의 과거 연환산 변동성이 가장 높게 계산되었습니다({highest['annualized_volatility']:.1%}).")
        else:
            st.info("변동성을 비교할 수 있는 시장이 없습니다.")
    with col2:
        st.markdown("#### 🔗 서로 비슷하게 움직이는 시장은?")
        pivot = clean_df.pivot_table(index="Date", columns="Index", values="daily_return_used", aggfunc="mean")
        valid = pivot.columns[pivot.count() >= 60].tolist()
        corr = pivot[valid].corr(min_periods=60) if len(valid) >= 2 else pd.DataFrame()
        if not corr.empty:
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().dropna()
            if not upper.empty:
                pair = upper.abs().idxmax()
                value = corr.loc[pair[0], pair[1]]
                st.write(f"**{pair[0]}**과 **{pair[1]}**의 상관계수는 {value:.2f}입니다.")
                st.caption("상관관계는 인과관계나 비례관계를 뜻하지 않습니다.")
            else:
                st.info("비교 가능한 시장 쌍이 없습니다.")
        else:
            st.info("상관관계를 계산할 만큼 데이터가 겹치는 시장이 부족합니다.")
        st.markdown("#### 🧭 최근 데이터에서 확인할 점")
        if not clean_df.empty:
            latest_date = clean_df["Date"].max()
            latest_rows = clean_df[clean_df["Date"] == latest_date]
            st.write(f"가장 최근 날짜는 **{pd.Timestamp(latest_date).date()}**이며, 해당 날짜에 {latest_rows['Index'].nunique()}개 시장 기록이 있습니다.")

    st.divider()
    st.subheader("기존 분석 기능 바로가기")
    st.write("왼쪽 메뉴에서 데이터 탐색, 데이터 전처리, 기존 시장 분석 기능을 계속 사용할 수 있습니다.")
    st.info("왼쪽의 '급락 위험 예측' 메뉴에서 세 가지 예측 방법을 비교할 수 있습니다.")
    st.warning("이 앱은 과거 금융 데이터 분석을 위한 학습 도구이며 미래 수익이나 손실 방지를 보장하지 않습니다.")


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


def model_evaluation_page():
    st.title("📉 주식시장 급락 위험 예측")
    st.caption(
        "과거 시장 흐름을 바탕으로 다음 거래일의 큰 하락 가능성을 분류하고, "
        "서로 다른 예측 방법 3가지를 비교합니다."
    )

    if raw_df is None:
        st.error("데이터를 준비하지 못해 급락 위험을 분석할 수 없습니다.")
        st.exception(data_error)
        st.stop()

    threshold_pct = st.slider(
        "어느 정도 하락을 '큰 하락'으로 볼까요?",
        min_value=-10.0,
        max_value=-1.0,
        value=-2.0,
        step=0.5,
        help="예를 들어 -2%를 선택하면 다음 거래일 수익률이 -2% 이하인 경우를 큰 하락으로 분류합니다."
    )
    class_df, class_features = make_classification_data(
        clean_df, threshold_pct / 100
    )

    tab_question, tab_data, tab_baseline, tab_eval = st.tabs([
        "① 무엇을 예측할까?",
        "② 어떤 데이터를 볼까?",
        "③ 무엇과 비교할까?",
        "④ 예측 결과"
    ])

    with tab_question:
        st.header("무엇을 해결하려는 분석인가요?")
        st.markdown("""
        <div class="info-card">
        <b>분석 질문</b><br>
        최근 주식시장 움직임을 살펴보고, 다음 거래일에 큰 하락이 발생할 가능성이 있는지 미리 분류할 수 있을까요?
        </div>
        """, unsafe_allow_html=True)

        st.subheader("왜 분류 모델을 사용하나요?")
        st.write(
            "이번 분석의 결과는 다음 거래일 수익률의 정확한 숫자가 아니라 "
            "'큰 하락' 또는 '큰 하락이 아님' 두 가지 범주입니다. "
            "따라서 연속된 숫자를 예측하는 회귀보다 분류 방식이 적합합니다."
        )

        left, right = st.columns(2)
        with left:
            st.markdown("#### 🔴 큰 하락")
            st.write(
                f"다음 거래일 수익률이 **{threshold_pct:.1f}% 이하**인 경우입니다."
            )
        with right:
            st.markdown("#### 🟢 그 외의 움직임")
            st.write(
                f"다음 거래일 수익률이 **{threshold_pct:.1f}%보다 높은** 경우입니다."
            )

        st.subheader("비교할 예측 방법 3가지")
        model_cols = st.columns(3)
        with model_cols[0]:
            st.markdown("#### 1. 로지스틱 회귀")
            st.write(
                "최근 수익률과 변동성의 조합을 바탕으로 하락 여부를 구분합니다. "
                "결과를 비교적 이해하기 쉽다는 장점이 있습니다."
            )
        with model_cols[1]:
            st.markdown("#### 2. 랜덤 포레스트")
            st.write(
                "여러 개의 결정 트리가 각자 판단한 결과를 모읍니다. "
                "변수 사이의 복잡한 관계를 찾는 데 활용할 수 있습니다."
            )
        with model_cols[2]:
            st.markdown("#### 3. 부스팅 모델")
            st.write(
                "앞선 예측에서 놓친 부분을 다음 단계에서 보완하며 여러 작은 예측기를 결합합니다. "
                "복잡한 패턴을 학습할 수 있습니다."
            )

        st.info(
            "세 모델은 같은 데이터와 같은 입력 변수를 사용합니다. "
            "기준선은 후보 모델 3개와 별도로 두어, 학습한 모델이 단순한 예측보다 나은지 확인합니다."
        )
        st.warning(
            "이 분석은 과거 자료를 이용한 학습용 실험입니다. "
            "급락을 확실하게 예측하거나 투자 손실을 막아 주는 도구는 아닙니다."
        )

    with tab_data:
        st.header("분석에 사용하는 데이터")
        st.write(
            "원본 주가 자료를 정리한 뒤, 시장별로 날짜 순서에 따라 "
            "최근 움직임을 계산해 예측에 사용합니다."
        )

        dates = pd.to_datetime(clean_df["Date"], errors="coerce").dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("정리된 데이터", f"{len(clean_df):,}행")
        c2.metric("분석 시장", f"{clean_df['Index'].nunique():,}개")
        c3.metric("급락 기준", f"{threshold_pct:.1f}%")

        if not dates.empty:
            st.write(
                f"**자료 기간:** {dates.min().date()}부터 {dates.max().date()}까지"
            )

        st.subheader("예측에 사용하는 정보")
        feature_explanations = pd.DataFrame({
            "사용 정보": [
                "당일 수익률",
                "최근 5거래일 누적 수익률",
                "최근 20거래일 변동성",
                "최근 20거래일 평균 수익률"
            ],
            "쉽게 말하면": [
                "오늘 시장이 얼마나 오르거나 내렸는지",
                "최근 일주일 정도의 흐름이 상승인지 하락인지",
                "최근 한 달 정도 시장 움직임이 얼마나 크게 흔들렸는지",
                "최근 한 달 정도 수익률의 평균적인 방향"
            ]
        })
        st.dataframe(
            feature_explanations,
            use_container_width=True,
            hide_index=True
        )

        if not class_df.empty:
            counts = class_df["target"].value_counts().reindex(
                [0, 1], fill_value=0
            )
            total = int(counts.sum())
            a, b, c = st.columns(3)
            a.metric("예측에 사용 가능한 기록", f"{total:,}건")
            b.metric(
                "큰 하락이 아닌 기록",
                f"{int(counts[0]):,}건 ({counts[0] / total:.1%})"
            )
            c.metric(
                "큰 하락 기록",
                f"{int(counts[1]):,}건 ({counts[1] / total:.1%})"
            )

            st.caption(
                "큰 하락 사례가 드물 수 있습니다. 이런 데이터에서는 모든 날을 "
                "'큰 하락이 아님'이라고 예측해도 정확도가 높아질 수 있으므로 "
                "실제 하락을 얼마나 찾아냈는지도 함께 살펴봅니다."
            )
            with st.expander("예측 데이터 일부 보기"):
                preview = class_df[
                    ["Index", "Date"] + class_features + ["next_return", "target"]
                ].head(30).copy()
                preview = preview.rename(columns={
                    "Index": "시장",
                    "Date": "날짜",
                    "return_today": "당일 수익률",
                    "return_5d": "최근 5일 수익률",
                    "volatility_20d": "최근 20일 변동성",
                    "mean_return_20d": "최근 20일 평균 수익률",
                    "next_return": "다음 거래일 수익률",
                    "target": "큰 하락 여부(1=예)"
                })
                st.dataframe(preview, use_container_width=True, hide_index=True)
        else:
            st.warning(
                "현재 기준으로 계산할 수 있는 예측 데이터가 부족합니다. "
                "자료 기간이나 급락 기준을 바꾸어 확인해 주세요."
            )

    with tab_baseline:
        st.header("비교의 출발점: 단순 예측")
        st.write(
            "새로운 모델이 정말 도움이 되는지 알아보려면, "
            "먼저 복잡한 계산을 하지 않는 단순한 예측과 비교해야 합니다."
        )
        st.subheader("기준선: 가장 자주 나타난 결과만 예측")
        st.markdown("""
        예를 들어 과거 기록 대부분이 '큰 하락이 아님'이라면,
        기준선은 모든 테스트 날짜를 '큰 하락이 아님'이라고 예측합니다.

        최근 시장 흐름이나 변동성은 전혀 살펴보지 않습니다.
        따라서 이 단순한 기준선보다 세 후보 모델이 더 나은 결과를 내는지 확인할 수 있습니다.
        """)

        st.subheader("공정한 비교를 위해")
        st.markdown("""
        - 세 후보 모델과 기준선에 같은 학습 자료와 시험 자료를 사용합니다.
        - 미래 정보를 미리 보지 않도록 과거 날짜를 학습에, 이후 날짜를 시험에 사용합니다.
        - 네 가지 입력 정보와 큰 하락 기준을 동일하게 유지합니다.
        - 정확도뿐 아니라 정밀도, 재현율, F1 점수도 함께 확인합니다.
        """)

        st.info(
            "**기준선은 네 번째 후보 모델이 아닙니다.** "
            "세 모델의 성능이 단순 예측보다 나은지 확인하기 위한 비교용 출발점입니다."
        )

    with tab_eval:
        st.header("세 가지 모델의 실제 예측 결과")
        st.write(
            "과거 날짜 약 80%를 학습에 사용하고, 그 이후 날짜 약 20%를 "
            "시험 자료로 남겨 성능을 비교합니다."
        )

        if len(class_df) < 100 or class_df["target"].nunique() < 2:
            st.warning(
                "비교를 진행하려면 유효한 기록이 100건 이상이고, "
                "큰 하락과 그 외의 사례가 모두 있어야 합니다."
            )
        else:
            unique_dates = np.array(sorted(class_df["Date"].unique()))
            split_pos = int(len(unique_dates) * 0.8)

            if split_pos <= 0 or split_pos >= len(unique_dates):
                st.warning("날짜 순서에 따른 학습·시험 자료 분할이 어렵습니다.")
            else:
                split_date = unique_dates[split_pos]
                train = class_df[class_df["Date"] < split_date].copy()
                test = class_df[class_df["Date"] >= split_date].copy()

                X_train = train[class_features]
                y_train = train["target"]
                X_test = test[class_features]
                y_test = test["target"]

                if (
                    test.empty
                    or y_train.nunique() < 2
                    or y_test.nunique() < 2
                ):
                    st.warning(
                        "학습 또는 시험 자료에 큰 하락과 그 외의 사례가 모두 포함되어야 "
                        "안정적인 비교가 가능합니다. 급락 기준을 조정해 보세요."
                    )
                else:
                    # 기준선 1개 + 실제 비교 대상 모델 3개
                    models = {
                        "기준선: 가장 많은 범주": DummyClassifier(
                            strategy="most_frequent"
                        ),
                        "로지스틱 회귀": Pipeline([
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("model", LogisticRegression(
                                class_weight="balanced",
                                max_iter=1000,
                                random_state=42
                            ))
                        ]),
                        "랜덤 포레스트": Pipeline([
                            ("imputer", SimpleImputer(strategy="median")),
                            ("model", RandomForestClassifier(
                                n_estimators=300,
                                class_weight="balanced",
                                random_state=42,
                                n_jobs=-1,
                                min_samples_leaf=2
                            ))
                        ]),
                        "히스토그램 기반 부스팅": Pipeline([
                            ("imputer", SimpleImputer(strategy="median")),
                            ("model", HistGradientBoostingClassifier(
                                max_iter=150,
                                learning_rate=0.08,
                                random_state=42
                            ))
                        ])
                    }

                    rows = []
                    prediction_store = {}

                    for name, model in models.items():
                        try:
                            model.fit(X_train, y_train)
                            pred = model.predict(X_test)

                            proba = None
                            if hasattr(model, "predict_proba"):
                                proba = model.predict_proba(X_test)[:, 1]

                            auc = (
                                roc_auc_score(y_test, proba)
                                if proba is not None
                                else np.nan
                            )

                            rows.append({
                                "모델": name,
                                "정확도": accuracy_score(y_test, pred),
                                "정밀도": precision_score(
                                    y_test, pred, zero_division=0
                                ),
                                "재현율": recall_score(
                                    y_test, pred, zero_division=0
                                ),
                                "F1 점수": f1_score(
                                    y_test, pred, zero_division=0
                                ),
                                "ROC-AUC": auc
                            })
                            prediction_store[name] = (model, pred)

                        except Exception as err:
                            st.warning(
                                f"{name} 계산 중 문제가 발생했습니다: {err}"
                            )

                    if rows:
                        result = pd.DataFrame(rows)
                        baseline_name = "기준선: 가장 많은 범주"
                        baseline_rows = result[
                            result["모델"] == baseline_name
                        ]
                        candidate_rows = result[
                            result["모델"] != baseline_name
                        ]

                        st.subheader("성능 비교표")
                        st.dataframe(
                            result.style.format({
                                "정확도": "{:.3f}",
                                "정밀도": "{:.3f}",
                                "재현율": "{:.3f}",
                                "F1 점수": "{:.3f}",
                                "ROC-AUC": "{:.3f}"
                            }),
                            use_container_width=True,
                            hide_index=True
                        )

                        st.caption(
                            "각 점수는 0~1 범위입니다. 일반적으로 높을수록 좋지만, "
                            "급락 예측에서는 실제 하락을 찾아내는 재현율과 "
                            "정밀도·F1을 함께 살펴봐야 합니다."
                        )

                        if not candidate_rows.empty and not baseline_rows.empty:
                            best_row = candidate_rows.sort_values(
                                ["F1 점수", "재현율"],
                                ascending=[False, False]
                            ).iloc[0]
                            base_row = baseline_rows.iloc[0]

                            delta_f1 = float(
                                best_row["F1 점수"] - base_row["F1 점수"]
                            )
                            delta_recall = float(
                                best_row["재현율"] - base_row["재현율"]
                            )

                            st.subheader("가장 높은 평가를 받은 모델")
                            st.markdown(
                                f"### {best_row['모델']}"
                            )
                            m1, m2, m3 = st.columns(3)
                            m1.metric(
                                "F1 점수",
                                f"{best_row['F1 점수']:.3f}",
                                f"{delta_f1:+.3f} (기준선 대비)"
                            )
                            m2.metric(
                                "재현율",
                                f"{best_row['재현율']:.3f}",
                                f"{delta_recall:+.3f} (기준선 대비)"
                            )
                            m3.metric(
                                "정밀도",
                                f"{best_row['정밀도']:.3f}"
                            )

                            if delta_f1 > 0 and delta_recall > 0:
                                st.success(
                                    "이 모델은 기준선보다 F1과 재현율이 모두 높습니다. "
                                    "이번 시험 자료에서는 단순 예측보다 나은 결과입니다."
                                )
                            elif delta_f1 > 0:
                                st.info(
                                    "F1 점수는 기준선보다 높지만 재현율은 개선되지 않았습니다. "
                                    "실제 급락 사례를 놓치는 정도를 추가로 살펴봐야 합니다."
                                )
                            else:
                                st.warning(
                                    "가장 높은 후보 모델도 F1 점수에서 기준선을 넘지 못했습니다. "
                                    "현재 결과만으로 후보 모델이 더 낫다고 판단하기 어렵습니다."
                                )

                            st.subheader("선택 모델이 맞히고 놓친 사례")
                            chosen_name = best_row["모델"]
                            chosen_model, chosen_pred = prediction_store[chosen_name]
                            cm = confusion_matrix(
                                y_test, chosen_pred, labels=[0, 1]
                            )
                            cm_df = pd.DataFrame(
                                cm,
                                index=["실제 큰 하락 아님", "실제 큰 하락"],
                                columns=["큰 하락 아님 예측", "큰 하락 예측"]
                            )
                            fig_cm = px.imshow(
                                cm_df,
                                text_auto=True,
                                aspect="auto",
                                color_continuous_scale="Blues",
                                title="실제 결과와 모델 예측 비교",
                                labels={
                                    "x": "모델의 예측",
                                    "y": "실제 결과",
                                    "color": "기록 수"
                                }
                            )
                            st.plotly_chart(
                                fig_cm, use_container_width=True
                            )

                            st.write(
                                f"**학습 기간:** "
                                f"{pd.Timestamp(train['Date'].min()).date()} ~ "
                                f"{pd.Timestamp(train['Date'].max()).date()}"
                            )
                            st.write(
                                f"**시험 기간:** "
                                f"{pd.Timestamp(test['Date'].min()).date()} ~ "
                                f"{pd.Timestamp(test['Date'].max()).date()}"
                            )

                            st.download_button(
                                "📥 모델 비교 결과 내려받기",
                                data=result.to_csv(
                                    index=False
                                ).encode("utf-8-sig"),
                                file_name="crash_risk_model_comparison.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning(
                                "기준선과 세 후보 모델의 결과를 모두 계산하지 못했습니다."
                            )
                    else:
                        st.error(
                            "모델 결과를 계산하지 못했습니다. 데이터와 라이브러리 버전을 확인해 주세요."
                        )

        st.divider()
        st.subheader("점수를 해석하는 방법")
        st.markdown("""
        - **정확도:** 전체 예측 중 맞힌 비율
        - **정밀도:** 큰 하락이라고 예측한 날 중 실제로 큰 하락이었던 비율
        - **재현율:** 실제 큰 하락 중 모델이 찾아낸 비율
        - **F1 점수:** 정밀도와 재현율을 함께 고려한 점수
        - **ROC-AUC:** 큰 하락과 그 외의 사례를 구별하는 능력

        특히 큰 하락 사례가 드물면 정확도만으로 모델을 평가하면 안 됩니다.
        점수가 기준선보다 높더라도 다른 시기의 데이터에서도 비슷한 결과가 나오는지 추가로 확인해야 합니다.
        """)

    st.divider()
    st.caption(
        "이 결과는 과거 데이터에 대한 학습용 비교입니다. "
        "실제 시장의 급락을 확정적으로 예측하거나 투자 손실을 방지하지 않습니다."
    )

if not PAGES_DIR.exists():
    st.error("pages 폴더를 찾을 수 없습니다.")
    st.stop()

page_specs = [
    ("1_탐색.py", "데이터 탐색", "🔍"),
    ("2_데이터_전처리.py", "데이터 전처리", "🧹"),
    ("3_모델_학습_결과.py", "시장 유형·관계 분석", "🧠")
]
for filename, _, _ in page_specs:
    if not (PAGES_DIR / filename).is_file():
        st.error(f"페이지 파일이 없습니다: pages/{filename}")
        st.stop()

home = st.Page(home_page, title="홈", icon="🏠", default=True)
explore = st.Page(str(PAGES_DIR / "1_탐색.py"), title="데이터 탐색", icon="🔍")
preprocess = st.Page(str(PAGES_DIR / "2_데이터_전처리.py"), title="데이터 전처리", icon="🧹")
existing_analysis = st.Page(str(PAGES_DIR / "3_모델_학습_결과.py"), title="시장 유형·관계 분석", icon="📊")
evaluation = st.Page(model_evaluation_page, title="급락 위험 예측", icon="📉")

pg = st.navigation(
    {
        "서비스": [home, explore, preprocess],
        "분석": [existing_analysis],
        "위험 분석": [evaluation]
    },
    position="sidebar"
)
with st.sidebar:
    st.divider()
    st.caption("세계 금융시장 분석 · 학습용")
pg.run()
