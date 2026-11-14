import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from market_utils import load_raw_data, prepare_market_data, build_market_features

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="세계 금융시장 한눈에 보기",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 교육용 데이터 분석 앱 디자인 테마
st.markdown("""
<style>
.stApp {
    background-color: #f7f9fc;
}
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e1e8f0;
    padding: 16px 18px;
    border-radius: 14px;
    box-shadow: 0 2px 4px rgba(23, 43, 77, 0.03);
}
div[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e1e8f0;
    border-radius: 12px;
}
h1, h2, h3 {
    color: #172b4d;
}
.intro-card {
    background: #ffffff;
    padding: 24px 26px;
    border-radius: 16px;
    border-left: 6px solid #1e40af;
    border-top: 1px solid #e1e8f0;
    border-right: 1px solid #e1e8f0;
    border-bottom: 1px solid #e1e8f0;
    margin-bottom: 20px;
    box-shadow: 0 2px 6px rgba(23, 43, 77, 0.04);
}
.info-card {
    background: #eef4ff;
    padding: 18px 20px;
    border-radius: 14px;
    border: 1px solid #cfe0fc;
    margin-bottom: 14px;
    color: #172b4d;
}
.badge-drop {
    color: #dc2626;
    font-weight: 700;
}
.badge-normal {
    color: #16a34a;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR / "pages"


# 3. 데이터 로드 및 기본 전처리 (캐싱)
@st.cache_data(show_spinner="세계 금융시장 데이터를 불러오고 정리하는 중...")
def get_data():
    raw = load_raw_data()
    clean, price_col, audit = prepare_market_data(raw, winsorize=False)
    features = build_market_features(clean, price_col, min_observations=60)
    return raw, clean, price_col, audit, features


try:
    raw_df, clean_df, price_col, audit, market_features = get_data()
    data_error = None
except Exception as e:
    raw_df, clean_df, price_col, audit, market_features = None, None, None, None, None
    data_error = e


# 4. 홈 화면
def home_page():
    st.title("🌍 세계 금융시장 한눈에 보기")
    st.markdown("""
    <div class="intro-card">
      <h3 style="margin-top:0;">여러 나라의 주식시장은 어떻게 움직일까요?</h3>
      <div>세계 여러 금융시장의 움직임을 분석하고, 비슷한 시장과 시장 간 관계를 살펴보며, 
      과거 수익률과 변동성으로 다음 거래일의 급락 위험을 분류하는 교육용 데이터 분석 서비스입니다.</div>
    </div>
    """, unsafe_allow_html=True)

    if raw_df is None:
        st.error("분석 데이터를 준비하지 못했습니다. 데이터 출처(URL)와 원본 CSV의 열 구성을 확인해 주세요.")
        st.exception(data_error)
        return

    dates = pd.to_datetime(clean_df["Date"], errors="coerce").dropna()
    market_count = clean_df["Index"].nunique()
    start_date = dates.min().date() if not dates.empty else None
    end_date = dates.max().date() if not dates.empty else None
    avg_vol = market_features["annualized_volatility"].mean() if not market_features.empty else np.nan

    a, b, c, d = st.columns(4)
    a.metric("분석 대상 시장 수", f"{market_count:,}개")
    b.metric("전처리 후 데이터 행", f"{len(clean_df):,}행")
    c.metric("데이터 수집 기간", f"{start_date} ~ {end_date}" if start_date and end_date else "확인 불가")
    d.metric("평균 연환산 변동성", f"{avg_vol:.1%}" if pd.notna(avg_vol) else "계산 불가")

    st.divider()
    st.subheader("데이터에서 살펴볼 수 있는 네 가지 핵심 질문")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧩 1. 시장 유형은 어떻게 나뉠까?")
        if len(market_features) >= 3:
            st.write(
                f"현재 데이터에서 **{len(market_features)}개 시장**을 평균 수익률, 변동성, 최대 낙폭 등의 "
                "특징에 따라 유사한 시장끼리 군집화하여 비교할 수 있습니다."
            )
        else:
            st.info("군집화에 필요한 시장별 특징값이 충분하지 않습니다.")

        st.markdown("#### 📉 2. 변동성이 가장 크게 나타난 시장은?")
        if not market_features.empty and market_features["annualized_volatility"].notna().any():
            highest = market_features.loc[market_features["annualized_volatility"].idxmax()]
            st.write(
                f"과거 데이터 기준 **{highest['Index']}** 지수의 연환산 변동성이 가장 높게 계산되었습니다 "
                f"(**{highest['annualized_volatility']:.1%}**)."
            )
        else:
            st.info("변동성을 비교할 수 있는 시장 데이터가 부족합니다.")

    with col2:
        st.markdown("#### 🔗 3. 서로 비슷하게 움직이는 시장은?")
        pivot = clean_df.pivot_table(index="Date", columns="Index", values="daily_return_used", aggfunc="mean")
        valid = pivot.columns[pivot.count() >= 60].tolist()
        corr = pivot[valid].corr(min_periods=60) if len(valid) >= 2 else pd.DataFrame()
        if not corr.empty:
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().dropna()
            if not upper.empty:
                pair = upper.abs().idxmax()
                value = corr.loc[pair[0], pair[1]]
                st.write(f"**{pair[0]}**과 **{pair[1]}**의 일별 수익률 상관계수는 **{value:.2f}**입니다.")
                st.caption("※ 상관관계는 두 시장이 함께 움직이는 경향을 보여줄 뿐, 인과관계나 비례관계를 의미하지 않습니다.")
            else:
                st.info("상관관계를 비교할 수 있는 시장 쌍이 없습니다.")
        else:
            st.info("상관관계를 계산할 만큼 겹치는 거래일이 충분한 시장이 부족합니다.")

        st.markdown("#### 🧭 4. 최근 데이터 기록 현황")
        if not clean_df.empty:
            latest_date = clean_df["Date"].max()
            latest_rows = clean_df[clean_df["Date"] == latest_date]
            st.write(
                f"데이터의 가장 최근 기록일은 **{pd.Timestamp(latest_date).date()}**이며, "
                f"해당 날짜에 **{latest_rows['Index'].nunique()}개 시장**의 기록이 포함되어 있습니다."
            )

    st.divider()
    st.subheader("메뉴 안내")
    st.write("- **데이터 탐색 / 데이터 전처리**: 원본 주가지수 데이터의 기간, 결측치, 중복 제거 및 수익률 변환 과정을 확인합니다.")
    st.write("- **시장 유형·관계 분석**: K-Means 군집화와 시장 간 수익률 상관관계 히트맵을 탐색합니다.")
    st.info("왼쪽 사이드바의 **'급락 위험 예측'** 메뉴에서 원하는 모델들을 선택해 기준선과 함께 비교할 수 있습니다.")
    st.warning("본 서비스는 과거 금융시장 데이터를 이해하기 위한 교육·연구용 분석 도구이며, 실제 투자 수익이나 손실 방지를 보장하지 않습니다.")


# 5. 분류 문제용 특징 변수 및 정답 라벨 생성 (데이터 누수 방지)
@st.cache_data(show_spinner=False)
def make_classification_data(data: pd.DataFrame, threshold: float):
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
    valid_df = df.dropna(subset=features + ["next_return"]).copy()
    return valid_df, features


# 6. 개별 모델 생성 헬퍼 함수
def build_selected_model(model_name: str):
    if model_name == "로지스틱 회귀":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ))
        ])
    elif model_name == "랜덤 포레스트":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])
    elif model_name == "히스토그램 기반 부스팅":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.08,
                random_state=42
            ))
        ])
    else:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {model_name}")


# 7. 급락 위험 예측 페이지
def model_evaluation_page():
    st.title("📉 주식시장 급락 위험 예측")
    st.caption(
        "과거 시장의 수익률과 변동성 정보를 바탕으로 다음 거래일의 큰 하락 위험을 분류하고, "
        "원하는 모델들을 조합하여 선택한 뒤 단순 기준선과 비교합니다."
    )

    if raw_df is None:
        st.error("데이터를 준비하지 못해 급락 위험을 분석할 수 없습니다.")
        st.exception(data_error)
        st.stop()

    threshold_pct = st.slider(
        "어느 정도 하락을 '큰 하락(급락)'으로 정의할까요?",
        min_value=-10.0,
        max_value=-1.0,
        value=-2.0,
        step=0.5,
        help="예: -2.0%를 선택하면 다음 거래일 수익률이 -2.0% 이하인 경우를 '큰 하락(1)'으로 분류합니다."
    )
    class_df, class_features = make_classification_data(clean_df, threshold_pct / 100.0)

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
        <b>핵심 연구 질문</b><br>
        과거 세계 금융시장 데이터를 분석하여 시장별 특징과 관계를 파악하고, 
        과거의 수익률 및 변동성 정보를 활용해 다음 거래일의 큰 하락 위험을 분류할 수 있는가?
        </div>
        """, unsafe_allow_html=True)

        st.subheader("왜 '이진 분류' 문제로 접근하나요?")
        st.write(
            "내일의 정확한 주가지수나 소수점 단위의 수익률을 맞추는 것은 매우 어렵습니다. "
            "대신 다음 거래일의 결과를 두 가지 범주로 나누어 위험 신호를 포착하는 분류 문제로 접근합니다."
        )

        left, right = st.columns(2)
        with left:
            st.markdown("#### <span class='badge-drop'>🔴 큰 하락 (위험 신호: 1)</span>", unsafe_allow_html=True)
            st.write(f"다음 거래일 수익률이 **{threshold_pct:.1f}% 이하**로 떨어지는 경우입니다.")
        with right:
            st.markdown("#### <span class='badge-normal'>🟢 큰 하락이 아님 (일반 움직임: 0)</span>", unsafe_allow_html=True)
            st.write(f"다음 거래일 수익률이 **{threshold_pct:.1f}%보다 높은** 경우입니다.")

        st.subheader("비교할 수 있는 분류 모델 3가지")
        model_cols = st.columns(3)
        with model_cols[0]:
            st.markdown("#### 1. 로지스틱 회귀")
            st.write("선형 조합을 통해 급락 확률을 계산하며, 결과를 이해하기 쉽습니다.")
        with model_cols[1]:
            st.markdown("#### 2. 랜덤 포레스트")
            st.write("여러 개의 결정 트리가 투표하여 복잡한 비선형 관계를 학습합니다.")
        with model_cols[2]:
            st.markdown("#### 3. 히스토그램 기반 부스팅")
            st.write("순차적으로 오차를 보완하며 대규모 데이터 패턴을 정교하게 학습합니다.")

        st.info(
            "아래 '예측 결과' 탭에서 **원하는 모델들을 항목별로 자유롭게 선택(1개, 일부, 혹은 전부)**하여 "
            "기준선과 함께 실행하고 비교할 수 있습니다."
        )

    with tab_data:
        st.header("예측 모델에 들어가는 데이터")
        st.write(
            "전처리된 주가지수 데이터에서 시장별로 날짜 순서를 정렬한 뒤, "
            "예측 시점까지 알 수 있는 과거 정보 4가지만 추출하여 입력 변수로 사용합니다."
        )

        dates = pd.to_datetime(clean_df["Date"], errors="coerce").dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("전처리 완료 데이터", f"{len(clean_df):,}행")
        c2.metric("분석 대상 시장 수", f"{clean_df['Index'].nunique():,}개")
        c3.metric("현재 급락 기준", f"{threshold_pct:.1f}%")

        if not dates.empty:
            st.write(f"**전체 데이터 기간:** {dates.min().date()} ~ {dates.max().date()}")

        st.subheader("모델이 참고하는 4가지 과거 정보 (특징 변수)")
        feature_explanations = pd.DataFrame({
            "변수명": class_features,
            "사용 정보": [
                "당일 수익률",
                "최근 5거래일 누적 수익률",
                "최근 20거래일 변동성",
                "최근 20거래일 평균 수익률"
            ],
            "쉬운 설명": [
                "오늘 시장이 어제 대비 얼마나 오르거나 내렸는지",
                "최근 약 1주일 동안의 누적 상승·하락 흐름",
                "최근 약 한 달(20거래일) 동안 주가가 얼마나 크게 출렁였는지(표준편차)",
                "최근 약 한 달 동안 하루 평균 수익률의 방향"
            ]
        })
        st.dataframe(feature_explanations, use_container_width=True, hide_index=True)

        if not class_df.empty:
            counts = class_df["target"].value_counts().reindex([0, 1], fill_value=0)
            total = int(counts.sum())
            a, b, c = st.columns(3)
            a.metric("모델 학습·시험 가능 기록", f"{total:,}건")
            b.metric("큰 하락이 아닌 기록 (0)", f"{int(counts[0]):,}건 ({counts[0] / total:.1%})")
            c.metric("큰 하락 기록 (1)", f"{int(counts[1]):,}건 ({counts[1] / total:1%})")

            st.caption(
                "※ 금융시장 데이터는 '큰 하락'보다 '큰 하락이 아님'이 훨씬 많은 불균형 데이터입니다. "
                "따라서 단순 정확도뿐 아니라 실제 급락을 찾아내는 재현율(Recall)과 F1 점수를 함께 확인해야 합니다."
            )
            with st.expander("모델 입력 데이터 미리보기 (상위 30행)"):
                preview = class_df[["Index", "Date"] + class_features + ["next_return", "target"]].head(30).copy()
                preview = preview.rename(columns={
                    "Index": "시장",
                    "Date": "날짜",
                    "return_today": "당일 수익률",
                    "return_5d": "최근 5일 누적 수익률",
                    "volatility_20d": "최근 20일 변동성",
                    "mean_return_20d": "최근 20일 평균 수익률",
                    "next_return": "다음 거래일 수익률(정답용)",
                    "target": "급락 여부(1=예)"
                })
                st.dataframe(preview, use_container_width=True, hide_index=True)
        else:
            st.warning("현재 기준으로 계산할 수 있는 예측 데이터가 부족합니다. 급락 기준을 조정해 주세요.")

    with tab_baseline:
        st.header("비교의 기준점: 단순 기준선 (Baseline)")
        st.write(
            "머신러닝 모델이 낸 점수가 진짜 의미 있는 성과인지 확인하려면, "
            "아무런 시장 패턴을 학습하지 않는 가장 단순한 규칙과 먼저 비교해야 합니다."
        )

        st.subheader("기준선: 가장 자주 나타난 결과만 예측 (`DummyClassifier`)")
        st.markdown("""
        - 학습 데이터에서 가장 많이 등장한 범주(대부분의 경우 **'큰 하락이 아님'**)만 모든 시험 날짜에 동일하게 예측합니다.
        - 최근 수익률이나 변동성 정보를 전혀 사용하지 않습니다.
        - **기준선은 후보 모델들과 별도로 항상 함께 비교되며**, 사용자가 어떤 모델을 선택하든 동일한 시험 데이터에서 나란히 평가됩니다.
        """)

        st.subheader("데이터 누수 방지 및 공정한 평가 원칙")
        st.markdown("""
        1. **시간 순서 분할:** 무작위로 섞지 않고 **과거 날짜 약 80%를 학습용**, **이후 날짜 약 20%를 시험용**으로 나눕니다.
        2. **날짜 단위 경계:** 동일한 날짜의 여러 시장 데이터가 학습과 시험에 동시에 섞이지 않도록 기준 날짜를 중심으로 분리합니다.
        3. **전처리 누수 차단:** 결측값 대체(`SimpleImputer`)와 표준화(`StandardScaler`)는 오직 학습 데이터에서만 기준을 계산한 뒤 시험 데이터에 적용합니다.
        4. **미래 정보 차단:** 다음 거래일 수익률은 정답 라벨(`target`)을 만들 때만 쓰며 입력 변수에는 절대 포함하지 않습니다.
        """)

    with tab_eval:
        st.header("모델 다중 선택 및 기준선 비교 결과")
        st.write(
            "아래에서 비교하고 싶은 모델들을 **자유롭게 여러 개 선택**한 뒤 실행 버튼을 누르면, "
            "선택한 모델들만 각각 학습하여 단순 기준선과 함께 성능을 비교합니다."
        )

        if len(class_df) < 100 or class_df["target"].nunique() < 2:
            st.warning(
                "모델을 학습하려면 유효한 기록이 100건 이상이고, "
                "큰 하락(1)과 큰 하락 아님(0) 사례가 모두 존재해야 합니다."
            )
            return

        unique_dates = np.array(sorted(class_df["Date"].unique()))
        split_pos = int(len(unique_dates) * 0.8)

        if split_pos <= 0 or split_pos >= len(unique_dates):
            st.warning("날짜 순서에 따른 학습·시험 데이터 분할이 어렵습니다.")
            return

        split_date = unique_dates[split_pos]
        train = class_df[class_df["Date"] < split_date].copy()
        test = class_df[class_df["Date"] >= split_date].copy()

        X_train = train[class_features]
        y_train = train["target"]
        X_test = test[class_features]
        y_test = test["target"]

        if test.empty or y_train.nunique() < 2 or y_test.nunique() < 2:
            st.warning(
                "학습 기간 또는 시험 기간 중 한쪽에 급락 사례가 전혀 없습니다. "
                "상단의 급락 기준 슬라이더를 조정해 주세요."
            )
            return

        train_start = pd.Timestamp(train["Date"].min()).date()
        train_end = pd.Timestamp(train["Date"].max()).date()
        test_start = pd.Timestamp(test["Date"].min()).date()
        test_end = pd.Timestamp(test["Date"].max()).date()

        st.info(
            f"**데이터 분할 기준:** 학습 기간 `{train_start} ~ {train_end}` ({len(train):,}건) | "
            f"시험 기간 `{test_start} ~ {test_end}` ({len(test):,}건)"
        )

        # 다중 선택 UI (st.multiselect) 적용
        all_candidate_models = ["로지스틱 회귀", "랜덤 포레스트", "히스토그램 기반 부스팅"]
        
        sel_col, btn_col = st.columns([3, 1])
        with sel_col:
            selected_models_list = st.multiselect(
                "비교할 모델을 선택하세요 (하나, 일부, 또는 모두 선택 가능)",
                options=all_candidate_models,
                default=all_candidate_models, # 기본값은 전체 선택
                help="원하는 모델을 추가하거나 제거하여 동시에 비교할 수 있습니다."
            )
        with btn_col:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            run_clicked = st.button("🚀 선택한 모델들 실행", type="primary", use_container_width=True)

        if not selected_models_list:
            st.warning("비교할 모델을 최소한 하나 이상 선택해 주세요.")
            return

        current_run_key = (tuple(sorted(selected_models_list)), threshold_pct)
        if run_clicked:
            st.session_state["last_eval_key"] = current_run_key
            st.session_state["should_run_eval"] = True

        if st.session_state.get("last_eval_key") != current_run_key:
            st.info(
                f"현재 선택된 모델 목록: **{', '.join(selected_models_list)}** (급락 기준 **{threshold_pct:.1f}%**)입니다. "
                "오른쪽의 **'선택한 모델들 실행'** 버튼을 누르면 학습과 평가가 시작됩니다."
            )
            return

        # 기준선 + 선택된 복수의 모델 학습 및 평가
        try:
            with st.spinner("선택된 모델들과 기준선을 학습·평가하는 중..."):
                baseline_model = DummyClassifier(strategy="most_frequent")
                baseline_model.fit(X_train, y_train)
                base_pred = baseline_model.predict(X_test)
                base_proba = baseline_model.predict_proba(X_test)[:, 1] if hasattr(baseline_model, "predict_proba") else None

                def calc_metrics(name, y_true, y_pred, y_prob):
                    auc_val = roc_auc_score(y_true, y_prob) if (y_prob is not None and len(np.unique(y_true)) == 2) else np.nan
                    return {
                        "구분": name,
                        "정확도": accuracy_score(y_true, y_pred),
                        "정밀도": precision_score(y_true, y_pred, zero_division=0),
                        "재현율": recall_score(y_true, y_pred, zero_division=0),
                        "F1 점수": f1_score(y_true, y_pred, zero_division=0),
                        "ROC-AUC": auc_val
                    }

                rows = [calc_metrics("기준선 (가장 많은 범주 예측)", y_test, base_pred, base_proba)]
                model_predictions = {}

                for m_name in selected_models_list:
                    model_pipe = build_selected_model(m_name)
                    model_pipe.fit(X_train, y_train)
                    m_pred = model_pipe.predict(X_test)
                    m_proba = model_pipe.predict_proba(X_test)[:, 1] if hasattr(model_pipe, "predict_proba") else None

                    rows.append(calc_metrics(m_name, y_test, m_pred, m_proba))
                    model_predictions[m_name] = (model_pipe, m_pred)

                result_df = pd.DataFrame(rows)

        except Exception as err:
            st.error(f"모델 학습 또는 평가 중 오류가 발생했습니다: {err}")
            st.exception(err)
            return

        st.subheader("📊 선택한 모델들과 기준선 성능 비교표")
        display_df = result_df.copy()
        for col in ["정확도", "정밀도", "재현율", "F1 점수"]:
            display_df[col] = display_df[col].map(lambda v: f"{v:.3f}")
        display_df["ROC-AUC"] = display_df["ROC-AUC"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "미지원")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.caption(
            "각 지표는 0~1 사이의 값입니다. 급락 예측에서는 단순 정확도뿐만 아니라 "
            "실제 하락을 찾아내는 재현율과 F1 점수를 함께 비교하는 것이 중요합니다."
        )

        # 후보 모델 중 F1 점수 기준 가장 우수한 모델 선정 및 상세 시각화
        candidate_result = result_df[result_df["구분"] != "기준선 (가장 많은 범주 예측)"]
        if not candidate_result.empty:
            best_row = candidate_result.sort_values(["F1 점수", "재현율"], ascending=[False, False]).iloc[0]
            best_name = best_row["구분"]

            st.subheader(f"🏆 선택된 모델 중 F1 기준 우수 모델: {best_name}")
            
            # 혼동행렬 시각화
            if best_name in model_predictions:
                _, best_pred = model_predictions[best_name]
                cm = confusion_matrix(y_test, best_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()

                cm_col1, cm_col2 = st.columns([1.3, 1])
                with cm_col1:
                    cm_df = pd.DataFrame(
                        cm,
                        index=["실제: 큰 하락 아님 (0)", "실제: 큰 하락 (1)"],
                        columns=["예측: 큰 하락 아님 (0)", "예측: 큰 하락 (1)"]
                    )
                    fig_cm = px.imshow(
                        cm_df,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="Blues",
                        title=f"{best_name} 혼동행렬 (예측 결과 분포)",
                        labels={"x": "모델의 예측", "y": "실제 결과", "color": "건수"}
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                with cm_col2:
                    st.markdown("#### 혼동행렬 상세 건수")
                    st.write(f"- **정상 통과 (TN):** `{tn:,}건`")
                    st.write(f"- **오경보 (FP):** `{fp:,}건`")
                    st.write(f"- **놓친 위험 (FN):** `{fn:,}건` ⚠️")
                    st.write(f"- **위험 포착 (TP):** `{tp:,}건` ✅")
                    st.caption("실제 급락을 비급락으로 잘못 예측하는 '놓친 위험(FN)'을 줄이는 것이 핵심입니다.")

        # 연구 기록 요약
        with st.expander("📑 학술포스터·연구 기록용 실측 결과 요약 보기"):
            st.caption("현재 화면에 나타난 비교 결과의 요약 정보입니다.")
            research_summary = f"""
[연구 및 실험 실측 기록]
- 연구 주제: 세계 금융시장 데이터 분석을 통한 주식시장 급락 위험 분류 모델 개발
- 데이터 기간: {train_start} ~ {test_end} (전체 유효 기록: {len(class_df):,}건, 시장 수: {clean_df['Index'].nunique()}개)
- 학습 / 시험 분할: 학습 {train_start} ~ {train_end} ({len(train):,}건) / 시험 {test_start} ~ {test_end} ({len(test):,}건)
- 급락 기준: 다음 거래일 수익률 {threshold_pct:.1f}% 이하
- 입력 특징 변수: {', '.join(class_features)}
- 비교 대상 모델 목록: {', '.join(selected_models_list)}
            """.strip()
            st.code(research_summary, language="text")

        st.download_button(
            label="📥 다중 모델 비교 결과 CSV 내려받기",
            data=result_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="crash_risk_multi_model_comparison.csv",
            mime="text/csv"
        )

        st.divider()
        st.subheader("평가 지표 읽는 법")
        st.markdown("""
        - **정확도 (Accuracy):** 전체 예측 중 맞힌 비율
        - **정밀도 (Precision):** 급락이라고 예측한 날 중 실제로 급락이었던 비율
        - **재현율 (Recall):** 실제 급락 중 모델이 찾아낸 비율
        - **F1 점수 (F1-Score):** 정밀도와 재현율의 조화평균
        - **ROC-AUC:** 급락과 비급락을 구별하는 전반적인 변별력 능력
        """)

    st.divider()
    st.caption(
        "※ 연구 한계 및 주의사항: 과거 금융시장 데이터는 미래의 급락을 확실히 예측하지 못하며, "
        "시장별 거래일·휴장일 차이와 데이터 불균형의 영향을 받습니다. 본 앱은 교육 및 연구 목적으로 제작되었습니다."
    )


# 8. 기존 페이지 파일 검증 및 내비게이션 구성
if not PAGES_DIR.exists():
    st.error("pages 폴더를 찾을 수 없습니다. 폴더 구조를 확인해 주세요.")
    st.stop()

page_specs = [
    ("1_탐색.py", "데이터 탐색", "🔍"),
    ("2_데이터_전처리.py", "데이터 전처리", "🧹"),
    ("3_모델_학습_결과.py", "시장 유형 분석", "🧩")
]
for filename, _, _ in page_specs:
    if not (PAGES_DIR / filename).is_file():
        st.error(f"필수 페이지 파일이 없습니다: pages/{filename}")
        st.stop()

home = st.Page(home_page, title="홈", icon="🏠", default=True)
explore = st.Page(str(PAGES_DIR / "1_탐색.py"), title="데이터 탐색", icon="🔍")
preprocess = st.Page(str(PAGES_DIR / "2_데이터_전처리.py"), title="데이터 전처리", icon="🧹")
existing_analysis = st.Page(str(PESSIONS_DIR := PAGES_DIR / "3_모델_학습_결과.py"), title="시장 유형·관계 분석", icon="📊") if 'PESSIONS_DIR' else st.Page(str(PAGES_DIR / "3_모델_학습_결과.py"), title="시장 유형·관계 분석", icon="📊")
evaluation = st.Page(model_evaluation_page, title="급락 위험 예측", icon="📉")

pg = st.navigation(
    {
        "서비스": [home, explore, preprocess],
        "금융시장 분석": [existing_analysis],
        "위험 분석": [evaluation]
    },
    position="sidebar"
)

with st.sidebar:
    st.divider()
    st.caption("세계 금융시장 분석 · 교육 및 연구용")

pg.run()
