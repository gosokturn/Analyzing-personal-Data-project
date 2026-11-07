
import streamlit as st
import pandas as pd
import numpy as np


DATA_URL = (
    "https://raw.githubusercontent.com/"
    "gosokturn/Analyzing-personal-Data-project/"
    "refs/heads/main/data/indexData.csv"
)


@st.cache_data(show_spinner=False)
def load_raw_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def prepare_market_data(raw_df, winsorize=False):
    """
    금융 지수 데이터를 정리하고 일별 수익률을 계산한다.

    winsorize=False:
        극단적인 수익률을 그대로 유지한다.

    winsorize=True:
        각 시장의 수익률 1%, 99% 분위수 경계로 조정한다.
    """

    df = raw_df.copy()

    required = ["Index", "Date"]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"필수 열이 없습니다: {col}"
            )

    # 지수 이름과 날짜 정리
    df["Index"] = df["Index"].astype("string").str.strip()
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # 숫자형 열 변환
    numeric_columns = [
        "Open", "High", "Low",
        "Close", "Adj Close", "Volume"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # 사용할 종가 선택
    if (
        "Adj Close" in df.columns
        and df["Adj Close"].notna().any()
    ):
        price_col = "Adj Close"
    elif "Close" in df.columns:
        price_col = "Close"
    else:
        raise ValueError(
            "Adj Close 또는 Close 열이 필요합니다."
        )

    original_rows = len(df)

    # 날짜, 지수명, 종가가 없는 행 제거
    df = df.dropna(
        subset=["Index", "Date", price_col]
    ).copy()

    # 0 이하의 가격은 수익률 계산에 사용할 수 없음
    df = df[df[price_col] > 0].copy()

    after_valid_rows = len(df)

    # 지수와 날짜가 같은 중복 기록 제거
    before_duplicate = len(df)

    df = df.drop_duplicates(
        subset=["Index", "Date"],
        keep="last"
    ).copy()

    duplicate_removed = before_duplicate - len(df)

    # 시장별 날짜 정렬
    df = df.sort_values(
        ["Index", "Date"]
    ).reset_index(drop=True)

    # 시장별 전일 대비 수익률
    df["daily_return"] = (
        df.groupby("Index")[price_col]
        .pct_change(fill_method=None)
    )

    df["daily_return_used"] = df["daily_return"]

    # 극단값 완화는 선택 사항
    if winsorize:

        def clip_market_returns(series):
            valid = series.dropna()

            if len(valid) < 60:
                return series

            lower = valid.quantile(0.01)
            upper = valid.quantile(0.99)

            return series.clip(
                lower=lower,
                upper=upper
            )

        df["daily_return_used"] = (
            df.groupby("Index")["daily_return"]
            .transform(clip_market_returns)
        )

    audit = {
        "original_rows": original_rows,
        "after_valid_rows": after_valid_rows,
        "duplicate_removed": duplicate_removed,
        "final_rows": len(df),
        "invalid_removed": original_rows - after_valid_rows
    }

    return df, price_col, audit


def build_market_features(
    clean_df,
    price_col,
    min_observations=60
):
    """
    시장별 수익률, 변동성, 하락 위험 등의 특징을 계산한다.
    """

    records = []

    for market, group in clean_df.groupby("Index"):

        group = group.sort_values("Date").copy()

        returns = (
            group["daily_return_used"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(returns) < min_observations:
            continue

        prices = group[price_col].dropna()

        if len(prices) < 2:
            continue

        # 연환산 평균 수익률의 근사치
        annualized_return = returns.mean() * 252

        # 연환산 변동성
        annualized_volatility = (
            returns.std() * np.sqrt(252)
        )

        # 하락일 변동성
        negative_returns = returns[returns < 0]

        if len(negative_returns) >= 2:
            downside_volatility = (
                negative_returns.std() * np.sqrt(252)
            )
        else:
            downside_volatility = np.nan

        # 최대 낙폭
        running_max = prices.cummax()
        drawdown = prices / running_max - 1
        max_drawdown = drawdown.min()

        # 상승한 거래일의 비율
        positive_day_ratio = (returns > 0).mean()

        # 하루 평균 절대 수익률
        average_absolute_return = returns.abs().mean()

        # 고가와 저가가 모두 있는 경우 일중 변동 폭
        average_intraday_range = np.nan

        if {"High", "Low"}.issubset(group.columns):

            previous_close = group[price_col].shift(1)

            intraday_range = (
                (group["High"] - group["Low"])
                / previous_close
            )

            intraday_range = intraday_range.replace(
                [np.inf, -np.inf],
                np.nan
            ).dropna()

            if not intraday_range.empty:
                average_intraday_range = intraday_range.mean()

        records.append({
            "Index": market,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "downside_volatility": downside_volatility,
            "max_drawdown": max_drawdown,
            "positive_day_ratio": positive_day_ratio,
            "average_absolute_return": average_absolute_return,
            "average_intraday_range": average_intraday_range,
            "observations": len(returns)
        })

    return pd.DataFrame(records)
