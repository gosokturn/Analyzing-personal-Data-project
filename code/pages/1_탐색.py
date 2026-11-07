import streamlit as st
import pandas as pd
import plotly.express as px

from market_utils import load_raw_data, prepare_market_data

st.title("🔍 데이터 탐색")
st.caption("어떤 금융시장과 기간의 기록이 있는지 살펴보고, 선택한 시장의 흐름을 확인합니다.")

try:
    raw = load_raw_data()
except Exception as e:
    st.error("원본 데이터를 불러오지 못했습니다. 잠시 후 다시 시도하거나 데이터 주소를 확인해 주세요.")
    with st.expander("오류 세부 정보"):
        st.exception(e)
    st.stop()

df = raw.copy()
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
date_values = df["Date"].dropna() if "Date" in df.columns else pd.Series(dtype="datetime64[ns]")
market_count = df["Index"].nunique() if "Index" in df.columns else 0

st.subheader("데이터 한눈에 보기")
c1, c2, c3, c4 = st.columns(4)
c1.metric("포함된 금융시장", f"{market_count:,}개")
c2.metric("전체 관측 행", f"{len(df):,}개")
c3.metric("전체 결측 셀", f"{int(df.isna().sum().sum()):,}개")
c4.metric("데이터 기간", f"{date_values.min().date()} ~ {date_values.max().date()}" if not date_values.empty else "확인 불가")

if "Index" not in df.columns:
    st.error("원본 데이터에 Index 열이 없어 시장별 탐색을 진행할 수 없습니다.")
    st.stop()

st.divider()
st.subheader("선택한 시장의 주가지수 흐름")
st.write("시장을 선택하고 날짜 범위를 조정하면 해당 기간의 종가 변화를 확인할 수 있습니다.")

markets = sorted(df["Index"].dropna().astype(str).unique().tolist())
selected = st.selectbox("확인할 금융시장", markets)
market_df = df[df["Index"].astype(str) == selected].copy()

if "Date" not in market_df.columns:
    st.warning("Date 열이 없어 시간 흐름 그래프를 표시할 수 없습니다.")
else:
    market_df["Date"] = pd.to_datetime(market_df["Date"], errors="coerce")
    value_col = "Adj Close" if "Adj Close" in market_df.columns and pd.to_numeric(market_df["Adj Close"], errors="coerce").notna().any() else "Close"
    if value_col not in market_df.columns:
        st.warning("Adj Close 또는 Close 열을 찾을 수 없어 주가지수 그래프를 표시할 수 없습니다.")
    else:
        market_df[value_col] = pd.to_numeric(market_df[value_col], errors="coerce")
        market_df = market_df.dropna(subset=["Date", value_col]).sort_values("Date")
        if market_df.empty:
            st.info("선택한 시장에는 표시할 수 있는 날짜와 종가가 없습니다.")
        else:
            min_date = market_df["Date"].min().date()
            max_date = market_df["Date"].max().date()
            date_range = st.date_input(
                "조회 기간",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                start, end = date_range
                chart_df = market_df[
                    (market_df["Date"].dt.date >= start) &
                    (market_df["Date"].dt.date <= end)
                ]
            else:
                chart_df = market_df
            st.caption(f"표시 기간: {chart_df['Date'].min().date() if not chart_df.empty else '자료 없음'} ~ {chart_df['Date'].max().date() if not chart_df.empty else '자료 없음'} · 사용 열: {value_col}")
            if not chart_df.empty:
                fig = px.line(
                    chart_df, x="Date", y=value_col,
                    title=f"{selected} 주가지수의 시간에 따른 변화",
                    labels={"Date": "날짜", value_col: "지수 값"},
                    hover_data={value_col: ":,.2f"}
                )
                fig.update_layout(hovermode="x unified", margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.info("그래프의 높고 낮음은 지수의 숫자 수준을 보여줍니다. 서로 다른 시장은 기준점이 다를 수 있으므로 시장 간 성과 비교에는 가격보다 수익률을 사용하는 편이 적절합니다.")
            else:
                st.info("선택한 기간에 표시할 데이터가 없습니다.")

with st.expander("원본 데이터 표 보기"):
    st.caption("아래 표는 전처리 전 원본 데이터입니다. 결측값이나 중복 기록이 포함될 수 있습니다.")
    st.dataframe(raw.head(300), use_container_width=True, hide_index=True)
    st.download_button(
        "📥 원본 데이터 CSV 다운로드",
        data=raw.to_csv(index=False).encode("utf-8-sig"),
        file_name="indexData_original.csv",
        mime="text/csv"
    )

with st.expander("열별 데이터 유형과 결측 현황"):
    schema = pd.DataFrame({
        "열 이름": raw.columns,
        "데이터 유형": [str(raw[col].dtype) for col in raw.columns],
        "결측 개수": [int(raw[col].isna().sum()) for col in raw.columns],
        "결측 비율": [raw[col].isna().mean() for col in raw.columns]
    })
    schema["결측 비율"] = schema["결측 비율"].map(lambda x: f"{x:.1%}")
    st.dataframe(schema, use_container_width=True, hide_index=True)

with st.expander("시장별 데이터 기록 수"):
    counts = raw["Index"].value_counts().rename_axis("금융시장").reset_index(name="관측 행 수")
    fig_counts = px.bar(counts, x="금융시장", y="관측 행 수", title="시장별 원본 기록 수")
    st.plotly_chart(fig_counts, use_container_width=True)
    st.dataframe(counts, use_container_width=True, hide_index=True)
