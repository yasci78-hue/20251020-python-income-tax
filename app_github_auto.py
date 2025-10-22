# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests, base64
from io import BytesIO

st.set_page_config(page_title="영세율 판별 검색 도구", layout="wide")

st.title("영세율 판별 검색 도구")
st.caption("키워드를 입력하면 해당 품목과 분류(사후환급신청 / 영세율TI 수취)를 찾아줍니다.")

@st.cache_data(show_spinner=False)
def load_excel_from_raw_url(raw_url: str):
    resp = requests.get(raw_url, timeout=30)
    resp.raise_for_status()
    return pd.read_excel(BytesIO(resp.content), engine="openpyxl")

@st.cache_data(show_spinner=False)
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    item_col = None
    if "Unnamed: 1" in cols:
        item_col = "Unnamed: 1"
    else:
        object_cols = [c for c in cols if df[c].dtype == 'object']
        if object_cols:
            avg_len = {c: df[c].dropna().astype(str).str.len().mean() for c in object_cols}
            item_col = max(avg_len, key=avg_len.get)
    kind_col = "구분" if "구분" in cols else None

    rename_map = {}
    if item_col and item_col != "품목":
        rename_map[item_col] = "품목"
    if kind_col and kind_col != "구분":
        rename_map[kind_col] = "구분"

    df = df.rename(columns=rename_map)
    if "품목" not in df.columns:
        raise ValueError("품목 열을 찾지 못했습니다.")
    if "구분" not in df.columns:
        df["구분"] = ""

    df["품목"] = df["품목"].astype(str).str.strip()
    df["구분"] = df["구분"].astype(str).str.strip()
    return df[["품목", "구분"]]

# 데이터 자동 로드
st.info("GitHub에서 자동으로 엑셀 파일을 불러오는 중입니다...")
try:
    df = load_excel_from_raw_url("https://raw.githubusercontent.com/yasci78-hue/20251020-python-income-tax/main/%EC%98%81%EC%84%B8%EC%9C%A8%ED%8C%90%EB%B3%84.xlsx")
    df = normalize_df(df)
    st.success("GitHub raw URL에서 데이터 불러오기 성공!")
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

query = st.text_input("검색어 입력 (예: 소독기, 필름, 펌프 ...)", value="")
case_sensitive = st.checkbox("대소문자 구분", value=False)
mode_and_or = st.radio("검색 방식", ["AND", "OR"], index=0, horizontal=True)

def search(df, q, case_sensitive=False, mode="AND"):
    if not q:
        return df.copy()
    tokens = [t.strip() for t in q.split(",") if t.strip()]
    if not tokens:
        return df.copy()
    res = df.copy()
    if mode == "AND":
        for t in tokens:
            res = res[res["품목"].str.contains(t, case=not case_sensitive, na=False)]
    else:
        mask = False
        for t in tokens:
            m = res["품목"].str.contains(t, case=not case_sensitive, na=False)
            mask = m if isinstance(mask, bool) else (mask | m)
        res = res[mask]
    return res

results = search(df, query, case_sensitive=case_sensitive, mode=mode_and_or)

left, right = st.columns([2, 1])
with right:
    st.subheader("분류별 개수")
    st.dataframe(results["구분"].value_counts(dropna=False).rename("건수").to_frame())

with left:
    st.subheader("검색 결과")
    st.dataframe(results, use_container_width=True, height=480)

def to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="검색 결과 CSV 다운로드",
    data=to_csv_bytes(results),
    file_name="검색결과.csv",
    mime="text/csv",
)

st.caption("데이터 출처: GitHub raw URL - https://raw.githubusercontent.com/yasci78-hue/20251020-python-income-tax/main/%EC%98%81%EC%84%B8%EC%9C%A8%ED%8C%90%EB%B3%84.xlsx")
