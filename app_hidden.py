# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# ==============================
# 페이지/레이아웃 설정 (모바일 최적화)
# ==============================
st.set_page_config(
    page_title="영세율 판별 검색 도구",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 글로벌 CSS (폰트/버튼/입력창 크게)
st.markdown("""
<style>
html, body, [class*="css"] { font-size: 18px !important; }
.dataframe td, .dataframe th { font-size: 16px !important; }
.stButton > button { width: 100%; height: 3em; font-size: 18px; }
input[type=text], textarea, .stTextInput input { font-size: 18px !important; }
</style>
""", unsafe_allow_html=True)

st.title("영세율 판별 검색 도구")
st.caption("키워드를 입력하면 해당 품목과 분류(사후환급신청 / 영세율TI 수취)를 찾아줍니다.")

# ==============================
# 데이터 로딩 유틸 (UI 미노출 - 자동 로드)
# ==============================
RAW_URL = "https://raw.githubusercontent.com/yasci78-hue/20251020-python-income-tax/main/%EC%98%81%EC%84%B8%EC%9C%A8%ED%8C%90%EB%B3%84.xlsx"

@st.cache_data(show_spinner=False)
def load_excel_from_raw_url(raw_url: str):
    resp = requests.get(raw_url, timeout=30)
    resp.raise_for_status()
    # openpyxl 엔진은 설치가 필요합니다 (requirements.txt에 openpyxl 추가)
    return pd.read_excel(BytesIO(resp.content), engine="openpyxl")

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    # 품목 열 추정
    item_col = "Unnamed: 1" if "Unnamed: 1" in cols else None
    if not item_col:
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
        raise ValueError("품목 열을 찾지 못했습니다. (예상 열: 'Unnamed: 1' 또는 텍스트가 많은 열)")
    if "구분" not in df.columns:
        df["구분"] = ""

    df["품목"] = df["품목"].astype(str).str.strip()
    df["구분"] = df["구분"].astype(str).str.strip()
    return df[["품목", "구분"]]

# ==============================
# 데이터 로드 (자동)
# ==============================
with st.spinner("데이터 로드 중..."):
    try:
        df = load_excel_from_raw_url(RAW_URL)
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        st.stop()

# 표준화
try:
    df = normalize_df(df)
except Exception as e:
    st.error(f"데이터 표준화 실패: {e}")
    st.stop()

# ==============================
# 검색 UI
# ==============================
query = st.text_input("검색어 입력 (쉼표로 여러 개, 예: 소독기, 필름, 펌프)", value="")
mode_and_or = st.radio("검색 방식", ["AND", "OR"], index=0, horizontal=True)
case_sensitive = st.toggle("대소문자 구분", value=False)

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

# ==============================
# 결과 표시
# ==============================
st.subheader("검색 결과")
st.dataframe(results, use_container_width=True, height=360)

with st.expander("분류별 개수 보기"):
    st.dataframe(results["구분"].value_counts(dropna=False).rename("건수").to_frame(), use_container_width=True)

# 다운로드 버튼
def to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="검색 결과 CSV 다운로드",
    data=to_csv_bytes(results),
    file_name="검색결과.csv",
    mime="text/csv"
)

st.caption("데이터는 고정된 GitHub raw URL에서 자동 로드됩니다.")