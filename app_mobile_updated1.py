# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# ==============================
# 페이지/레이아웃 설정 (모바일 최적화)
# ==============================
st.set_page_config(
    page_title="농업 등 기자재 영세율 판별",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 글로벌 CSS (폰트/버튼/입력창 크게 + 검색창 테두리/포커스 강조)
st.markdown("""
<style>
html, body, [class*="css"] { font-size: 16px !important; }
.dataframe td, .dataframe th { font-size: 14px !important; }
.stButton > button { width: 100%; height: 3em; font-size: 16px; }
input[type=text], textarea, .stTextInput input { font-size: 16px !important; }

/* 검색창 시인성 강화 */
.stTextInput input {
    border: 2px solid #1f6feb !important;   /* 진한 테두리 */
    border-radius: 10px !important;         /* 둥근 모서리 */
    padding: 10px 12px !important;          /* 넉넉한 패딩 */
    box-shadow: none !important;            /* 기본 값 초기화 */
}
.stTextInput input:focus {
    outline: none !important;
    border-color: #0d419d !important;       /* 포커스 색 */
    box-shadow: 0 0 0 3px rgba(31, 111, 235, 0.25) !important; /* 포커스 글로우 */
}
/* placeholder 색 살짝 진하게 */
.stTextInput input::placeholder {
    color: #5a6c7d !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

st.caption("별표 글씨 안보이시지예? 그냥 여기 검색하이소!")
st.subheader("노안의 그대를 위하여!!")
st.title("영세율 판별 도구")
st.caption("키워드 입력시 농업등 기자재의 [사후환급 / 영세율TI 수취] 대상이 분류됩니다.")

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
# 검색 UI (Enter로 제출 + '검색' 버튼 추가)
# ==============================
# 기존 선택값 유지용 기본값 설정
default_query = st.session_state.get("query", "")
default_mode = st.session_state.get("mode_and_or", "AND")
default_case = st.session_state.get("case_sensitive", False)

   with col_opts1:
        mode_and_or = st.radio("검색 방식", ["AND", "OR"], index=0 if default_mode == "AND" else 1, horizontal=True)
    with col_opts2:
        case_sensitive = st.toggle("대소문자 구분", value=default_case)

with st.form("search_form", clear_on_submit=False):
    # 입력과 옵션들을 하나의 form으로 묶어서 Enter로 제출되게 함
    query = st.text_input(
        "검색어 입력 (쉼표로 여러 개, 예: 소독기, 필름, 펌프)",
        value=default_query,
        placeholder="예: 비닐, 관수, 펌프"
    )
    col_opts1, col_opts2 = st.columns([1, 1], vertical_alignment="center")
 
    submitted = st.form_submit_button("검색")

# 폼 제출 시 상태 저장 (버튼 클릭 또는 Enter)
if submitted:
    st.session_state["query"] = query
    st.session_state["mode_and_or"] = mode_and_or
    st.session_state["case_sensitive"] = case_sensitive

# 상태값 불러오기 (초기 렌더링 또는 제출 후)
query = st.session_state.get("query", default_query)
mode_and_or = st.session_state.get("mode_and_or", default_mode)
case_sensitive = st.session_state.get("case_sensitive", default_case)

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
st.subheader("포항세무서 직원 아님.")
