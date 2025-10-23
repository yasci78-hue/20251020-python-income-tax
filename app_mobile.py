# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests, base64
from io import BytesIO

# ==============================
# 페이지/레이아웃 설정 (모바일 최적화)
# ==============================
st.set_page_config(
    page_title="영세율 판별 검색 도구",
    layout="centered",                 # ✅ 모바일에서도 보기 편한 폭
    initial_sidebar_state="collapsed"  # ✅ 사이드바 기본 접기
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
# 데이터 로딩 유틸
# ==============================
@st.cache_data(show_spinner=False)
def load_excel_from_raw_url(raw_url: str):
    resp = requests.get(raw_url, timeout=30)
    resp.raise_for_status()
    return pd.read_excel(BytesIO(resp.content), engine="openpyxl")

@st.cache_data(show_spinner=False)
def load_excel_local(fileobj_or_path):
    try:
        return pd.read_excel(fileobj_or_path, sheet_name=0, engine="openpyxl")
    except ImportError:
        st.error("엑셀(.xlsx) 읽기에는 openpyxl이 필요합니다. requirements.txt에 openpyxl을 추가하세요.")
        raise

@st.cache_data(show_spinner=False)
def load_excel_from_private_repo(owner:str, repo:str, path:str, ref:str="main"):
    token = st.secrets.get("github", {}).get("token")
    if not token:
        raise RuntimeError("secrets에 [github][token]이 없습니다. App settings → Secrets에 token을 추가하세요.")
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    content = base64.b64decode(r.json()["content"])
    return pd.read_excel(BytesIO(content), engine="openpyxl")

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
# 데이터 소스 선택 (모바일 친화: selectbox)
# ==============================
source = st.selectbox(
    "데이터 불러오기 방식",
    ["GitHub raw URL (공개)", "로컬 업로드", "GitHub API (비공개)"],
    index=0
)

df = None

if source == "GitHub raw URL (공개)":
    raw_url = st.text_input("raw.githubusercontent.com 링크", value="https://raw.githubusercontent.com/yasci78-hue/20251020-python-income-tax/main/%EC%98%81%EC%84%B8%EC%9C%A8%ED%8C%90%EB%B3%84.xlsx")
    if raw_url:
        with st.spinner("GitHub에서 데이터 로드 중..."):
            try:
                df = load_excel_from_raw_url(raw_url)
                st.success("GitHub raw URL에서 데이터 불러오기 성공!")
            except Exception as e:
                st.error(f"데이터 로드 실패: {e}")

elif source == "로컬 업로드":
    uploaded = st.file_uploader("엑셀 파일 업로드 (.xlsx)", type=["xlsx"])
    if uploaded is not None:
        with st.spinner("파일 로드 중..."):
            try:
                df = load_excel_local(uploaded)
                st.success("업로드한 파일을 사용 중입니다.")
            except Exception as e:
                st.error(f"엑셀 읽기 실패: {e}")

else:  # GitHub API (비공개)
    owner = st.text_input("Owner", value="")
    repo  = st.text_input("Repo", value="")
    path  = st.text_input("Path (예: data/영세율판별.xlsx)", value="영세율판별.xlsx")
    ref   = st.text_input("Branch/Tag", value="main")
    if owner and repo and path and st.button("불러오기"):
        with st.spinner("비공개 저장소에서 데이터 로드 중..."):
            try:
                df = load_excel_from_private_repo(owner, repo, path, ref)
                st.success("GitHub 비공개 저장소(API)에서 로드 완료")
            except Exception as e:
                st.error(f"비공개 저장소 로드 실패: {e}")

# 데이터가 없으면 중단
if df is None:
    st.stop()

# 표준화
try:
    df = normalize_df(df)
except Exception as e:
    st.error(f"데이터 표준화 실패: {e}")
    st.stop()

# ==============================
# 검색 UI (모바일: 한 줄에 하나씩 배치)
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
# 결과 표시 (모바일: 단일 컬럼, 표는 컨테이너 너비 사용)
# ==============================
st.subheader("검색 결과")
st.dataframe(results, use_container_width=True, height=360)

with st.expander("분류별 개수 보기"):
    st.dataframe(results["구분"].value_counts(dropna=False).rename("건수").to_frame(), use_container_width=True)

# 다운로드 버튼 (가로 100%)
def to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="검색 결과 CSV 다운로드",
    data=to_csv_bytes(results),
    file_name="검색결과.csv",
    mime="text/csv"
)

st.caption("데이터 출처(기본): GitHub raw URL - https://raw.githubusercontent.com/yasci78-hue/20251020-python-income-tax/main/%EC%98%81%EC%84%B8%EC%9C%A8%ED%8C%90%EB%B3%84.xlsx")
