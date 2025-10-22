# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests, base64
from io import BytesIO

st.set_page_config(page_title="영세율 판별 검색 도구", layout="wide")

st.title("영세율 판별 검색 도구")
st.caption("키워드를 입력하면 해당 품목과 분류(사후환급신청 / 영세율TI 수취)를 찾아줍니다.")

# ==============================
# 데이터 로드 유틸
# ==============================
@st.cache_data(show_spinner=False)
def load_excel_local(fileobj_or_path):
    """로컬 업로드 파일 또는 경로에서 엑셀 로드(.xlsx)."""
    try:
        df = pd.read_excel(fileobj_or_path, sheet_name=0, engine="openpyxl")
    except ImportError:
        st.error("엑셀(.xlsx) 읽기에는 openpyxl이 필요합니다. requirements.txt에 openpyxl을 추가하세요.")
        raise
    return df

@st.cache_data(show_spinner=False)
def load_excel_from_raw_url(raw_url: str):
    """공개 저장소의 raw.githubusercontent.com 링크에서 엑셀 로드."""
    resp = requests.get(raw_url, timeout=30)
    resp.raise_for_status()
    return pd.read_excel(BytesIO(resp.content), engine="openpyxl")

@st.cache_data(show_spinner=False)
def load_excel_from_private_repo(owner:str, repo:str, path:str, ref:str="main"):
    """비공개 저장소에서 GitHub Contents API + 토큰으로 엑셀 로드.
    Streamlit Secrets에 [github][token]을 넣어주세요.
    """
    token = st.secrets.get("github", {}).get("token")
    if not token:
        raise RuntimeError("secrets에 [github][token]이 없습니다. App settings → Secrets에 token을 추가하세요.")
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    j = r.json()
    content = base64.b64decode(j["content"])
    return pd.read_excel(BytesIO(content), engine="openpyxl")

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """열 이름 표준화: 품목/구분 자동 탐지 및 리네임."""
    cols = list(df.columns)
    # 품목 후보 열 찾기 (기본: 'Unnamed: 1')
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
        raise ValueError("품목 열을 찾지 못했습니다. 엑셀에서 품목이 있는 열을 'Unnamed: 1' 또는 문자열이 많은 열로 맞춰주세요.")
    if "구분" not in df.columns:
        df["구분"] = ""

    df["품목"] = df["품목"].astype(str).str.strip()
    df["구분"] = df["구분"].astype(str).str.strip()
    return df[["품목", "구분"]]

# ==============================
# 입력 수단: 업로드 / 공개 raw URL / 비공개 API
# ==============================
st.subheader("데이터 불러오기")
tab1, tab2, tab3 = st.tabs(["로컬 업로드", "GitHub 공개(raw URL)", "GitHub 비공개(API)"])

df = None

with tab1:
    uploaded = st.file_uploader("엑셀 파일 업로드 (.xlsx)", type=["xlsx"], key="uploader")
    if uploaded is not None:
        try:
            df = load_excel_local(uploaded)
            st.success("업로드한 파일을 사용 중입니다.")
        except Exception as e:
            st.error(f"엑셀 읽기 실패: {e}")

with tab2:
    raw_url = st.text_input("raw.githubusercontent.com 링크를 입력하세요",
                            placeholder="https://raw.githubusercontent.com/<USER>/<REPO>/main/<PATH>/영세율판별.xlsx")
    if raw_url:
        try:
            df = load_excel_from_raw_url(raw_url)
            st.success("GitHub 공개 저장소(raw)에서 로드 완료")
        except Exception as e:
            st.error(f"raw URL 로드 실패: {e}")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        owner = st.text_input("Owner", value="")
        repo  = st.text_input("Repo", value="")
    with col2:
        path  = st.text_input("Path (예: data/영세율판별.xlsx)", value="영세율판별.xlsx")
        ref   = st.text_input("Branch/Tag", value="main")
    if owner and repo and path:
        if st.button("비공개 저장소에서 불러오기"):
            try:
                df = load_excel_from_private_repo(owner, repo, path, ref)
                st.success("GitHub 비공개 저장소(API)에서 로드 완료")
            except Exception as e:
                st.error(f"비공개 저장소 로드 실패: {e}")

# df 없으면 안내 후 중단
if df is None:
    st.info("상단 탭에서 파일을 업로드하거나 GitHub 경로를 입력해 주세요.")
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
with st.container():
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        query = st.text_input("검색어 입력 (예: 소독기, 필름, 펌프 ...)", value="")
    with col2:
        case_sensitive = st.checkbox("대소문자 구분", value=False)
    with col3:
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
    else:  # OR
        mask = False
        for t in tokens:
            m = res["품목"].str.contains(t, case=not case_sensitive, na=False)
            mask = m if isinstance(mask, bool) else (mask | m)
        res = res[mask]
    return res

results = search(df, query, case_sensitive=case_sensitive, mode=mode_and_or)

# ==============================
# 결과 영역
# ==============================
left, right = st.columns([2, 1])
with right:
    st.subheader("분류별 개수")
    st.dataframe(results["구분"].value_counts(dropna=False).rename("건수").to_frame())

with left:
    st.subheader("검색 결과")
    st.caption("품목을 클릭해 전체 텍스트를 확인하세요.")
    st.dataframe(results, use_container_width=True, height=480)

# 다운로드
def to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="검색 결과 CSV 다운로드",
    data=to_csv_bytes(results),
    file_name="검색결과.csv",
    mime="text/csv",
)

with st.expander("검색 팁"):
    st.markdown("""
- 여러 키워드는 쉼표(,)로 구분해서 입력하세요. 예: `소독기, 양식장`
- 기본은 **AND 검색**입니다. OR 검색은 상단 토글에서 선택하세요.
- 공개 저장소는 **raw.githubusercontent.com** 링크를 사용하세요.
- 비공개 저장소는 **App → Settings → Secrets**에 `[github][token]`을 설정하세요.
""")

