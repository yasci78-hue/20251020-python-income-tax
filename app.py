import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="영세율 판별 검색 도구", layout="wide")

st.title("영세율 판별 검색 도구")
st.caption("키워드를 입력하면 해당 품목과 분류(사후환급신청 / 영세율TI 수취)를 찾아줍니다.")

# --- 파일 로드 영역 ---
default_path = "/mnt/data/영세율판별.xlsx"  # 업로드한 기본 경로
uploaded = st.file_uploader("엑셀 파일 업로드 (.xlsx)", type=["xlsx"])
path_info = st.empty()

@st.cache_data(show_spinner=False)
def load_excel(fileobj_or_path):
    # 파일 경로 또는 업로드 파일 모두 지원
    if isinstance(fileobj_or_path, str):
        df = pd.read_excel(fileobj_or_path, sheet_name=0)
    else:
        df = pd.read_excel(fileobj_or_path, sheet_name=0)

    # 열 정리: 우리가 확인한 구조에 맞춰 표준화
    cols = list(df.columns)
    # 품목 후보 열 찾기 (기본: 'Unnamed: 1')
    item_col = None
    if "Unnamed: 1" in cols:
        item_col = "Unnamed: 1"
    else:
        # 휴리스틱: 문자열 길이가 긴 object 열을 품목으로 가정
        object_cols = [c for c in cols if df[c].dtype == 'object']
        if object_cols:
            # 평균 문자열 길이 가장 긴 열 선택
            avg_len = {c: df[c].dropna().astype(str).str.len().mean() for c in object_cols}
            item_col = max(avg_len, key=avg_len.get)

    # 구분 열 찾기 (기본: '구분')
    kind_col = "구분" if "구분" in cols else None

    # 표준 컬럼명으로 리네임
    rename_map = {}
    if item_col and item_col != "품목":
        rename_map[item_col] = "품목"
    if kind_col and kind_col != "구분":
        rename_map[kind_col] = "구분"

    df = df.rename(columns=rename_map)

    # 최소 컬럼 유효성 확인
    if "품목" not in df.columns:
        raise ValueError("품목 열을 찾지 못했습니다. 엑셀에서 품목이 있는 열을 'Unnamed: 1' 또는 텍스트가 대부분인 열로 맞춰주세요.")
    if "구분" not in df.columns:
        # 구분이 없으면 빈 값으로 생성(옵션)
        df["구분"] = ""

    # 정리
    df["품목"] = df["품목"].astype(str).str.strip()
    df["구분"] = df["구분"].astype(str).str.strip()
    return df[["품목", "구분"]]

# 파일 선택
if uploaded is not None:
    df = load_excel(uploaded)
    path_info.info("업로드한 파일을 사용 중입니다.")
else:
    # 기본 경로 사용 (로컬/서버 환경에 이 경로가 없을 수도 있으니 예외처리)
    try:
        df = load_excel(default_path)
        path_info.info(f"기본 파일 사용 중: {default_path}")
    except Exception as e:
        st.warning("기본 경로의 파일을 불러올 수 없습니다. 상단에서 엑셀을 업로드 해주세요.")
        st.stop()

# --- 검색 UI ---
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("검색어 입력 (예: 소독기, 필름, 펌프 ...)", value="")
    with col2:
        case_sensitive = st.checkbox("대소문자 구분", value=False)

# --- 검색 로직 ---
def search(df, q, case_sensitive=False):
    if not q:
        return df.copy()
    # 쉼표로 여러 키워드 지원: "소독기, 필름"
    tokens = [t.strip() for t in q.split(",") if t.strip()]
    if not tokens:
        return df.copy()

    # contains 조건을 모두 AND로 묶기 (필요 시 OR 로직으로 변경 가능)
    res = df.copy()
    for t in tokens:
        if case_sensitive:
            res = res[res["품목"].str.contains(t, na=False)]
        else:
            res = res[res["품목"].str.contains(t, case=False, na=False)]
    return res

results = search(df, query, case_sensitive=case_sensitive)

# --- 요약 ---
left, right = st.columns([2, 1])
with right:
    st.subheader("분류별 개수")
    if "구분" in results.columns:
        st.dataframe(results["구분"].value_counts(dropna=False).rename("건수").to_frame())

with left:
    st.subheader("검색 결과")
    st.caption("품목을 클릭해 전체 텍스트를 확인하세요.")
    st.dataframe(results, use_container_width=True, height=480)

# --- 다운로드 ---
def to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")

csv_bytes = to_csv_bytes(results)
st.download_button(
    label="검색 결과 CSV 다운로드",
    data=csv_bytes,
    file_name="검색결과.csv",
    mime="text/csv",
)

# --- 도움말/팁 ---
with st.expander("검색 팁"):
    st.markdown("""
- 여러 키워드는 쉼표(,)로 구분해서 입력하세요. 예: `소독기, 양식장`
- 현재는 **품목 내 포함 검색**(부분 일치) 기준입니다.
- 기본 로직은 **AND 검색**으로, 입력한 모든 키워드를 포함하는 항목만 보여줍니다.
    - OR 검색이 필요하면, 키워드 하나씩 검색하거나 코드의 `search` 함수를 간단히 수정하세요.
""")
