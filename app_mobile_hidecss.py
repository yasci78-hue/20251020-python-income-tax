# -*- coding: utf-8 -*-
"""
검색창과 챗봇만 보이게 (강제 숨김 CSS 포함)
- 데이터 불러오기 UI는 코드상 제거 + CSS로도 1차/2차 방어
"""

import os
from io import BytesIO
import requests
import pandas as pd
import streamlit as st

# ==============================
# 전역 CSS (데이터 로딩 UI만 숨기고 검색창은 살리기)
# ==============================
st.markdown(
    """
    <style>
      /* '데이터 불러오기 방식' Selectbox만 숨김 */
      div[data-testid="stSelectbox"] label:contains("데이터 불러오기 방식") {
          display: none !important;
      }
      /* 'raw.githubusercontent.com 링크' TextInput 숨김 */
      div[data-testid="stTextInput"] label:contains("raw.githubusercontent.com 링크") {
          display: none !important;
      }
      /* 'GitHub raw URL에서 데이터 불러오기 성공' 알림 숨김 */
      div[role="alert"]:has(p:contains("GitHub raw URL에서 데이터 불러오기 성공")) {
          display: none !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)


# ==============================
# 기본 설정
# ==============================
st.set_page_config(page_title="영세율 판별 검색 도구", page_icon="🧾", layout="centered")
st.title("영세율 판별 검색 도구")
st.caption("키워드를 입력하면 해당 품목과 분류(사후환급신청/영세율TI 수취)를 찾아줍니다. (데이터 로딩 UI는 숨김 상태입니다)")

# ==============================
# 데이터 자동 로드
# ==============================
@st.cache_data(show_spinner=True)
def load_data():
    url = (
        st.secrets.get("data", {}).get("url")
        or os.environ.get("DATA_URL")
        or "https://raw.githubusercontent.com/yasci78-hue/20251020-python-income-tax/main/%EC%98%81%EC%84%B8%EC%9C%A8%ED%8C%90%EB%B3%84.xlsx"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return pd.read_excel(BytesIO(resp.content), engine="openpyxl")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

# 최소 컬럼 정규화
if "품목" not in df.columns or "구분" not in df.columns:
    cols = list(df.columns)
    if len(cols) >= 2:
        df = df.rename(columns={cols[0]: "품목", cols[1]: "구분"})
    else:
        df["품목"] = df.iloc[:, 0].astype(str)
        df["구분"] = ""

df["품목"] = df["품목"].astype(str).str.strip()
df["구분"] = df["구분"].astype(str).str.strip()
df = df[["품목", "구분"]]

# ==============================
# 검색 UI
# ==============================
st.subheader("🔎 검색창")
query = st.text_input("검색어 입력 (쉼표로 여러 개, 예: 소독기, 펌프, 필름)")
mode_and_or = st.radio("검색 방식", ["AND", "OR"], horizontal=True)
case_sensitive = st.toggle("대소문자 구분", value=False)

def search(df, q, case_sensitive=False, mode="AND"):
    if not q:
        return df.copy()
    tokens = [t.strip() for t in q.split(",") if t.strip()]
    res = df.copy()
    if mode == "AND":
        for t in tokens:
            res = res[res["품목"].str.contains(t, case=case_sensitive, na=False)]
    else:
        mask = False
        for t in tokens:
            m = res["품목"].str.contains(t, case=case_sensitive, na=False)
            mask = m if isinstance(mask, bool) else (mask | m)
        res = res[mask]
    return res

results = search(df, query, case_sensitive=case_sensitive, mode=mode_and_or)

st.dataframe(results, use_container_width=True, height=420)
st.download_button(
    label="검색 결과 CSV 다운로드",
    data=results.to_csv(index=False).encode("utf-8-sig"),
    file_name="검색결과.csv",
    mime="text/csv"
)

# ==============================
# AI 챗봇
# ==============================
st.divider()
st.subheader("🤖 AI 챗봇 (검색 도우미)")

if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "system", "content": "너는 영세율 판별 검색 도우미야. 사용자의 질문에 간결하고 친절하게 한국어로 답해줘. 검색 팁(쉼표 분리, AND/OR, 대소문자 옵션)도 함께 안내해."}
    ]

for msg in st.session_state.chat:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input("무엇이든 물어보세요. (예: '의료용 소독기 검색 팁 알려줘')")
if user_q:
    st.session_state.chat.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    OPENAI_API_KEY = (
        st.secrets.get("openai", {}).get("api_key")
        or os.environ.get("OPENAI_API_KEY")
    )

    if not OPENAI_API_KEY:
        with st.chat_message("assistant"):
            st.warning("OpenAI API 키가 설정되어 있지 않습니다. Streamlit Secrets에 [openai][api_key]를 추가하세요.")
    else:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            streamed = ""
            try:
                with client.responses.stream(
                    model="gpt-5",
                    input=[{"role": m["role"], "content": m["content"]} for m in st.session_state.chat],
                ) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta":
                            streamed += event.delta
                            placeholder.markdown(streamed)
                    final = stream.get_final_response()
                    assistant_text = final.output_text
            except Exception:
                resp = client.chat.completions.create(
                    model="gpt-5",
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.chat],
                    stream=True,
                )
                assistant_text = ""
                for chunk in resp:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
                        assistant_text += delta
                        placeholder.markdown(assistant_text)

            st.session_state.chat.append({"role": "assistant", "content": streamed or assistant_text})
            placeholder.markdown(streamed or assistant_text)

st.caption("💡 팁: 쉼표로 여러 키워드를 입력하고 AND/OR 옵션으로 검색 폭을 조절하세요.")
