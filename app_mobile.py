import streamlit as st
import pandas as pd
import re
from openai import OpenAI
import os

# ----------------------------
# 기본 설정
# ----------------------------
st.set_page_config(page_title="기자재 영세율 AI 도우미", layout="wide")
st.title("🧾 기자재 영세율 판별 AI 도우미")
st.caption("AI에게 물어보세요 — 기자재 영세율 관련 검색, 설명, 판별을 도와드립니다.")

# ----------------------------
# 데이터 불러오기 (자동)
# ----------------------------
DATA_URL = "https://raw.githubusercontent.com/yasci78-hue/20251020-python-income-tax/main/영세율_기자재_DB.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(DATA_URL)

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

# ----------------------------
# 검색 함수
# ----------------------------
def search_dataframe(df, query, case_sensitive=False):
    """키워드 검색"""
    if not query.strip():
        return pd.DataFrame()
    keywords = [k.strip() for k in query.split(",") if k.strip()]
    flags = 0 if case_sensitive else re.IGNORECASE

    mask = df.apply(lambda row: any(
        re.search(k, str(row), flags) for k in keywords
    ), axis=1)
    return df[mask]

# ----------------------------
# OpenAI 클라이언트 설정
# ----------------------------
OPENAI_API_KEY = (
    st.secrets.get("openai", {}).get("api_key")
    or os.environ.get("OPENAI_API_KEY")
)

if not OPENAI_API_KEY:
    st.warning("⚠️ OpenAI API 키가 설정되어 있지 않습니다. Streamlit Secrets에 [openai][api_key] 추가 후 다시 실행하세요.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# 세션 상태 초기화
# ----------------------------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "system", "content": (
            "너는 국세청 기자재 영세율 판별 AI 도우미야. "
            "사용자의 질문이 기자재 검색과 관련되면 df에서 검색해 표로 보여줘. "
            "그리고 표 내용을 분석해 주요 품목, 수입품 여부, 의료기기 관련 항목 등을 요약해서 한국어로 설명해줘. "
            "기타 질문은 일반적인 설명을 제공해."
        )}
    ]

# ----------------------------
# 챗봇 인터페이스
# ----------------------------
for msg in st.session_state.chat_messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_input = st.chat_input("무엇이든 물어보세요. (예: '소독기 영세율 찾아줘')")

if user_input:
    # 사용자 입력 표시
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # “검색 관련” 문장일 경우 내부 검색 실행
    search_terms = ["검색", "찾아", "영세율", "품목", "기자재"]
    if any(word in user_input for word in search_terms):
        query = re.sub("|".join(search_terms), "", user_input)
        results = search_dataframe(df, query)

        with st.chat_message("assistant"):
            if not results.empty:
                st.markdown(f"**🔍 '{query.strip()}' 관련 검색 결과입니다:**")
                st.dataframe(results, use_container_width=True)
                st.download_button(
                    "CSV로 다운로드",
                    results.to_csv(index=False).encode("utf-8-sig"),
                    file_name="검색결과.csv",
                    mime="text/csv"
                )

                # 🔸 데이터 요약 요청 (AI)
                summary_prompt = (
                    f"다음 표 데이터를 분석해서 품목, 분류(의료용, 수입품 등), "
                    f"영세율 관련 특징을 간략히 요약해줘. 표 내용은 다음과 같아:\n\n"
                    f"{results.head(20).to_markdown(index=False)}"
                )

                summary_response = client.responses.create(
                    model="gpt-5-mini",
                    input=summary_prompt
                )
                summary_text = summary_response.output_text
                st.markdown("**📊 AI 요약 분석:**")
                st.info(summary_text)

                ai_reply = f"'{query.strip()}' 관련 데이터 {len(results)}건을 찾았습니다.\n\n요약: {summary_text[:200]}..."
            else:
                ai_reply = f"'{query.strip()}' 관련 검색 결과가 없습니다."
                st.warning(ai_reply)

        st.session_state.chat_messages.append({"role": "assistant", "content": ai_reply})

    else:
        # 일반 대화 (OpenAI)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            answer = ""

            with client.responses.stream(
                model="gpt-5-mini",
                input=[{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages],
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        answer += event.delta
                        placeholder.markdown(answer)
                final = stream.get_final_response()
                answer = final.output_text

            placeholder.markdown(answer)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

st.caption("💡 예시: '소독기 영세율 찾아줘', '펌프 품목 분류 알려줘', '영세율 신청 절차 설명해줘'")
