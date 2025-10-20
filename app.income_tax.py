# app.py
import streamlit as st

st.set_page_config(page_title="소득·세금 계산기", layout="centered")
st.title("💰 소득·세금 계산기 (Streamlit)")

st.markdown("""
- 입력 단위: **만원**
- 세금 = 소득 × 세율
- 소득계층: 7,000만원 이상=고소득층 / 3,000~6,999만원=중산층 / 그 미만=하위층
""")

with st.sidebar:
    st.header("입력값")
    income = st.number_input("소득 (만원)", min_value=0, step=100, value=5000)
    tax_rate = st.number_input("세율 (%)", min_value=0.0, step=0.1, value=15.0, format="%.1f")
    run = st.button("계산하기")

def classify(income_manwon: int) -> str:
    if income_manwon >= 7000:
        return "고소득층"
    elif income_manwon >= 3000:
        return "중산층"
    else:
        return "하위층"

def fmt_manwon(x: float) -> str:
    return f"{x:,.0f}만원"

if run:
    tax = income * (tax_rate / 100.0)
    level = classify(income)

    st.subheader("🔎 결과")
    c1, c2, c3 = st.columns(3)
    c1.metric("소득", fmt_manwon(income))
    c2.metric("세율", f"{tax_rate:.1f}%")
    c3.metric("세금", fmt_manwon(tax))

    st.success(f"소득계층: **{level}**")
else:
    st.info("왼쪽 사이드바에서 값을 입력하고 **[계산하기]**를 눌러주세요.")
