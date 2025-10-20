# 소득과 세율 변수 선언
income = int(input("당신의 소득을 입력하세요 (만원 단위로 입력): "))
tax_rate = 0.15  # 기본 세율 15%
tax = income * tax_rate

# 소득 구간에 따른 계층 분류
if income >= 7000:
    level = "고소득층"
elif income >= 3000:
    level = "중산층"
else:
    level = "하위층"

# 결과 출력
print("\n===== 결과 =====")
print(f"소득: {income:,}만원")
print(f"세율: {tax_rate * 100:.0f}%")
print(f"세금: {tax:,.0f}만원")
print(f"소득계층: {level}")
