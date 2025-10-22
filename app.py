import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="������ �Ǻ� �˻� ����", layout="wide")

st.title("������ �Ǻ� �˻� ����")
st.caption("Ű���带 �Է��ϸ� �ش� ǰ��� �з�(����ȯ�޽�û / ������TI ����)�� ã���ݴϴ�.")

# --- ���� �ε� ���� ---
default_path = "/mnt/data/�������Ǻ�.xlsx"  # ���ε��� �⺻ ���
uploaded = st.file_uploader("���� ���� ���ε� (.xlsx)", type=["xlsx"])
path_info = st.empty()

@st.cache_data(show_spinner=False)
def load_excel(fileobj_or_path):
    # ���� ��� �Ǵ� ���ε� ���� ��� ����
    if isinstance(fileobj_or_path, str):
        df = pd.read_excel(fileobj_or_path, sheet_name=0)
    else:
        df = pd.read_excel(fileobj_or_path, sheet_name=0)

    # �� ����: �츮�� Ȯ���� ������ ���� ǥ��ȭ
    cols = list(df.columns)
    # ǰ�� �ĺ� �� ã�� (�⺻: 'Unnamed: 1')
    item_col = None
    if "Unnamed: 1" in cols:
        item_col = "Unnamed: 1"
    else:
        # �޸���ƽ: ���ڿ� ���̰� �� object ���� ǰ������ ����
        object_cols = [c for c in cols if df[c].dtype == 'object']
        if object_cols:
            # ��� ���ڿ� ���� ���� �� �� ����
            avg_len = {c: df[c].dropna().astype(str).str.len().mean() for c in object_cols}
            item_col = max(avg_len, key=avg_len.get)

    # ���� �� ã�� (�⺻: '����')
    kind_col = "����" if "����" in cols else None

    # ǥ�� �÷������� ������
    rename_map = {}
    if item_col and item_col != "ǰ��":
        rename_map[item_col] = "ǰ��"
    if kind_col and kind_col != "����":
        rename_map[kind_col] = "����"

    df = df.rename(columns=rename_map)

    # �ּ� �÷� ��ȿ�� Ȯ��
    if "ǰ��" not in df.columns:
        raise ValueError("ǰ�� ���� ã�� ���߽��ϴ�. �������� ǰ���� �ִ� ���� 'Unnamed: 1' �Ǵ� �ؽ�Ʈ�� ��κ��� ���� �����ּ���.")
    if "����" not in df.columns:
        # ������ ������ �� ������ ����(�ɼ�)
        df["����"] = ""

    # ����
    df["ǰ��"] = df["ǰ��"].astype(str).str.strip()
    df["����"] = df["����"].astype(str).str.strip()
    return df[["ǰ��", "����"]]

# ���� ����
if uploaded is not None:
    df = load_excel(uploaded)
    path_info.info("���ε��� ������ ��� ���Դϴ�.")
else:
    # �⺻ ��� ��� (����/���� ȯ�濡 �� ��ΰ� ���� ���� ������ ����ó��)
    try:
        df = load_excel(default_path)
        path_info.info(f"�⺻ ���� ��� ��: {default_path}")
    except Exception as e:
        st.warning("�⺻ ����� ������ �ҷ��� �� �����ϴ�. ��ܿ��� ������ ���ε� ���ּ���.")
        st.stop()

# --- �˻� UI ---
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("�˻��� �Է� (��: �ҵ���, �ʸ�, ���� ...)", value="")
    with col2:
        case_sensitive = st.checkbox("��ҹ��� ����", value=False)

# --- �˻� ���� ---
def search(df, q, case_sensitive=False):
    if not q:
        return df.copy()
    # ��ǥ�� ���� Ű���� ����: "�ҵ���, �ʸ�"
    tokens = [t.strip() for t in q.split(",") if t.strip()]
    if not tokens:
        return df.copy()

    # contains ������ ��� AND�� ���� (�ʿ� �� OR �������� ���� ����)
    res = df.copy()
    for t in tokens:
        if case_sensitive:
            res = res[res["ǰ��"].str.contains(t, na=False)]
        else:
            res = res[res["ǰ��"].str.contains(t, case=False, na=False)]
    return res

results = search(df, query, case_sensitive=case_sensitive)

# --- ��� ---
left, right = st.columns([2, 1])
with right:
    st.subheader("�з��� ����")
    if "����" in results.columns:
        st.dataframe(results["����"].value_counts(dropna=False).rename("�Ǽ�").to_frame())

with left:
    st.subheader("�˻� ���")
    st.caption("ǰ���� Ŭ���� ��ü �ؽ�Ʈ�� Ȯ���ϼ���.")
    st.dataframe(results, use_container_width=True, height=480)

# --- �ٿ�ε� ---
def to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")

csv_bytes = to_csv_bytes(results)
st.download_button(
    label="�˻� ��� CSV �ٿ�ε�",
    data=csv_bytes,
    file_name="�˻����.csv",
    mime="text/csv",
)

# --- ����/�� ---
with st.expander("�˻� ��"):
    st.markdown("""
- ���� Ű����� ��ǥ(,)�� �����ؼ� �Է��ϼ���. ��: `�ҵ���, �����`
- ����� **ǰ�� �� ���� �˻�**(�κ� ��ġ) �����Դϴ�.
- �⺻ ������ **AND �˻�**����, �Է��� ��� Ű���带 �����ϴ� �׸� �����ݴϴ�.
    - OR �˻��� �ʿ��ϸ�, Ű���� �ϳ��� �˻��ϰų� �ڵ��� `search` �Լ��� ������ �����ϼ���.
""")
