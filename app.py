import streamlit as st
import pandas as pd
from collections import Counter
from math import ceil

st.title("🖨️ Pre-Press Plate Planner")

st.write("Tag size & QTY input দিন, নিচে ফলাফল দেখাবে।")

sizes = st.text_input("Tag sizes (comma separated):", "XS,S,M,L,XL,XXL,3XL").split(",")
qtys = st.text_input("Quantities (comma separated):", "100,100,150,150,100,100,50").split(",")
capacity = st.number_input("Plate capacity (tags per plate):", 12)

demand = {s.strip(): int(q) for s, q in zip(sizes, qtys)}

if st.button("Generate Plan"):
    total = sum(demand.values())
    raw = {k: (v * capacity) / total for k, v in demand.items()}
    layout = {k: round(v) for k, v in raw.items()}
    produced = {k: layout[k] * 100 for k in layout}  # placeholder calc
    df = pd.DataFrame({"Size": layout.keys(), "Per Plate": layout.values()})
    st.subheader("Plate Layout Suggestion")
    st.dataframe(df)
    st.download_button("Download Layout CSV", df.to_csv(index=False), "plate_layout.csv")
