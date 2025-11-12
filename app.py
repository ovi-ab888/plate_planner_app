# app.py — Improved Calculation Logic (Accurate Impressions)
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
from io import BytesIO
from collections import Counter
from math import ceil
import string

st.set_page_config(page_title="Pre-Press Auto Planner", page_icon="🖨️", layout="wide")

# ---------- Helpers ----------
def plate_name(n):
    n -= 1
    chars = string.ascii_uppercase
    out = ""
    while True:
        out = chars[n % 26] + out
        n = n // 26 - 1
        if n < 0:
            break
    return out

def proportional_layout(remaining, cap):
    """build proportional plate layout"""
    total = sum(remaining.values())
    if total == 0:
        return {}
    layout = {k: int((remaining[k] / total) * cap) for k in remaining if remaining[k] > 0}
    # distribute leftover slots
    while sum(layout.values()) < cap:
        for k in sorted(remaining, key=lambda x: remaining[x], reverse=True):
            if sum(layout.values()) >= cap:
                break
            layout[k] = layout.get(k, 0) + 1
    # remove zeros
    layout = {k: v for k, v in layout.items() if v > 0}
    return layout

def auto_plan(demand, cap, max_plates=20):
    remaining = demand.copy()
    plates = []
    safeguard = 1000
    while any(v > 0 for v in remaining.values()) and len(plates) < max_plates and safeguard > 0:
        safeguard -= 1
        layout = proportional_layout(remaining, cap)
        if not layout:
            break

        # calculate max possible sheets for this plate (no tag underflows)
        possible_sheets = [
            ceil(remaining[tag] / ups) if ups > 0 else 0 for tag, ups in layout.items()
        ]
        sheets = min(possible_sheets) if possible_sheets else 0
        sheets = max(1, sheets)  # ensure at least one sheet

        # update remaining demand
        for tag, ups in layout.items():
            remaining[tag] = max(0, remaining[tag] - ups * sheets)

        plates.append({"name": plate_name(len(plates) + 1), "layout": layout, "sheets": sheets})

    # Final total produced
    produced = Counter()
    for p in plates:
        for tag, ups in p["layout"].items():
            produced[tag] += ups * p["sheets"]

    return plates, dict(produced)

# ---------- UI ----------
st.title("🖨️ Auto Multi-Plate Planner (Accurate Calculation)")
col1, col2, col3 = st.columns(3)
n = col1.number_input("কতটি Tag", 1, 50, 6)
cap = col2.number_input("Plate capacity", 1, 64, 12)
maxp = col3.number_input("সর্বোচ্চ Plate", 1, 200, 20)

st.markdown("---")
st.subheader("📦 Tag QTY দিন")
l, r = st.columns(2)
tags, qty = [], []
for i in range(n):
    name = l.text_input(f"Tag {i+1}", f"Tag {i+1}", key=f"t{i}")
    q = r.number_input(f"{name} Qty", 0, step=10, key=f"q{i}")
    tags.append(name)
    qty.append(q)
demand = {t: int(q) for t, q in zip(tags, qty) if q > 0}

if st.button("🚀 Generate Plan"):
    if not demand:
        st.error("Tag QTY দিন")
        st.stop()

    progress = st.progress(0, text="🔄 Calculating Plates...")
    plates, prod = auto_plan(demand, cap, maxp)
    progress.progress(100, text="✅ Done!")

    if not plates:
        st.warning("পরিকল্পনা তৈরি হয়নি")
        st.stop()

    cols = ["Plate"] + list(demand.keys()) + ["Sheets"]
    rows = []
    for p in plates:
        row = {"Plate": p["name"], "Sheets": p["sheets"]}
        for t in demand.keys():
            row[t] = p["layout"].get(t, 0)
        rows.append(row)
    df = pd.DataFrame(rows, columns=cols)

    total = sum(p["sheets"] for p in plates)
    st.markdown("### 🧾 প্রতি Plate-এর সাইজ-আপ + ইমপ্রেশন")
    st.dataframe(df, use_container_width=True)
    st.success(f"✅ মোট শিট: {total}")

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Plates", index=False)
        pd.DataFrame(
            [{"Tag": k, "Demand": demand[k], "Produced": prod.get(k, 0)} for k in demand]
        ).to_excel(w, sheet_name="Summary", index=False)
    bio.seek(0)
    st.download_button(
        "⬇️ Excel",
        data=bio,
        file_name="accurate_plate_plan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
