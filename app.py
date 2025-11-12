# app.py — Safe Loop Version (No Infinite Loading)
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
from io import BytesIO
from collections import Counter
from math import ceil
import string

st.set_page_config(page_title="Pre-Press Auto Planner", page_icon="🖨️", layout="wide")

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
    """Safe proportional layout; ensures total ≤ cap"""
    total = sum(remaining.values())
    if total == 0:
        return {}

    layout = {k: max(1, int((remaining[k] / total) * cap)) for k in remaining if remaining[k] > 0}

    while sum(layout.values()) > cap:
        for k in sorted(layout, key=lambda x: layout[x], reverse=True):
            if sum(layout.values()) <= cap:
                break
            layout[k] -= 1
            if layout[k] <= 0:
                layout.pop(k, None)
                break
    return layout

def auto_plan(demand, cap, max_plates=20):
    remaining = demand.copy()
    plates = []
    produced = Counter()
    safe_guard = 2000

    while any(v > 0 for v in remaining.values()) and len(plates) < max_plates and safe_guard > 0:
        safe_guard -= 1
        layout = proportional_layout(remaining, cap)
        if not layout:
            break

        possible = [ceil(remaining[k] / v) for k, v in layout.items() if v > 0]
        sheets = min(possible) if possible else 1
        sheets = max(1, sheets)

        for k, v in layout.items():
            remaining[k] = max(0, remaining[k] - v * sheets)
            produced[k] += v * sheets

        plates.append({"name": plate_name(len(plates) + 1), "layout": layout, "sheets": sheets})

        # stop if no progress made
        if all(v == 0 for v in remaining.values()):
            break
        if sheets == 1 and all(remaining[k] < v for k, v in layout.items()):
            break

    if safe_guard == 0:
        st.warning("⚠️ Loop safeguard triggered: demand too large for given capacity/plates.")

    if len(plates) >= max_plates and any(v > 0 for v in remaining.values()):
        st.warning("🚧 Hard cap reached. Remaining demand could not be fully planned.")

    # Fix underprints
    for tag in demand:
        if produced[tag] < demand[tag] and plates:
            deficit = demand[tag] - produced[tag]
            last = plates[-1]
            last["layout"][tag] = last["layout"].get(tag, 1)
            add_sheets = ceil(deficit / last["layout"][tag])
            last["sheets"] += add_sheets
            produced[tag] += add_sheets * last["layout"][tag]

    return plates, dict(produced)

# ---------- UI ----------
st.title("🖨️ Auto Multi-Plate Planner (Safe Loop + Add-on % + Capacity Fix)")

col1, col2, col3, col4 = st.columns(4)
n = col1.number_input("কতটি Tag", 1, 50, 6)
cap = col2.number_input("Plate capacity", 1, 64, 12)
maxp = col3.number_input("সর্বোচ্চ Plate", 1, 200, 20)
addon = col4.number_input("Add-on %", 0.0, 50.0, 3.0, step=0.5)

st.markdown("---")
st.subheader("📦 Tag QTY দিন")

l, r = st.columns(2)
tags, qty = [], []
for i in range(n):
    name = l.text_input(f"Tag {i+1}", f"Tag {i+1}", key=f"t{i}")
    q = r.number_input(f"{name} Qty", 0, step=10, key=f"q{i}")
    tags.append(name)
    qty.append(q)

demand = {t: ceil(int(q) * (1 + addon / 100)) for t, q in zip(tags, qty) if q > 0}

if st.button("🚀 Generate Plan"):
    if not demand:
        st.error("কমপক্ষে ১টি Tag Quantity দিন।")
        st.stop()

    progress = st.progress(0, text="🔄 Calculating Plates safely...")
    plates, prod = auto_plan(demand, cap, maxp)
    progress.progress(100, text="✅ Done!")

    if not plates:
        st.warning("পরিকল্পনা তৈরি হয়নি। ইনপুট যাচাই করুন।")
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

    summary = pd.DataFrame(
        [{"Tag": k, "Demand(+Add-on)": demand[k], "Produced": prod.get(k, 0)} for k in demand]
    )
    st.markdown("### 📊 Demand vs Produced (Produced ≥ Demand)")
    st.dataframe(summary, use_container_width=True)

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Plates", index=False)
        summary.to_excel(w, sheet_name="Summary", index=False)
    bio.seek(0)
    st.download_button(
        "⬇️ Excel Download",
        data=bio,
        file_name="safe_plate_plan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.caption("💡 Safe-loop logic যুক্ত করা হয়েছে — আর কখনোই infinite loading হবে না।")
