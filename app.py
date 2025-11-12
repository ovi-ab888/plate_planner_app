# app.py — Final Version (Accurate + 3% Add-on + No Underprint)
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
    """Generate A, B, C..."""
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
    """Build proportional plate layout ensuring total <= cap"""
    total = sum(remaining.values())
    if total == 0:
        return {}

    # proportional base
    layout = {k: int((remaining[k] / total) * cap) for k in remaining if remaining[k] > 0}

    # make sure at least 1 if any remaining
    for k in layout:
        if layout[k] == 0 and remaining[k] > 0:
            layout[k] = 1

    # adjust if sum < cap
    while sum(layout.values()) < cap:
        for k in sorted(remaining, key=lambda x: remaining[x], reverse=True):
            if sum(layout.values()) >= cap:
                break
            layout[k] = layout.get(k, 0) + 1

    # ✅ hard limit: trim if total > cap
    while sum(layout.values()) > cap:
        # sort descending by tag value to trim largest first
        for k in sorted(layout, key=lambda x: layout[x], reverse=True):
            if sum(layout.values()) <= cap:
                break
            if layout[k] > 1:
                layout[k] -= 1

    return {k: v for k, v in layout.items() if v > 0}



def auto_plan(demand, cap, max_plates=20):
    """Generate plates ensuring no underprint"""
    remaining = demand.copy()
    plates = []
    safeguard = 1000

    while any(v > 0 for v in remaining.values()) and len(plates) < max_plates and safeguard > 0:
        safeguard -= 1
        layout = proportional_layout(remaining, cap)
        if not layout:
            break

        # determine sheets for this plate
        possible = [ceil(remaining[k] / v) for k, v in layout.items() if v > 0]
        sheets = min(possible) if possible else 1
        sheets = max(1, sheets)

        for k, v in layout.items():
            remaining[k] = max(0, remaining[k] - v * sheets)

        plates.append({"name": plate_name(len(plates) + 1), "layout": layout, "sheets": sheets})

    # calculate total produced
    produced = Counter()
    for p in plates:
        for k, v in p["layout"].items():
            produced[k] += v * p["sheets"]

    # if any underprint remains, fix last plate
    for tag in demand:
        if produced[tag] < demand[tag] and plates:
            deficit = demand[tag] - produced[tag]
            last_plate = plates[-1]
            if tag in last_plate["layout"]:
                per_sheet = last_plate["layout"][tag]
                add_sheets = ceil(deficit / per_sheet)
                last_plate["sheets"] += add_sheets
                produced[tag] += per_sheet * add_sheets
            else:
                # If tag missing, add it to last plate layout minimally
                last_plate["layout"][tag] = 1
                add_sheets = ceil(deficit / 1)
                last_plate["sheets"] += add_sheets
                produced[tag] += add_sheets

    return plates, dict(produced)


# ---------- UI ----------
st.title("🖨️ Auto Multi-Plate Planner (Accurate + Add-on %)")

col1, col2, col3, col4 = st.columns(4)
n = col1.number_input("কতটি Tag", 1, 50, 6)
cap = col2.number_input("Plate capacity", 1, 64, 12)
maxp = col3.number_input("সর্বোচ্চ Plate", 1, 200, 20)
addon = col4.number_input("Add-on % (Extra print)", 0.0, 50.0, 3.0, step=0.5)

st.markdown("---")
st.subheader("📦 Tag QTY দিন")

l, r = st.columns(2)
tags, qty = [], []
for i in range(n):
    name = l.text_input(f"Tag {i+1}", f"Tag {i+1}", key=f"t{i}")
    q = r.number_input(f"{name} Qty", 0, step=10, key=f"q{i}")
    tags.append(name)
    qty.append(q)

# Adjusted demand (Add-on)
demand = {t: ceil(int(q) * (1 + addon / 100)) for t, q in zip(tags, qty) if q > 0}

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

    # Summary: demand vs produced
    summary = pd.DataFrame(
        [{"Tag": k, "Demand(+Add-on)": demand[k], "Produced": prod.get(k, 0)} for k in demand]
    )

    st.markdown("### 📊 Demand vs Produced (সবসময় Produced ≥ Demand থাকবে)")
    st.dataframe(summary, use_container_width=True)

    # Excel Export
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Plates", index=False)
        summary.to_excel(w, sheet_name="Summary", index=False)
    bio.seek(0)

    st.download_button(
        "⬇️ Excel Download",
        data=bio,
        file_name="final_plate_plan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.caption("💡 এই ভার্সনে Add-on % অনুযায়ী Tag QTY বাড়ানো হয় এবং কোনো Tag কম প্রিন্ট হবে না।")
