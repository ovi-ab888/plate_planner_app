# app.py — VERSION 2 OPTIMIZER (LOW WASTE LOGIC)

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
from io import BytesIO
from collections import Counter
from math import ceil, floor
import string

st.set_page_config(
    page_title="Pre-Press Planner V2",
    page_icon="🖨️",
    layout="wide"
)

# =====================================================
# Plate Name
# =====================================================

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


# =====================================================
# VERSION 2: SMART BALANCED UPS (CORE LOGIC)
# =====================================================

def smart_layout(demand, cap):

    total = sum(demand.values())

    if total == 0:
        return {}

    # Step 1: fractional allocation
    raw = {}
    floor_vals = {}
    remainders = {}

    for k, v in demand.items():
        ratio = (v / total) * cap
        raw[k] = ratio
        floor_vals[k] = floor(ratio)
        remainders[k] = ratio - floor_vals[k]

    layout = dict(floor_vals)

    # Step 2: fill remaining capacity using largest remainder
    remaining_cap = cap - sum(layout.values())

    while remaining_cap > 0:

        best = max(remainders, key=remainders.get)

        layout[best] += 1
        remainders[best] = 0  # prevent repeat bias
        remaining_cap -= 1

    return layout


# =====================================================
# Auto Planner
# =====================================================

def auto_plan(demand, cap, max_plates):

    remaining = demand.copy()
    plates = []
    produced = Counter()

    for i in range(max_plates):

        if not any(v > 0 for v in remaining.values()):
            break

        layout = smart_layout(remaining, cap)

        if not layout:
            break

        # sheets calculation (balanced)
        possible = [
            ceil(remaining[k] / v)
            for k, v in layout.items()
            if v > 0
        ]

        sheets = max(1, min(possible))

        for k, v in layout.items():

            used = v * sheets

            remaining[k] = max(0, remaining[k] - used)

            produced[k] += used

        plates.append({
            "name": plate_name(len(plates) + 1),
            "layout": layout,
            "sheets": sheets
        })

    # overflow handling
    if any(v > 0 for v in remaining.values()) and plates:

        last = plates[-1]

        for k in remaining:

            if remaining[k] > 0:

                per_sheet = last["layout"].get(k, 1)

                add_sheets = ceil(remaining[k] / per_sheet)

                last["sheets"] += add_sheets

                produced[k] += add_sheets * per_sheet

                remaining[k] = 0

    return plates, dict(produced)


# =====================================================
# UI
# =====================================================

st.title("🖨️ Pre-Press Planner V2 (Low Waste Optimizer)")

col1, col2, col3, col4 = st.columns(4)

n = col1.number_input("Tag Count", 1, 50, 6)
cap = col2.number_input("Plate Capacity", 1, 64, 12)
maxp = col3.number_input("Max Plates", 1, 50, 3)
addon = col4.number_input("Add-on %", 0.0, 50.0, 3.0)

st.markdown("---")
st.subheader("📦 Input Tags")

l, r = st.columns(2)

tags = []
qty = []

for i in range(n):

    name = l.text_input(f"Tag {i+1}", f"Tag {i+1}", key=f"t{i}")
    q = r.number_input(f"{name} Qty", 0, step=10, key=f"q{i}")

    tags.append(name)
    qty.append(q)


# =====================================================
# Data
# =====================================================

original = {
    t: int(q)
    for t, q in zip(tags, qty)
    if q > 0
}

demand = {
    t: ceil(int(q) * (1 + addon / 100))
    for t, q in zip(tags, qty)
    if q > 0
}


# =====================================================
# Generate
# =====================================================

if st.button("🚀 Generate V2 Plan"):

    if not demand:
        st.error("Enter at least 1 tag")
        st.stop()

    plates, produced = auto_plan(demand, cap, maxp)

    # =================================================
    # FINAL TABLE
    # =================================================

    rows = []

    for tag in demand:

        row = {
            "Tag": tag,
            "Original QTY": original[tag],
            "Produced (+Add-on)": demand[tag]
        }

        total = 0

        for p in plates:

            ups = p["layout"].get(tag, 0)

            row[f"Plate {p['name']}"] = ups

            total += ups * p["sheets"]

        excess = total - demand[tag]

        row["Total Produced QTY"] = total
        row["Excess"] = excess
        row["Excess %"] = round((excess / demand[tag]) * 100, 2)

        rows.append(row)

    df = pd.DataFrame(rows)

    st.markdown("## 📊 V2 Optimized Production Summary")

    st.dataframe(df, use_container_width=True)

    # =================================================
    # Plate Info
    # =================================================

    st.markdown("## 🧾 Plate Info")

    pinfo = []

    for p in plates:

        pinfo.append({
            "Plate": p["name"],
            "Sheets": p["sheets"],
            "Total UPS": sum(p["layout"].values())
        })

    st.dataframe(pd.DataFrame(pinfo), use_container_width=True)

    # =================================================
    # Excel Export
    # =================================================

    bio = BytesIO()

    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="V2 Summary", index=False)

    bio.seek(0)

    st.download_button(
        "⬇️ Download Excel",
        data=bio,
        file_name="v2_optimized_plan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.caption("🔥 V2 Optimizer: Low waste, proportional UPS, manual-like accuracy")
