# app.py — VERSION 3 OPTIMIZER (LOW WASTE + TOTAL ROW)

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
from io import BytesIO
from collections import Counter
from math import ceil, floor
import string

st.set_page_config(
    page_title="Pre-Press Planner V3",
    page_icon="🖨️",
    layout="wide"
)

# =====================================================
# Plate Name Generator
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
# SMART BALANCED UPS
# =====================================================

def smart_layout(demand, cap):

    total = sum(demand.values())

    if total == 0:
        return {}

    raw = {}
    floor_vals = {}
    remainders = {}

    for k, v in demand.items():

        ratio = (v / total) * cap

        raw[k] = ratio

        floor_vals[k] = floor(ratio)

        remainders[k] = ratio - floor_vals[k]

    layout = dict(floor_vals)

    # ensure at least 1 UPS
    for k in layout:

        if layout[k] == 0:
            layout[k] = 1

    # fix overflow
    while sum(layout.values()) > cap:

        biggest = max(layout, key=layout.get)

        if layout[biggest] > 1:
            layout[biggest] -= 1
        else:
            break

    # fill remaining
    remaining_cap = cap - sum(layout.values())

    while remaining_cap > 0:

        best = max(remainders, key=remainders.get)

        layout[best] += 1

        remainders[best] = 0

        remaining_cap -= 1

    return layout


# =====================================================
# AUTO PLAN
# =====================================================

def auto_plan(demand, cap, max_plates):

    remaining = demand.copy()

    plates = []

    produced = Counter()

    for i in range(max_plates):

        if not any(v > 0 for v in remaining.values()):
            break

        layout = smart_layout(
            remaining,
            cap
        )

        if not layout:
            break

        possible = [
            ceil(remaining[k] / v)
            for k, v in layout.items()
            if v > 0
        ]

        sheets = max(1, min(possible))

        for k, v in layout.items():

            produced_qty = v * sheets

            remaining[k] = max(
                0,
                remaining[k] - produced_qty
            )

            produced[k] += produced_qty

        plates.append({
            "name": plate_name(len(plates) + 1),
            "layout": layout,
            "sheets": sheets
        })

    # =================================================
    # AUTO OVERPRINT FIX
    # =================================================

    if any(v > 0 for v in remaining.values()) and plates:

        last = plates[-1]

        for k in remaining:

            if remaining[k] > 0:

                per_sheet = max(
                    1,
                    last["layout"].get(k, 1)
                )

                add_sheets = ceil(
                    remaining[k] / per_sheet
                )

                last["sheets"] += add_sheets

                produced[k] += add_sheets * per_sheet

                remaining[k] = 0

    return plates, dict(produced)


# =====================================================
# UI
# =====================================================

st.title("🖨️ Pre-Press Planner V3")

col1, col2, col3, col4 = st.columns(4)

n = col1.number_input(
    "Tag Count",
    1,
    50,
    6
)

cap = col2.number_input(
    "Plate Capacity",
    1,
    64,
    12
)

maxp = col3.number_input(
    "Max Plates",
    1,
    50,
    2
)

addon = col4.number_input(
    "Add-on %",
    0.0,
    50.0,
    0.0,
    step=0.5
)

st.markdown("---")

st.subheader("📦 Tag QTY")

left, right = st.columns(2)

tags = []
qty = []

for i in range(n):

    name = left.text_input(
        f"Tag {i+1}",
        f"Tag {i+1}",
        key=f"tag_{i}"
    )

    q = right.number_input(
        f"{name} Qty",
        0,
        step=10,
        key=f"qty_{i}"
    )

    tags.append(name)

    qty.append(q)

# =====================================================
# DATA
# =====================================================

original_qty = {
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
# GENERATE
# =====================================================

if st.button("🚀 Generate V3 Plan"):

    if not demand:

        st.error("কমপক্ষে ১টি Tag Qty দিন")

        st.stop()

    progress = st.progress(
        0,
        text="🔄 Optimizing..."
    )

    plates, produced = auto_plan(
        demand,
        cap,
        maxp
    )

    progress.progress(
        100,
        text="✅ Done!"
    )

    # =================================================
    # FINAL SUMMARY
    # =================================================

    rows = []

    for tag in demand:

        row = {
            "Tag": tag,
            "Original QTY": original_qty[tag],
            "Produced (+Add-on)": demand[tag]
        }

        total_produced = 0

        # dynamic plates
        for p in plates:

            ups = p["layout"].get(tag, 0)

            row[f"Plate {p['name']}"] = ups

            total_produced += (
                ups * p["sheets"]
            )

        excess = total_produced - demand[tag]

        excess_percent = (
            round(
                (excess / demand[tag]) * 100,
                2
            )
            if demand[tag]
            else 0
        )

        row["Total Produced QTY"] = total_produced

        row["Excess"] = excess

        row["Excess %"] = excess_percent

        rows.append(row)

    df = pd.DataFrame(rows)

    # =================================================
    # TOTAL ROW
    # =================================================

    total_row = {
        "Tag": "TOTAL",
        "Original QTY": df["Original QTY"].sum(),
        "Produced (+Add-on)": df["Produced (+Add-on)"].sum(),
    }

    for p in plates:

        col = f"Plate {p['name']}"

        total_row[col] = df[col].sum()

    total_row["Total Produced QTY"] = df["Total Produced QTY"].sum()

    total_row["Excess"] = df["Excess"].sum()

    total_row["Excess %"] = round(
        (
            total_row["Excess"]
            /
            total_row["Produced (+Add-on)"]
        ) * 100,
        2
    )

    # add total row
    df = pd.concat(
        [
            df,
            pd.DataFrame([total_row])
        ],
        ignore_index=True
    )

    # =================================================
    # SHOW TABLE
    # =================================================

    st.markdown("## 📊 V3 Optimized Production Summary")

    st.dataframe(
        df,
        use_container_width=True
    )

    # =================================================
    # PLATE INFO
    # =================================================

    st.markdown("## 🧾 Plate Information")

    plate_rows = []

    for p in plates:

        plate_rows.append({
            "Plate": p["name"],
            "Sheets": p["sheets"],
            "Total UPS": sum(
                p["layout"].values()
            )
        })

    plate_df = pd.DataFrame(plate_rows)

    st.dataframe(
        plate_df,
        use_container_width=True
    )

    # =================================================
    # TOTAL INFO
    # =================================================

    total_sheets = sum(
        p["sheets"]
        for p in plates
    )

    total_excess = df.iloc[:-1]["Excess"].sum()

    st.success(
        f"✅ Total Sheets: {total_sheets}"
    )

    st.info(
        f"🧾 Total Excess: {total_excess}"
    )

    # =================================================
    # EXCEL EXPORT
    # =================================================

    bio = BytesIO()

    with pd.ExcelWriter(
        bio,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="V3 Production Summary",
            index=False
        )

    bio.seek(0)

    st.download_button(
        "⬇️ Download Excel",
        data=bio,
        file_name="v3_optimized_plan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =====================================================
# FOOTER
# =====================================================

st.caption(
    "🔥 Version 3 Optimizer — Low Waste + Smart UPS + Total Summary"
)
