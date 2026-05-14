# app.py — VERSION 6
# Advanced Industrial Optimizer (NO PASSWORD)

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
from io import BytesIO
from math import ceil
import string
import copy

st.set_page_config(
    page_title="Pre-Press Planner V6",
    page_icon="🖨️",
    layout="wide"
)

# =========================================================
# HELPERS
# =========================================================

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


# =========================================================
# SMART PROPORTIONAL UPS
# =========================================================

def proportional_layout(remaining, capacity):

    active = {
        k: v
        for k, v in remaining.items()
        if v > 0
    }

    if not active:
        return {}

    total_qty = sum(active.values())

    layout = {}

    decimal_map = {}

    # =====================================================
    # INITIAL UPS
    # =====================================================

    for tag, qty in active.items():

        ideal = (
            qty / total_qty
        ) * capacity

        base = int(ideal)

        if base < 1:
            base = 1

        layout[tag] = base

        decimal_map[tag] = ideal - int(ideal)

    # =====================================================
    # FIX OVER CAPACITY
    # =====================================================

    while sum(layout.values()) > capacity:

        biggest = max(
            layout,
            key=layout.get
        )

        if layout[biggest] > 1:
            layout[biggest] -= 1
        else:
            break

    # =====================================================
    # FILL REMAINING UPS
    # =====================================================

    while sum(layout.values()) < capacity:

        best = max(
            decimal_map,
            key=decimal_map.get
        )

        layout[best] += 1

        decimal_map[best] = 0

    return layout


# =========================================================
# CALCULATE SCORE
# =========================================================

def calculate_score(summary_rows):

    total_excess = sum(
        row["Excess"]
        for row in summary_rows
    )

    total_produced = sum(
        row["Total Produced QTY"]
        for row in summary_rows
    )

    if total_produced == 0:
        return 999999

    waste_percent = (
        total_excess / total_produced
    ) * 100

    return waste_percent


# =========================================================
# BUILD SUMMARY
# =========================================================

def build_summary(
    plates,
    demand,
    original_qty
):

    rows = []

    for tag in demand.keys():

        row = {
            "Tag": tag,
            "Original QTY": original_qty[tag],
            "Produced (+Add-on)": demand[tag]
        }

        total_produced = 0

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

    return rows


# =========================================================
# V6 OPTIMIZER
# =========================================================

def v6_optimizer(
    demand,
    capacity,
    max_plates
):

    best_score = 999999

    best_plates = None

    # =====================================================
    # MULTIPLE ATTEMPTS
    # =====================================================

    for variation in range(15):

        remaining = copy.deepcopy(demand)

        plates = []

        for p in range(max_plates):

            active = {
                k: v
                for k, v in remaining.items()
                if v > 0
            }

            if not active:
                break

            layout = proportional_layout(
                active,
                capacity
            )

            # =============================================
            # SHEET STRATEGY
            # =============================================

            possible = []

            for tag, ups in layout.items():

                if ups > 0:

                    sheets = ceil(
                        remaining[tag] / ups
                    )

                    possible.append(sheets)

            if not possible:
                break

            # =============================================
            # DIFFERENT BALANCE STRATEGY
            # =============================================

            possible = sorted(possible)

            strategy_index = min(
                variation % len(possible),
                len(possible) - 1
            )

            sheets = possible[strategy_index]

            sheets = max(1, sheets)

            produced = {}

            for tag, ups in layout.items():

                qty = ups * sheets

                produced[tag] = qty

                remaining[tag] = max(
                    0,
                    remaining[tag] - qty
                )

            plates.append({
                "name": plate_name(len(plates) + 1),
                "layout": layout,
                "sheets": sheets,
                "produced": produced
            })

        # =================================================
        # AUTO FIX REMAINING
        # =================================================

        if any(v > 0 for v in remaining.values()) and plates:

            last = plates[-1]

            for tag in remaining:

                if remaining[tag] > 0:

                    ups = max(
                        1,
                        last["layout"].get(tag, 1)
                    )

                    add_sheets = ceil(
                        remaining[tag] / ups
                    )

                    last["sheets"] += add_sheets

                    remaining[tag] = 0

        # =================================================
        # SCORE
        # =================================================

        summary_rows = build_summary(
            plates,
            demand,
            original_qty
        )

        score = calculate_score(summary_rows)

        if score < best_score:

            best_score = score

            best_plates = plates

    return best_plates, best_score


# =========================================================
# UI
# =========================================================

st.title("🖨️ Pre-Press Planner V6")

st.caption(
    "AI Style Multi-Variation Industrial Optimizer"
)

col1, col2, col3, col4 = st.columns(4)

n = col1.number_input(
    "Tag Count",
    1,
    50,
    8
)

capacity = col2.number_input(
    "Plate Capacity",
    1,
    64,
    12
)

max_plates = col3.number_input(
    "Max Plates",
    1,
    20,
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

left, right = st.columns(2)

tags = []
qtys = []

for i in range(n):

    tag = left.text_input(
        f"Tag {i+1}",
        f"Tag {i+1}",
        key=f"tag{i}"
    )

    qty = right.number_input(
        f"{tag} Qty",
        0,
        step=10,
        key=f"qty{i}"
    )

    tags.append(tag)
    qtys.append(qty)

# =========================================================
# DATA
# =========================================================

original_qty = {
    t: int(q)
    for t, q in zip(tags, qtys)
    if q > 0
}

demand = {
    t: ceil(int(q) * (1 + addon / 100))
    for t, q in zip(tags, qtys)
    if q > 0
}

# =========================================================
# GENERATE
# =========================================================

if st.button("🚀 Generate V6 Plan"):

    if not demand:

        st.error("কমপক্ষে ১টি Qty দিন")

        st.stop()

    progress = st.progress(
        0,
        text="🔄 Running Multi-Variation Optimization..."
    )

    plates, best_score = v6_optimizer(
        demand,
        capacity,
        max_plates
    )

    progress.progress(
        100,
        text="✅ Optimization Complete"
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    rows = build_summary(
        plates,
        demand,
        original_qty
    )

    df = pd.DataFrame(rows)

    # =====================================================
    # TOTAL ROW
    # =====================================================

    total_row = {
        "Tag": "TOTAL",
        "Original QTY": df["Original QTY"].sum(),
        "Produced (+Add-on)": df["Produced (+Add-on)"].sum(),
    }

    for p in plates:

        col = f"Plate {p['name']}"

        total_row[col] = df[col].sum()

    total_row["Total Produced QTY"] = (
        df["Total Produced QTY"].sum()
    )

    total_row["Excess"] = (
        df["Excess"].sum()
    )

    total_row["Excess %"] = round(
        (
            total_row["Excess"]
            /
            total_row["Produced (+Add-on)"]
        ) * 100,
        2
    )

    df = pd.concat(
        [
            df,
            pd.DataFrame([total_row])
        ],
        ignore_index=True
    )

    # =====================================================
    # SHOW
    # =====================================================

    st.markdown("## 📊 Final Production Summary")

    st.dataframe(
        df,
        use_container_width=True
    )

    # =====================================================
    # SCORE
    # =====================================================

    st.success(
        f"🔥 Optimization Score: {round(best_score, 2)}% Waste"
    )

    # =====================================================
    # PLATE INFO
    # =====================================================

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

    # =====================================================
    # TOTALS
    # =====================================================

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

    # =====================================================
    # EXCEL EXPORT
    # =====================================================

    bio = BytesIO()

    with pd.ExcelWriter(
        bio,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="V6 Optimized Summary",
            index=False
        )

    bio.seek(0)

    st.download_button(
        "⬇️ Download Excel",
        data=bio,
        file_name="v6_optimized_plan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =========================================================
# FOOTER
# =========================================================

st.caption(
    "🔥 Version 6 • Multi-Variation AI Style Optimization"
)
