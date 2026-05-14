# app.py — VERSION 7
# Industrial AI Optimizer + Smart Plate Balancing
# Streamlit App

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
from io import BytesIO
from math import ceil, floor
import string
import random
import copy

st.set_page_config(
    page_title="Pre-Press Planner V7",
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
# RANDOMIZED SMART LAYOUT
# =========================================================

def generate_layout(active, capacity):

    total_qty = sum(active.values())

    layout = {}

    decimal_map = {}

    # =====================================================
    # IDEAL UPS
    # =====================================================

    for tag, qty in active.items():

        ideal = (
            qty / total_qty
        ) * capacity

        base = floor(ideal)

        if base < 1:
            base = 1

        layout[tag] = base

        decimal_map[tag] = ideal - floor(ideal)

    # =====================================================
    # RANDOM ADJUSTMENT
    # =====================================================

    random_tags = list(active.keys())

    random.shuffle(random_tags)

    while sum(layout.values()) > capacity:

        biggest = max(layout, key=layout.get)

        if layout[biggest] > 1:
            layout[biggest] -= 1
        else:
            break

    while sum(layout.values()) < capacity:

        best = max(decimal_map, key=decimal_map.get)

        layout[best] += 1

        decimal_map[best] = 0

    # =====================================================
    # RANDOM MICRO MUTATION
    # =====================================================

    if len(layout) >= 2:

        for _ in range(2):

            a = random.choice(random_tags)
            b = random.choice(random_tags)

            if a != b and layout[a] > 1:

                layout[a] -= 1
                layout[b] += 1

                if sum(layout.values()) > capacity:
                    layout[b] -= 1
                    layout[a] += 1

    return layout


# =========================================================
# SMART SHEET ENGINE
# =========================================================

def choose_sheet_strategy(layout, remaining):

    options = []

    for tag, ups in layout.items():

        if ups > 0:

            s = ceil(
                remaining[tag] / ups
            )

            options.append(s)

    options = sorted(list(set(options)))

    if not options:
        return 1

    # =====================================================
    # MULTI STRATEGY RANDOM PICK
    # =====================================================

    strategy = random.choice(options)

    return max(1, strategy)


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

            produced_qty = (
                ups * p["sheets"]
            )

            total_produced += produced_qty

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
# AI SCORE ENGINE
# =========================================================

def calculate_score(rows):

    total_excess = sum(
        row["Excess"]
        for row in rows
    )

    total_produced = sum(
        row["Total Produced QTY"]
        for row in rows
    )

    if total_produced == 0:
        return 999999

    waste_percent = (
        total_excess / total_produced
    ) * 100

    # =====================================================
    # BALANCE PENALTY
    # =====================================================

    excess_list = [
        row["Excess"]
        for row in rows
    ]

    balance_penalty = (
        max(excess_list) - min(excess_list)
    ) if excess_list else 0

    final_score = (
        waste_percent
        +
        (balance_penalty * 0.001)
    )

    return final_score


# =========================================================
# V7 OPTIMIZER
# =========================================================

def v7_optimizer(
    demand,
    capacity,
    max_plates,
    iterations=200
):

    best_score = 999999

    best_plates = None

    # =====================================================
    # AI ITERATIONS
    # =====================================================

    for attempt in range(iterations):

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

            # =============================================
            # AI LAYOUT
            # =============================================

            layout = generate_layout(
                active,
                capacity
            )

            # =============================================
            # SMART SHEETS
            # =============================================

            sheets = choose_sheet_strategy(
                layout,
                remaining
            )

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

                    extra = ceil(
                        remaining[tag] / ups
                    )

                    last["sheets"] += extra

                    remaining[tag] = 0

        # =================================================
        # SCORE
        # =================================================

        rows = build_summary(
            plates,
            demand,
            original_qty
        )

        score = calculate_score(rows)

        if score < best_score:

            best_score = score

            best_plates = copy.deepcopy(plates)

    return best_plates, best_score


# =========================================================
# UI
# =========================================================

st.title("🖨️ Pre-Press Planner V7")

st.caption(
    "AI Industrial Optimizer • Smart Mutation • Waste Minimizer"
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

if st.button("🚀 Generate V7 AI Plan"):

    if not demand:

        st.error("কমপক্ষে ১টি Qty দিন")

        st.stop()

    progress = st.progress(
        0,
        text="🤖 AI Optimizing..."
    )

    plates, score = v7_optimizer(
        demand,
        capacity,
        max_plates,
        iterations=300
    )

    progress.progress(
        100,
        text="✅ AI Optimization Complete"
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

    st.success(
        f"🔥 AI Optimization Score: {round(score, 3)}"
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
            sheet_name="V7 AI Optimized Summary",
            index=False
        )

    bio.seek(0)

    st.download_button(
        "⬇️ Download Excel",
        data=bio,
        file_name="v7_ai_optimized_plan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =========================================================
# FOOTER
# =========================================================

st.caption(
    "🔥 Version 7 • AI Mutation Engine • Industrial Optimization"
)
