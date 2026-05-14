# app.py — V4 COMMON SHEET OPTIMIZER
# Industrial Style Low-Waste Planner

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
from io import BytesIO
from math import ceil
import string

st.set_page_config(
    page_title="Pre-Press Planner V4",
    page_icon="🖨️",
    layout="wide"
)

# =========================================================
# PASSWORD
# =========================================================

APP_PASSWORD = "aapress"


def check_password():

    st.markdown("""
    <style>
    .login-box{
        background:linear-gradient(135deg,#667eea,#764ba2);
        padding:60px;
        border-radius:25px;
        text-align:center;
        color:white;
        margin-top:40px;
    }

    .title{
        font-size:48px;
        font-weight:700;
    }

    .subtitle{
        font-size:20px;
        opacity:0.9;
        margin-top:10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="login-box">
        <div class="title">🖨️ Pre-Press Planner V4</div>
        <div class="subtitle">
            Common Sheet Optimizer • Industrial Low Waste Planning
        </div>
    </div>
    """, unsafe_allow_html=True)

    password = st.text_input(
        "🔐 Enter Password",
        type="password"
    )

    if password == APP_PASSWORD:
        st.session_state["auth"] = True
        st.rerun()

    elif password:
        st.error("❌ Incorrect Password")


if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    check_password()
    st.stop()

# =========================================================
# Helper
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
# COMMON SHEET OPTIMIZER
# =========================================================

def common_sheet_optimizer(demand, capacity, max_plates):

    total_qty = sum(demand.values())

    # estimated common sheet target
    target_sheets = ceil(total_qty / capacity)

    remaining = demand.copy()

    plates = []

    for p in range(max_plates):

        active = {
            k: v
            for k, v in remaining.items()
            if v > 0
        }

        if not active:
            break

        # =================================================
        # IDEAL UPS CALCULATION
        # =================================================

        ideal = {}

        for tag, qty in active.items():

            ups = qty / target_sheets

            ideal[tag] = ups

        # =================================================
        # INITIAL INTEGER UPS
        # =================================================

        layout = {
            k: max(1, round(v))
            for k, v in ideal.items()
        }

        # =================================================
        # FIX TOTAL UPS
        # =================================================

        while sum(layout.values()) > capacity:

            biggest = max(layout, key=layout.get)

            if layout[biggest] > 1:
                layout[biggest] -= 1
            else:
                break

        while sum(layout.values()) < capacity:

            # highest remaining qty gets extra UPS
            biggest = max(active, key=active.get)

            layout[biggest] += 1

        # =================================================
        # COMMON SHEET CALCULATION
        # =================================================

        possible_sheets = []

        for tag, ups in layout.items():

            if ups > 0:

                sheets_needed = ceil(
                    remaining[tag] / ups
                )

                possible_sheets.append(sheets_needed)

        sheets = min(possible_sheets)

        sheets = max(1, sheets)

        # =================================================
        # PRODUCE
        # =================================================

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

    # =====================================================
    # AUTO OVERPRINT FIX
    # =====================================================

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

                last["produced"][tag] = (
                    last["produced"].get(tag, 0)
                    +
                    add_sheets * ups
                )

                remaining[tag] = 0

    return plates


# =========================================================
# UI
# =========================================================

st.title("🖨️ Pre-Press Planner V4")

st.caption(
    "Industrial Common Sheet Optimizer • Low Waste • Smart UPS"
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
# DEMAND
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

if st.button("🚀 Generate Optimized Plan"):

    if not demand:

        st.error("কমপক্ষে ১টি Qty দিন")

        st.stop()

    progress = st.progress(
        0,
        text="🔄 Optimizing..."
    )

    plates = common_sheet_optimizer(
        demand,
        capacity,
        max_plates
    )

    progress.progress(
        100,
        text="✅ Optimization Complete"
    )

    # =====================================================
    # SUMMARY TABLE
    # =====================================================

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
            sheet_name="V4 Optimized Summary",
            index=False
        )

    bio.seek(0)

    st.download_button(
        "⬇️ Download Excel",
        data=bio,
        file_name="v4_optimized_plan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =========================================================
# FOOTER
# =========================================================

st.caption(
    "🔥 Version 4 • Common Sheet Optimization • Industrial Low Waste Planning"
)
