# app.py — FINAL CLEAN VERSION

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
from io import BytesIO
from collections import Counter
from math import ceil
import string

st.set_page_config(
    page_title="Pre-Press Auto Planner",
    page_icon="🖨️",
    layout="wide"
)

# ---------- Helper Functions ----------

def plate_name(n):
    """Generate plate names A, B, C..."""
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

    total = sum(remaining.values())

    if total == 0:
        return {}

    layout = {
        k: int((remaining[k] / total) * cap)
        for k in remaining if remaining[k] > 0
    }

    # minimum 1
    for k in layout:
        if layout[k] == 0:
            layout[k] = 1

    # fill capacity
    while sum(layout.values()) < cap:

        for k in sorted(
            remaining,
            key=lambda x: remaining[x],
            reverse=True
        ):

            if sum(layout.values()) >= cap:
                break

            layout[k] += 1

    # trim capacity
    while sum(layout.values()) > cap:

        for k in sorted(
            layout,
            key=lambda x: layout[x],
            reverse=True
        ):

            if sum(layout.values()) <= cap:
                break

            if layout[k] > 1:
                layout[k] -= 1

    return layout


def auto_plan(demand, cap, max_plates=3):

    remaining = demand.copy()

    plates = []

    produced = Counter()

    for i in range(max_plates):

        if not any(v > 0 for v in remaining.values()):
            break

        layout = proportional_layout(
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

        sheets = min(possible) if possible else 1

        sheets = max(1, sheets)

        for k, v in layout.items():

            remaining[k] = max(
                0,
                remaining[k] - (v * sheets)
            )

            produced[k] += v * sheets

        plates.append({
            "name": plate_name(len(plates) + 1),
            "layout": layout,
            "sheets": sheets
        })

    # auto overprint
    if any(v > 0 for v in remaining.values()) and plates:

        last = plates[-1]

        for tag in remaining:

            if remaining[tag] > 0:

                per_sheet = last["layout"].get(tag, 1)

                add_sheets = ceil(
                    remaining[tag] / per_sheet
                )

                last["sheets"] += add_sheets

                produced[tag] += add_sheets * per_sheet

                remaining[tag] = 0

    return plates, dict(produced)


# ---------- UI ----------

st.title("🖨️ Auto Multi-Plate Planner")

col1, col2, col3, col4 = st.columns(4)

n = col1.number_input(
    "কতটি Tag",
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
    "কতটি Plate বানাতে চান",
    1,
    50,
    3
)

addon = col4.number_input(
    "Add-on %",
    0.0,
    50.0,
    3.0,
    step=0.5
)

st.markdown("---")

st.subheader("📦 Tag QTY দিন")

l, r = st.columns(2)

tags = []
qty = []

for i in range(n):

    name = l.text_input(
        f"Tag {i+1}",
        f"Tag {i+1}",
        key=f"t{i}"
    )

    q = r.number_input(
        f"{name} Qty",
        0,
        step=10,
        key=f"q{i}"
    )

    tags.append(name)
    qty.append(q)

# ---------- Demand ----------

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

# ---------- Generate ----------

if st.button("🚀 Generate Plan"):

    if not demand:

        st.error("কমপক্ষে ১টি Tag Qty দিন")

        st.stop()

    progress = st.progress(
        0,
        text="🔄 Calculating..."
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

    # ---------- Final Summary ----------

    summary_rows = []

    for tag in demand.keys():

        row = {
            "Tag": tag,
            "Original QTY": original_qty[tag],
            "Produced (+Add-on)": demand[tag]
        }

        total_produced = 0

        for p in plates:

            # UPS qty
            ups = p["layout"].get(tag, 0)

            row[f"Plate {p['name']}"] = ups

            # produced qty
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

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    st.markdown("## 📊 Final Production Summary")

    st.dataframe(
        summary_df,
        use_container_width=True
    )

    # ---------- Plate Information ----------

    st.markdown("## 🧾 Plate Sheet Information")

    plate_info = []

    for p in plates:

        plate_info.append({
            "Plate": p["name"],
            "Sheets": p["sheets"]
        })

    plate_info_df = pd.DataFrame(plate_info)

    st.dataframe(
        plate_info_df,
        use_container_width=True
    )

    # ---------- Totals ----------

    total_sheets = sum(
        p["sheets"]
        for p in plates
    )

    total_excess = summary_df["Excess"].sum()

    st.success(f"✅ Total Sheets: {total_sheets}")

    st.info(f"🧾 Total Excess: {total_excess}")

    # ---------- Excel Export ----------

    bio = BytesIO()

    with pd.ExcelWriter(
        bio,
        engine="openpyxl"
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="Production Summary",
            index=False
        )

    bio.seek(0)

    st.download_button(
        "⬇️ Excel Download",
        data=bio,
        file_name="production_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.caption(
    "💡 Dynamic multi-plate planning with automatic overprint handling."
)
