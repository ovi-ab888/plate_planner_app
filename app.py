# app.py — FINAL HARD CAP FREE VERSION (Accurate + Fixed Capacity + Auto Overprint + Add-on %)
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
from io import BytesIO
from collections import Counter
from math import ceil
import string

st.set_page_config(page_title="Pre-Press Auto Planner", page_icon="🖨️", layout="wide")


# ---------- Helper Functions ----------
def plate_name(n):
    """Generate sequential plate names (A, B, C, ...)."""
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
    """Build proportional plate layout ensuring total == cap (fixed capacity)."""
    total = sum(remaining.values())
    if total == 0:
        return {}

    layout = {k: int((remaining[k] / total) * cap) for k in remaining if remaining[k] > 0}

    # ensure at least 1 for active tags
    for k in layout:
        if layout[k] == 0 and remaining[k] > 0:
            layout[k] = 1

    # fill if less than cap
    while sum(layout.values()) < cap:
        for k in sorted(remaining, key=lambda x: remaining[x], reverse=True):
            if sum(layout.values()) >= cap:
                break
            layout[k] = layout.get(k, 0) + 1

    # trim if greater than cap
    while sum(layout.values()) > cap:
        for k in sorted(layout, key=lambda x: layout[x], reverse=True):
            if sum(layout.values()) <= cap:
                break
            if layout[k] > 1:
                layout[k] -= 1

    # final correction to match capacity exactly
    diff = cap - sum(layout.values())
    if diff > 0:
        for k in sorted(remaining, key=lambda x: remaining[x], reverse=True):
            if diff <= 0:
                break
            layout[k] = layout.get(k, 0) + 1
            diff -= 1
    elif diff < 0:
        for k in sorted(layout, key=lambda x: layout[x], reverse=True):
            if diff >= 0:
                break
            if layout[k] > 1:
                layout[k] -= 1
                diff += 1

    return {k: v for k, v in layout.items() if v > 0}


def auto_plan(demand, cap, max_plates=9999):
    """Generate full plate plan (no hard cap, fixed capacity, no underprint)."""
    remaining = demand.copy()
    plates = []
    produced = Counter()
    safe_guard = 5000  # safety stop (prevent infinite loop)

    while any(v > 0 for v in remaining.values()) and safe_guard > 0:
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

        if all(v == 0 for v in remaining.values()):
            break

    # Fix any underprints (Produced ≥ Demand)
    for tag in demand:
        if produced[tag] < demand[tag] and plates:
            deficit = demand[tag] - produced[tag]
            last = plates[-1]
            last["layout"][tag] = last["layout"].get(tag, 1)
            per_sheet = last["layout"][tag]
            add_sheets = ceil(deficit / per_sheet)
            last["sheets"] += add_sheets
            produced[tag] += add_sheets * per_sheet

    return plates, dict(produced)


# ---------- UI ----------
st.title("🖨️ Auto Multi-Plate Planner (Hard Cap Free Final Version)")

col1, col2, col3, col4 = st.columns(4)
n = col1.number_input("কতটি Tag", 1, 50, 6)
cap = col2.number_input("Plate capacity (tags per plate)", 1, 64, 12)
addon = col3.number_input("Add-on % (Extra print)", 0.0, 50.0, 3.0, step=0.5)
maxp = col4.number_input("Safety Limit (Max Plates)", 50, 9999, 500)

st.markdown("---")
st.subheader("📦 Tag QTY দিন")

l, r = st.columns(2)
tags, qty = [], []
for i in range(n):
    name = l.text_input(f"Tag {i+1}", f"Tag {i+1}", key=f"t{i}")
    q = r.number_input(f"{name} Qty", 0, step=10, key=f"q{i}")
    tags.append(name)
    qty.append(q)

# Adjust demand with add-on %
demand = {t: ceil(int(q) * (1 + addon / 100)) for t, q in zip(tags, qty) if q > 0}

if st.button("🚀 Generate Plan"):
    if not demand:
        st.error("কমপক্ষে ১টি Tag Quantity দিন।")
        st.stop()

    progress = st.progress(0, text="🔄 Calculating Plates...")
    plates, prod = auto_plan(demand, cap, maxp)
    progress.progress(100, text="✅ Done!")

    if not plates:
        st.warning("পরিকল্পনা তৈরি হয়নি। ইনপুট যাচাই করুন।")
        st.stop()

    # Plate layout table
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

    # Summary table
    summary = pd.DataFrame(
        [
            {
                "Tag": k,
                "Demand(+Add-on)": demand[k],
                "Produced": prod.get(k, 0),
                "Extra(Overprint)": prod.get(k, 0) - demand[k],
            }
            for k in demand
        ]
    )

    # Total extra overprint count
    total_extra = sum(summary["Extra(Overprint)"])
    st.markdown("### 📊 Demand vs Produced (Produced ≥ Demand)")
    st.dataframe(summary, use_container_width=True)
    st.info(f"🧾 মোট Extra(Overprint): {total_extra} pcs")

    # Excel export
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Plates", index=False)
        summary.to_excel(w, sheet_name="Summary", index=False)
    bio.seek(0)

    st.download_button(
        "⬇️ Excel Download",
        data=bio,
        file_name="final_plate_plan_hardcap_free.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.caption("💡 এই ভার্সনে Plate capacity সবসময় fixed থাকে, Produced ≥ Demand থাকে, Hard cap warning নেই, এবং প্রয়োজন অনুযায়ী Extra(Overprint) auto adjust হয়।")
