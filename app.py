# app.py — Auto Multi-Plate Pre-Press Planner (Hard cap + Overproduction adjust)
import streamlit as st
import pandas as pd
from collections import Counter
from io import BytesIO
import string
from math import ceil

st.set_page_config(page_title="Pre-Press Plate Planner (Auto)", page_icon="🖨️", layout="wide")

# ========== Helpers ==========
def plate_name(n: int) -> str:
    """Generate A, B, ... Z, AA, AB..."""
    n -= 1
    chars = string.ascii_uppercase
    out = ""
    while True:
        out = chars[n % 26] + out
        n = n // 26 - 1
        if n < 0:
            break
    return out

def normalize_layout(layout: dict, cap: int) -> dict:
    """Keep layout within plate capacity."""
    clean = {k: int(v) for k, v in layout.items() if int(v) > 0}
    total = sum(clean.values())
    if total <= cap:
        return clean
    overflow = total - cap
    for k, v in sorted(clean.items(), key=lambda kv: kv[1]):
        if overflow <= 0:
            break
        take = min(v, overflow)
        clean[k] -= take
        overflow -= take
        if clean[k] <= 0:
            clean.pop(k, None)
    return {k: v for k, v in clean.items() if v > 0}

def proportional_layout(remaining: dict, cap: int) -> dict:
    """Create one plate layout proportional to remaining demand."""
    total = sum(remaining.values())
    if total == 0:
        return {}
    raw = {k: (remaining[k] * cap) / total for k in remaining if remaining[k] > 0}
    floored = {k: int(raw[k]) for k in raw}
    used = sum(floored.values())
    zeros = [k for k in raw if floored[k] == 0]
    for k in zeros:
        if used < cap:
            floored[k] = 1
            used += 1
    remain_slots = max(0, cap - used)
    fracs = sorted(((k, raw[k] - floored[k]) for k in raw), key=lambda x: x[1], reverse=True)
    for i in range(remain_slots):
        if i < len(fracs):
            floored[fracs[i][0]] += 1
    return normalize_layout(floored, cap)

def auto_multi_plate_plan(demand: dict, cap: int, max_plates: int = 20):
    """Automatically build plates until demand met or hard cap reached."""
    remaining = {k: int(v) for k, v in demand.items() if int(v) > 0}
    plates = []
    safe_guard = 10000

    while any(v > 0 for v in remaining.values()) and len(plates) < max_plates and safe_guard > 0:
        layout = proportional_layout(remaining, cap)
        if not layout:
            break

        possible = []
        for sz, ups in layout.items():
            if ups <= 0:
                continue
            possible.append(remaining[sz] // ups)
        sheets = max(1, min(possible) if possible else 1)

        for sz, ups in layout.items():
            remaining[sz] = max(0, remaining[sz] - ups * sheets)

        plates.append({"layout": layout, "sheets": sheets})
        safe_guard -= 1

        if len(plates) >= max_plates:
            st.warning(f"🚧 Hard cap reached ({max_plates} plates). Remaining demand will not be fully covered.")
            break

    # Assign names
    for i, p in enumerate(plates, start=1):
        p["name"] = plate_name(i)

    # Overproduction control — shrink last plate if too much extra
    produced = Counter()
    for p in plates:
        for sz, ups in p["layout"].items():
            produced[sz] += ups * p["sheets"]

    over = {sz: max(0, produced[sz] - demand[sz]) for sz in demand}
    total_over = sum(over.values())
    if total_over > 0 and plates:
        last = plates[-1]
        # reduce last plate’s sheets to minimize overage
        adjust_ratio = max(0.5, 1 - (total_over / sum(demand.values())))
        last["sheets"] = max(1, int(last["sheets"] * adjust_ratio))

        # recompute final production
        produced = Counter()
        for p in plates:
            for sz, ups in p["layout"].items():
                produced[sz] += ups * p["sheets"]

    return plates, dict(produced)

# ========== UI ==========
st.title("🖨️ Pre-Press Plate Planner — Auto Multi-Plate (with Hard Cap)")
st.caption("Tag count, Plate capacity, ও সর্বোচ্চ Plate limit দিন। অ্যাপ স্বয়ংক্রিয়ভাবে Plate A/B/C তৈরি করে প্রতি-Plate ইমপ্রেশন দেখাবে।")

col1, col2, col3 = st.columns(3)
num_tags = col1.number_input("Tag QTY input সংখ্যা", 1, 50, 6)
capacity = col2.number_input("Plate capacity (tags per plate)", 1, 64, 12)
max_plates = col3.number_input("সর্বোচ্চ Plate সংখ্যা (Hard Cap)", 1, 200, 20)

st.markdown("---")
st.subheader("📦 Tag QTY ইনপুট দিন 👇")

left, right = st.columns(2)
tags, qtys = [], []
for i in range(num_tags):
    tname = left.text_input(f"Tag {i+1} Name", f"Tag {i+1}", key=f"t{i}")
    tqty  = right.number_input(f"{tname} Quantity", min_value=0, step=10, key=f"q{i}")
    tags.append(tname)
    qtys.append(tqty)

demand = {t: int(q) for t, q in zip(tags, qtys) if int(q) > 0}

st.markdown("---")

if st.button("🚀 Generate Plan"):
    if not demand:
        st.error("কমপক্ষে ১টি Tag Quantity দিন।")
        st.stop()

    plates, produced = auto_multi_plate_plan(demand, capacity, max_plates)

    if not plates:
        st.warning("পরিকল্পনা তৈরি করা যায়নি। ইনপুট যাচাই করুন।")
        st.stop()

    # Build output table
    cols = ["Plate"] + list(demand.keys()) + ["Sheets (impressions)"]
    rows = []
    for p in plates:
        row = {"Plate": p["name"], "Sheets (impressions)": p["sheets"]}
        for t in demand.keys():
            row[t] = p["layout"].get(t, 0)
        rows.append(row)
    df_plate = pd.DataFrame(rows, columns=cols)

    total_sheets = int(sum(p["sheets"] for p in plates))
    st.markdown("### 🧾 প্রতি Plate-এর সাইজ-আপ + ইমপ্রেশন")
    st.dataframe(df_plate, use_container_width=True)
    st.success(f"✅ মোট শিট (সব Plate মিলিয়ে): {total_sheets}")

    # Excel export
    xout = BytesIO()
    with pd.ExcelWriter(xout, engine="openpyxl") as writer:
        df_plate.to_excel(writer, sheet_name="Per-Plate Layout", index=False)
        pd.DataFrame([{"Tag": k, "Demand": demand[k], "Produced": produced.get(k, 0)} for k in demand])\
            .to_excel(writer, sheet_name="Demand vs Produced", index=False)
    xout.seek(0)
    st.download_button("⬇️ Download Excel", data=xout, file_name="auto_multi_plate_plan.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("💡 এখন Hard Cap অনুযায়ী Plate সীমাবদ্ধ থাকবে, এবং অতিরিক্ত প্রিন্ট হলে শেষ Plate-কে স্বয়ংক্রিয়ভাবে কমিয়ে দেবে।")
