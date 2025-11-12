# app.py (Simplified Pre-Press Planner)
import streamlit as st
import pandas as pd
from collections import Counter
from io import BytesIO

st.set_page_config(page_title="Pre-Press Plate Planner", page_icon="🖨️", layout="wide")

# =========================
# Helper functions
# =========================
def ratio_single_plate_layout(demand: dict, cap: int):
    """Build 1 plate layout by proportional allocation to demand."""
    total = sum(demand.values())
    if total == 0:
        return {}
    raw = {k: (demand[k] * cap) / total for k in demand}
    floored = {k: int(raw[k]) for k in raw}
    used = sum(floored.values())
    remain = cap - used

    # distribute remaining slots
    fracs = sorted(((k, raw[k] - floored[k]) for k in raw), key=lambda x: x[1], reverse=True)
    layout = dict(floored)
    for i in range(remain):
        if i < len(fracs):
            layout[fracs[i][0]] += 1

    return {k: v for k, v in layout.items() if v > 0}


def greedy_sheet_plan(demand: dict, layout: dict):
    """Compute how many sheets needed for a single plate layout."""
    remaining = dict(demand)
    sheets = 0
    safe_guard = 10000  # stop infinite loop
    
    while any(v > 0 for v in remaining.values()) and safe_guard > 0:
        sheets += 1
        for sz, cnt in layout.items():
            if remaining.get(sz, 0) > 0:
                remaining[sz] = max(0, remaining[sz] - cnt)
        safe_guard -= 1

    produced = Counter()
    for sz, cnt in layout.items():
        produced[sz] += cnt * sheets

    overage = {sz: produced[sz] - demand[sz] for sz in demand}
    return sheets, dict(produced), overage



# =========================
# UI — Step 1
# =========================
st.title("🖨️ Pre-Press Plate Planner")
st.caption("সহজভাবে Tag Quantity আর Plate capacity দিয়ে হিসাব করুন।")

num_tags = st.number_input("কতটা Tag QTY ইনপুট দিতে চান?", min_value=1, max_value=50, value=7, step=1)
capacity = st.number_input("Plate capacity (tags per plate):", min_value=1, max_value=64, value=12, step=1)

st.markdown("---")
st.subheader("Tag QTY ইনপুট দিন 👇")

tags = []
qtys = []
cols = st.columns(2)
for i in range(num_tags):
    tag_name = cols[0].text_input(f"Tag {i+1} Name", f"Tag {i+1}")
    tag_qty = cols[1].number_input(f"{tag_name} Quantity", min_value=0, step=10, key=f"qty{i}")
    tags.append(tag_name)
    qtys.append(tag_qty)

st.markdown("---")

if st.button("🚀 Generate Plan"):
    demand = {t: q for t, q in zip(tags, qtys) if q > 0}
    if not demand:
        st.error("কমপক্ষে ১টা Tag Quantity দিন।")
        st.stop()

    # Calculate
    layout = ratio_single_plate_layout(demand, capacity)
    sheets, produced, over = greedy_sheet_plan(demand, layout)

    st.success(f"✅ মোট প্রয়োজনীয় শিট: {sheets}")

    # Show layout
    st.markdown("### 🧩 Plate Layout Suggestion (per sheet)")
    df_layout = pd.DataFrame(list(layout.items()), columns=["Tag", "Per Plate"])
    st.dataframe(df_layout, use_container_width=True)

    # Balance
    st.markdown("### 📊 Production Summary")
    df_balance = pd.DataFrame([
        {"Tag": sz, "Demand": demand[sz], "Produced": produced.get(sz, 0), "Over/Short": over[sz]}
        for sz in demand
    ])
    st.dataframe(df_balance, use_container_width=True)

    # Excel export
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_layout.to_excel(writer, sheet_name="Plate Layout", index=False)
        df_balance.to_excel(writer, sheet_name="Summary", index=False)
    output.seek(0)
    st.download_button("⬇️ Download Excel Report", data=output, file_name="plate_plan.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("💡 এই অ্যাপ স্বয়ংক্রিয়ভাবে demand অনুযায়ী plate layout তৈরি করে এবং মোট impression সংখ্যা বের করে।")
