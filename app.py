# app.py — Final Pre-Press Planner (Safe loop + Multi-plate + Excel Export)
import streamlit as st
import pandas as pd
from collections import Counter
from io import BytesIO

st.set_page_config(page_title="Pre-Press Plate Planner", page_icon="🖨️", layout="wide")

# =========================
# Helper Functions
# =========================
def normalize_layout(layout: dict, cap: int):
    """Ensure layout sums <= cap."""
    layout = {k: int(v) for k, v in layout.items() if int(v) > 0}
    total = sum(layout.values())
    if total <= cap:
        return layout
    # Trim smallest values
    overflow = total - cap
    for k, v in sorted(layout.items(), key=lambda x: x[1]):
        if overflow <= 0:
            break
        take = min(v, overflow)
        layout[k] -= take
        overflow -= take
    return {k: v for k, v in layout.items() if v > 0}


def ratio_single_plate_layout(demand: dict, cap: int):
    """Auto proportional layout for one plate."""
    total = sum(demand.values())
    if total == 0:
        return {}
    raw = {k: (demand[k] * cap) / total for k in demand}
    floored = {k: int(raw[k]) for k in raw}
    used = sum(floored.values())
    remain = cap - used
    fracs = sorted(((k, raw[k] - floored[k]) for k in raw), key=lambda x: x[1], reverse=True)
    for i in range(remain):
        if i < len(fracs):
            floored[fracs[i][0]] += 1
    return normalize_layout(floored, cap)


def greedy_multi_plan(demand: dict, layouts: list, names: list):
    """Greedy planning for multiple plates with safe loop."""
    remaining = dict(demand)
    sheet_counts = [0] * len(layouts)
    safe_guard = 10000

    def coverage(layout, remaining):
        return sum(min(remaining.get(sz, 0), cnt) for sz, cnt in layout.items())

    while any(v > 0 for v in remaining.values()) and safe_guard > 0:
        coverages = [coverage(layouts[i], remaining) for i in range(len(layouts))]
        best_i = max(range(len(layouts)), key=lambda i: coverages[i])
        if coverages[best_i] == 0:
            break
        sheet_counts[best_i] += 1
        for sz, cnt in layouts[best_i].items():
            if remaining.get(sz, 0) > 0:
                remaining[sz] = max(0, remaining[sz] - cnt)
        safe_guard -= 1

    produced = Counter()
    for i, layout in enumerate(layouts):
        for sz, cnt in layout.items():
            produced[sz] += cnt * sheet_counts[i]

    overage = {sz: produced.get(sz, 0) - demand.get(sz, 0) for sz in demand}
    return sheet_counts, dict(produced), overage


# =========================
# UI STEP 1 — Tag inputs
# =========================
st.title("🖨️ Pre-Press Plate Planner (Multi-Plate Version)")
st.caption("সহজভাবে Tag Quantity, Plate Capacity ও Plate Layout দিয়ে প্রিন্ট প্ল্যান হিসাব করুন।")

col1, col2 = st.columns(2)
num_tags = col1.number_input("কতটি Tag QTY ইনপুট দিতে চান?", 1, 50, 6)
capacity = col2.number_input("Plate capacity (tags per plate)", 1, 64, 12)

st.markdown("---")
st.subheader("📦 Tag QTY ইনপুট দিন 👇")

tags, qtys = [], []
cols = st.columns(2)
for i in range(num_tags):
    tag_name = cols[0].text_input(f"Tag {i+1} Name", f"Tag {i+1}", key=f"tag{i}")
    tag_qty = cols[1].number_input(f"{tag_name} Quantity", min_value=0, step=10, key=f"qty{i}")
    tags.append(tag_name)
    qtys.append(tag_qty)

demand = {t: q for t, q in zip(tags, qtys) if q > 0}

st.markdown("---")

# =========================
# UI STEP 2 — Plate Layouts
# =========================
st.subheader("🎨 Plate Layout নির্ধারণ করুন (A/B/C...)")
st.caption(f"প্রতি Plate-এ সাইজ-আপ (প্রতি শিট) লিখুন। প্রত্যেক Plate এর sum ≤ {capacity} রাখবেন।")

# Editable table
plate_init = pd.DataFrame({"Plate": ["A", "B"], **{t: [0, 0] for t in demand.keys()}})
edited = st.data_editor(plate_init, num_rows="dynamic", use_container_width=True, key="layout_table")

# =========================
# GENERATE PLAN
# =========================
if st.button("🚀 Generate Plan"):
    if not demand:
        st.error("কমপক্ষে ১টি Tag Quantity দিন।")
        st.stop()

    # Build layouts
    layouts, names = [], []
    for _, row in edited.iterrows():
        pname = str(row["Plate"]).strip() if pd.notna(row["Plate"]) else f"Plate{_+1}"
        lay = {t: int(row.get(t, 0)) for t in demand.keys() if int(row.get(t, 0)) > 0}
        if lay:
            layouts.append(normalize_layout(lay, capacity))
            names.append(pname)

    if not layouts:
        st.warning("কোনো বৈধ Plate Layout পাওয়া যায়নি।")
        st.stop()

    # Calculate
    sheets, produced, over = greedy_multi_plan(demand, layouts, names)
    total_sheets = sum(sheets)

    st.success(f"✅ মোট প্রয়োজনীয় শিট: {total_sheets}")

    # =========================
    # OUTPUT TABLES
    # =========================
    plate_rows = []
    for i, layout in enumerate(layouts):
        plate_rows.append({
            "Plate": f"Plate {names[i]}",
            "Layout (tag:ups)": ", ".join(f"{k}:{v}" for k, v in layout.items()),
            "Sheets (impressions)": sheets[i]
        })
    df_plate = pd.DataFrame(plate_rows)

    df_balance = pd.DataFrame([
        {"Tag": t, "Demand": demand[t], "Produced": produced.get(t, 0), "Over/Short": over[t]}
        for t in demand
    ])

    df_summary = pd.DataFrame([{
        "Distinct Plates": len(layouts),
        "Total Sheets": total_sheets
    }])

    st.markdown("### 🧾 Plate Details")
    st.dataframe(df_plate, use_container_width=True)

    st.markdown("### 📊 Size Balance")
    st.dataframe(df_balance, use_container_width=True)

    st.markdown("### ✅ Summary")
    st.dataframe(df_summary, use_container_width=True)

    # =========================
    # EXCEL EXPORT
    # =========================
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_plate.to_excel(writer, sheet_name="Plate Details", index=False)
        df_balance.to_excel(writer, sheet_name="Size Balance", index=False)
    output.seek(0)

    st.download_button(
        "⬇️ Download Excel Report",
        data=output,
        file_name="plate_plan_multi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.markdown("---")
st.caption("💡 এই অ্যাপ আপনার দেওয়া Tag QTY ও Plate layout অনুযায়ী মোট impression সংখ্যা ও sheet requirement হিসাব করে।")
