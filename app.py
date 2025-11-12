# app.py — Simplified Output: Per-Plate Size-ups + Impressions
import streamlit as st
import pandas as pd
from collections import Counter
from io import BytesIO

st.set_page_config(page_title="Pre-Press Plate Planner", page_icon="🖨️", layout="wide")

# =============== Helpers ===============
def normalize_layout(layout: dict, cap: int):
    """Keep only positive ints and ensure sum <= cap by trimming the smallest first."""
    clean = {k: int(v) for k, v in layout.items() if pd.notna(v) and int(v) > 0}
    total = sum(clean.values())
    if total <= cap:
        return clean, False  # False = not trimmed
    overflow = total - cap
    # trim smallest first
    for k, v in sorted(clean.items(), key=lambda kv: kv[1]):
        if overflow <= 0:
            break
        take = min(v, overflow)
        clean[k] -= take
        overflow -= take
        if clean[k] <= 0:
            clean.pop(k, None)
    return {k: v for k, v in clean.items() if v > 0}, True  # True = trimmed

def greedy_multi_plan(demand: dict, layouts: list):
    """Greedy planning with safe loop. Returns per-plate sheet counts + produced."""
    remaining = dict(demand)
    counts = [0] * len(layouts)
    safe_guard = 10000

    def coverage(layout, remaining):
        return sum(min(remaining.get(sz, 0), up) for sz, up in layout.items())

    while any(v > 0 for v in remaining.values()) and safe_guard > 0:
        coverages = [coverage(lay, remaining) for lay in layouts]
        best = max(range(len(layouts)), key=lambda i: coverages[i])
        if coverages[best] == 0:
            break  # nothing else covered
        counts[best] += 1
        for sz, up in layouts[best].items():
            if remaining.get(sz, 0) > 0:
                remaining[sz] = max(0, remaining[sz] - up)
        safe_guard -= 1

    produced = Counter()
    for i, lay in enumerate(layouts):
        for sz, up in lay.items():
            produced[sz] += up * counts[i]

    return counts, dict(produced)

# =============== UI — Step 1: Basic inputs ===============
st.title("🖨️ Pre-Press Plate Planner (Multi-Plate)")
st.caption("শুরুতে Tag সংখ্যা, Plate capacity, তারপর Tag নাম + QTY দিন। ফলাফল: প্রতি প্লেটের সাইজ-আপ ও ইমপ্রেশন।")

col1, col2 = st.columns(2)
num_tags = col1.number_input("কতটি Tag QTY ইনপুট দিতে চান?", 1, 50, 6, step=1)
capacity = col2.number_input("Plate capacity (tags per plate)", 1, 64, 12, step=1)

st.markdown("---")
st.subheader("📦 Tag QTY ইনপুট দিন 👇")
left, right = st.columns(2)
tags, qtys = [], []
for i in range(num_tags):
    tname = left.text_input(f"Tag {i+1} Name", f"Tag {i+1}", key=f"tag{i}")
    tqty  = right.number_input(f"{tname} Quantity", min_value=0, step=10, key=f"qty{i}")
    tags.append(tname)
    qtys.append(tqty)

# Demand dict (exclude zeros)
demand = {t: int(q) for t, q in zip(tags, qtys) if int(q) > 0}

st.markdown("---")

# =============== UI — Step 2: Plate layouts table ===============
st.subheader("🎨 প্রতি Plate-এ সাইজ-আপ (প্রতি শিট) লিখুন")
st.caption(f"প্রতি প্লেটের সাইজ-আপের যোগফল ≤ {capacity} রাখুন। (শিট/ইমপ্রেশন পরে ক্যালকুলেট হবে)")

# Build editable grid: start with two plates A, B; columns are the active tags
plate_df_init = pd.DataFrame({"Plate": ["A", "B"]})
for t in demand.keys():
    plate_df_init[t] = 0

plate_df = st.data_editor(
    plate_df_init,
    num_rows="dynamic",
    use_container_width=True,
    key="plate_editor",
)

st.markdown("---")

# =============== Generate Plan ===============
if st.button("🚀 Generate Plan"):
    if not demand:
        st.error("কমপক্ষে ১টি Tag Quantity দিন।")
        st.stop()

    # Build normalized layouts (respect capacity)
    layouts, names, trimmed_any = [], [], False
    for _, row in plate_df.iterrows():
        pname = str(row["Plate"]).strip() if pd.notna(row["Plate"]) and str(row["Plate"]).strip() else f"P{_+1}"
        raw = {t: int(row.get(t, 0)) for t in demand.keys() if pd.notna(row.get(t, 0)) and int(row.get(t, 0)) > 0}
        if not raw:
            continue
        norm, trimmed = normalize_layout(raw, capacity)
        trimmed_any = trimmed_any or trimmed
        layouts.append(norm)
        names.append(pname)

    if not layouts:
        st.warning("কোনো বৈধ Plate Layout পাওয়া যায়নি। টেবিলে মান দিন।")
        st.stop()

    if trimmed_any:
        st.info("ℹ️ কিছু প্লেটে capacity ছাড়ালে ছোট মান কেটে capacity-এর মধ্যে নামিয়ে আনা হয়েছে।")

    # Plan
    sheet_counts, produced = greedy_multi_plan(demand, layouts)

    # =============== OUTPUT (Only what you asked) ===============
    # 1) Per-plate size-ups (per sheet) in a single table, plus a ShEETS column
    # Build table with columns: Plate, each Tag..., Sheets
    out_cols = ["Plate"] + list(demand.keys()) + ["Sheets (impressions)"]
    table_rows = []
    for i, lay in enumerate(layouts):
        row = {"Plate": f"{names[i]}"}
        for t in demand.keys():
            row[t] = lay.get(t, 0)
        row["Sheets (impressions)"] = sheet_counts[i]
        table_rows.append(row)
    plate_table = pd.DataFrame(table_rows, columns=out_cols)

    st.markdown("### 🧾 প্রতি Plate-এ সাইজ-আপ (প্রতি শিট) + ইমপ্রেশন")
    st.dataframe(plate_table, use_container_width=True)

    # Optional: total sheets headline
    total_sheets = int(sum(sheet_counts))
    st.success(f"✅ মোট শিট (সব প্লেট মিলিয়ে): {total_sheets}")

    # =============== Excel export (same info) ===============
    xout = BytesIO()
    with pd.ExcelWriter(xout, engine="openpyxl") as writer:
        plate_table.to_excel(writer, sheet_name="Per-Plate Layout & Sheets", index=False)
    xout.seek(0)
    st.download_button(
        "⬇️ Download Excel (Per-Plate result)",
        data=xout,
        file_name="per_plate_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.caption("💡 উপরের টেবিলেই দেখা যাবে: A/B/C... Plate-এ প্রতি শিটে কোন ট্যাগ কতটা এবং প্রত্যেক Plate-এর শিট (impressions) সংখ্যা।")
