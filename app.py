# app.py — IMPROVED VERSION (Your Manual Strategy)

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
    """Generate layout ensuring total == cap (same as before)"""
    total = sum(remaining.values())
    if total == 0:
        return {}

    layout = {k: int((remaining[k] / total) * cap) for k in remaining if remaining[k] > 0}

    # ensure every tag gets at least 1 if possible
    for k in layout:
        if layout[k] == 0 and remaining[k] > 0:
            layout[k] = 1

    # fill or trim to match exact capacity
    while sum(layout.values()) < cap:
        for k in sorted(remaining, key=lambda x: remaining[x], reverse=True):
            if sum(layout.values()) >= cap:
                break
            layout[k] += 1

    while sum(layout.values()) > cap:
        for k in sorted(layout, key=lambda x: layout[x], reverse=True):
            if sum(layout.values()) <= cap:
                break
            if layout[k] > 1:
                layout[k] -= 1

    return layout


def create_variation_layout(base_layout, remaining, cap):
    """
    আগের লেআউট থেকে ভিন্ন একটি লেআউট তৈরি করে
    (যাতে বিভিন্ন প্লেটে UPS ভিন্ন হয়)
    """
    if not base_layout:
        return proportional_layout(remaining, cap)
    
    # নতুন লেআউট বানাই - বেস লেআউট থেকে সামান্য পরিবর্তন
    new_layout = base_layout.copy()
    
    # কিছু UPS শিফট করি (যাতে ভিন্নতা আসে)
    items = list(new_layout.keys())
    if len(items) >= 2:
        # বেশি কোয়ান্টিটি যাদের, তাদের বাড়াই
        max_item = max(items, key=lambda x: remaining.get(x, 0))
        min_item = min(items, key=lambda x: new_layout.get(x, 1))
        
        if new_layout.get(min_item, 1) > 1:
            new_layout[min_item] = new_layout.get(min_item, 1) - 1
            new_layout[max_item] = new_layout.get(max_item, 1) + 1
    
    # ক্যাপাসিটি রি-অ্যাডজাস্ট
    while sum(new_layout.values()) < cap:
        for k in sorted(remaining, key=lambda x: remaining.get(x, 0), reverse=True):
            if sum(new_layout.values()) >= cap:
                break
            new_layout[k] = new_layout.get(k, 0) + 1
    
    while sum(new_layout.values()) > cap:
        for k in sorted(new_layout, key=lambda x: new_layout[x], reverse=True):
            if sum(new_layout.values()) <= cap:
                break
            if new_layout[k] > 1:
                new_layout[k] -= 1
    
    return new_layout


def improved_auto_plan(demand, cap, max_plates=3):
    """
    উন্নত ভার্সন - আপনার ম্যানুয়াল স্ট্রাটেজি ফলো করে
    প্রতিটি প্লেট আলাদা UPS প্যাটার্ন পায়
    """
    remaining = demand.copy()
    plates = []
    produced = Counter()
    prev_layout = None
    
    for i in range(max_plates):
        # if no demand left, stop
        if not any(v > 0 for v in remaining.values()):
            break
        
        # Create layout (different from previous if possible)
        if i == 0 or prev_layout is None:
            layout = proportional_layout(remaining, cap)
        else:
            layout = create_variation_layout(prev_layout, remaining, cap)
        
        if not layout:
            break
        
        # Calculate sheets for this plate
        # YOUR MANUAL STRATEGY: use MAX sheets (not MIN)
        # কারণ বেশি শিট নিলে পরবর্তী প্লেটের প্রয়োজন কমে, Overprint কমে
        sheets_needed = []
        for k, v in layout.items():
            if v > 0 and remaining.get(k, 0) > 0:
                sheets_needed.append(ceil(remaining[k] / v))
        
        sheets = max(sheets_needed) if sheets_needed else 1
        sheets = max(1, sheets)  # কমপক্ষে 1 শিট
        
        # Reduce remaining demand
        for k, v in layout.items():
            remaining[k] = max(0, remaining[k] - v * sheets)
            produced[k] += v * sheets
        
        plates.append({
            "name": plate_name(len(plates) + 1),
            "layout": layout,
            "sheets": sheets
        })
        
        prev_layout = layout
    
    # 🧪 Post-optimization: যদি এখনো কিছু বাকি থাকে, শেষ প্লেটে যোগ করো
    if any(v > 0 for v in remaining.values()) and plates:
        last_plate = plates[-1]
        for tag, rem in remaining.items():
            if rem > 0:
                per_sheet = last_plate["layout"].get(tag, 1)
                if per_sheet > 0:
                    add_sheets = ceil(rem / per_sheet)
                    last_plate["sheets"] += add_sheets
                    produced[tag] += add_sheets * per_sheet
                    remaining[tag] = 0
    
    return plates, dict(produced)


# ---------- UI ----------
st.title("🖨️ Pre-Press Auto Planner (Manual Strategy)")
st.caption("🤖 প্রতিটি প্লেট ভিন্ন UPS প্যাটার্ন পায় — Overprint কমানোর জন্য")

col1, col2, col3, col4 = st.columns(4)
n = col1.number_input("কতটি Tag", 1, 50, 6)
cap = col2.number_input("Plate capacity (tags per plate)", 1, 64, 30)  # Default 30
maxp = col3.number_input("কতটি Plate বানাতে চান", 1, 10, 2)
addon = col4.number_input("Add-on % (Extra print)", 0.0, 50.0, 0.0, step=0.5)

st.markdown("---")
st.subheader("📦 Tag QTY দিন")

l, r = st.columns(2)
tags, qty = [], []
for i in range(n):
    name = l.text_input(f"Tag {i+1}", f"Label {i+1}", key=f"t{i}")
    q = r.number_input(f"{name} Qty", 0, step=10, value=100, key=f"q{i}")
    tags.append(name)
    qty.append(q)

# Apply Add-on %
demand = {t: ceil(int(q) * (1 + addon / 100)) for t, q in zip(tags, qty) if q > 0}

if st.button("🚀 Generate Plan"):
    if not demand:
        st.error("কমপক্ষে ১টি Tag Quantity দিন।")
        st.stop()

    progress = st.progress(0, text="🔄 Calculating Plates with Manual Strategy...")
    plates, prod = improved_auto_plan(demand, cap, maxp)
    progress.progress(100, text="✅ Done!")

    if not plates:
        st.warning("পরিকল্পনা তৈরি হয়নি। ইনপুট যাচাই করুন।")
        st.stop()

    # ---------- Plate Layout ----------
    cols = ["Plate"] + list(demand.keys()) + ["Sheets"]
    rows = []
    for p in plates:
        row = {"Plate": p["name"], "Sheets": p["sheets"]}
        for t in demand.keys():
            row[t] = p["layout"].get(t, 0)
        rows.append(row)
    df = pd.DataFrame(rows, columns=cols)

    total_sheets = sum(p["sheets"] for p in plates)
    st.markdown("### 🧾 প্রতি Plate-এর Layout (Ups per sheet)")
    st.dataframe(df, use_container_width=True)
    st.success(f"✅ মোট শিট: {total_sheets}")

    # ---------- Summary ----------
    summary_data = []
    total_extra = 0
    total_target = 0
    total_produced = 0
    
    for k in demand:
        target = demand[k]
        produced_val = prod.get(k, 0)
        extra = produced_val - target
        total_extra += max(0, extra)
        total_target += target
        total_produced += produced_val
        
        overprint_pct = round((extra / (target - ceil(target * addon/100) if addon > 0 else target) * 100), 2) if target > 0 else 0
        summary_data.append({
            "Tag": k,
            "Original QTY": qty[tags.index(k)] if k in tags else 0,
            "Target(+Add-on)": target,
            "Produced": produced_val,
            "Extra(Overprint)": extra,
            "Overprint (%)": overprint_pct
        })
    
    summary = pd.DataFrame(summary_data)
    
    # Total row
    total_row = {
        "Tag": "📊 TOTAL",
        "Original QTY": sum([qty[tags.index(k)] for k in demand if k in tags]),
        "Target(+Add-on)": total_target,
        "Produced": total_produced,
        "Extra(Overprint)": total_extra,
        "Overprint (%)": round((total_extra / sum(qty) * 100), 2) if sum(qty) > 0 else 0
    }
    summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)
    
    st.markdown("### 📊 Demand vs Produced")
    st.dataframe(summary, use_container_width=True)
    
    # Final Verdict
    final_overprint_pct = total_row["Overprint (%)"]
    st.divider()
    
    if final_overprint_pct <= 2:
        st.success(f"✅ অসাধারণ! মাত্র {final_overprint_pct}% Overprint — আপনার ম্যানুয়াল স্ট্রাটেজির কাছাকাছি!")
        st.balloons()
    elif final_overprint_pct <= 5:
        st.info(f"📈 {final_overprint_pct}% Overprint — গ্রহণযোগ্য স্তরে।")
    elif final_overprint_pct <= 10:
        st.warning(f"⚠️ {final_overprint_pct}% Overprint — আরও কমানো যায়। প্লেট সংখ্যা বাড়ানোর চেষ্টা করুন।")
    else:
        st.error(f"❌ {final_overprint_pct}% Overprint — খুব বেশি! প্লেট সংখ্যা বাড়ান বা ক্যাপাসিটি চেক করুন।")
    
    # ---------- Excel Export ----------
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Plates", index=False)
        summary.to_excel(w, sheet_name="Summary", index=False)
    bio.seek(0)

    st.download_button(
        "⬇️ Excel Download",
        data=bio,
        file_name="optimized_plate_plan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.caption("💡 উন্নত ভার্সন: প্রতিটি প্লেটে ভিন্ন UPS প্যাটার্ন + MAX sheets স্ট্রাটেজি → Overprint কম")
