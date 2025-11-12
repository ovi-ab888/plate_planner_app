# app.py — Auto Multi-Plate Pre-Press Planner
import streamlit as st
import pandas as pd
from collections import Counter
from io import BytesIO
import string
from math import ceil

st.set_page_config(page_title="Pre-Press Plate Planner (Auto)", page_icon="🖨️", layout="wide")

# =========================
# Helpers
# =========================
def plate_name(n: int) -> str:
    """Return A, B, ... Z, AA, AB ... style names (1-indexed)."""
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
    """Ensure integers, positive, and sum exactly <= cap (trim smallest if needed).
       If sum < cap, keep as-is (we allow partial fill)."""
    clean = {k: int(v) for k, v in layout.items() if int(v) > 0}
    total = sum(clean.values())
    if total <= cap:
        return clean
    overflow = total - cap
    for k, v in sorted(clean.items(), key=lambda kv: kv[1]):  # trim smallest first
        if overflow <= 0:
            break
        take = min(v, overflow)
        clean[k] -= take
        overflow -= take
        if clean[k] <= 0:
            clean.pop(k, None)
    return {k: v for k, v in clean.items() if v > 0}

def proportional_layout(remaining: dict, cap: int) -> dict:
    """Build a plate layout proportional to remaining demand (tries to use all cap slots)."""
    total = sum(remaining.values())
    if total == 0:
        return {}
    raw = {k: (remaining[k] * cap) / total for k in remaining if remaining[k] > 0}
    floored = {k: int(raw[k]) for k in raw}
    used = sum(floored.values())
    # Guarantee at least 1 for any size that had 0 after floor but still remaining (if room)
    zeros = [k for k in raw if floored[k] == 0]
    for k in zeros:
        if used < cap:
            floored[k] = 1
            used += 1
    # Distribute leftover slots by largest fractions
    remain_slots = max(0, cap - used)
    fracs = sorted(((k, raw[k] - floored[k]) for k in raw), key=lambda x: x[1], reverse=True)
    for i in range(remain_slots):
        if i < len(fracs):
            floored[fracs[i][0]] += 1
    layout = {k: v for k, v in floored.items() if v > 0}
    return normalize_layout(layout, cap)

def auto_multi_plate_plan(demand: dict, cap: int):
    """
    Build a sequence of plates automatically until demand becomes zero.
    Each plate gets a proportional per-sheet layout from remaining demand.
    For each plate, pick the max number of sheets we can print without
    overshooting any size in that plate (at least 1 sheet to make progress).
    """
    remaining = {k: int(v) for k, v in demand.items() if int(v) > 0}
    plates = []  # list of dicts: {"name": "A", "layout": {...}, "sheets": N}
    safe_guard = 10000

    while any(v > 0 for v in remaining.values()) and safe_guard > 0:
        layout = proportional_layout(remaining, cap)
        if not layout:
            break

        # max sheets we can print before any included tag would go negative
        possible = []
        for sz, ups in layout.items():
            if ups <= 0:
                continue
            # how many full sheets can we print for this size
            possible.append(remaining[sz] // ups)
        sheets = max(1, min(possible) if possible else 1)

        # If min floor is 0 (i.e., at least one size has remaining < ups), still print 1 sheet to progress
        # but cap the case where that would overshoot too much: we allow small overage only at the end.
        # Subtract and continue:
        for sz, ups in layout.items():
            remaining[sz] = max(0, remaining[sz] - ups * sheets)

        plates.append({"layout": layout, "sheets": sheets})
        safe_guard -= 1

        # If we are making very slow progress (e.g., all floors were 0 repeatedly),
        # force a final plate with sheets = ceil(max(remaining)/ups) to finish.
        if not any(v > 0 for v in remaining.values()):
            break

        if safe_guard == 9990:  # extremely unlikely; just a second safety fuse
            # Finish in one last plate using the same layout:
            # choose sheets so that at least one size finishes
            need_ratios = []
            for sz, ups in layout.items():
                if ups > 0 and remaining[sz] > 0:
                    need_ratios.append(ceil(remaining[sz] / ups))
            if need_ratios:
                extra_sheets = max(1, min(need_ratios))
                for sz, ups in layout.items():
                    remaining[sz] = max(0, remaining[sz] - ups * extra_sheets)
                plates.append({"layout": layout, "sheets": extra_sheets})
            break

    # Assign plate names A, B, C...
    for i, p in enumerate(plates, start=1):
        p["name"] = plate_name(i)

    # Produced summary (not strictly needed for output, but handy)
    produced = Counter()
    for p in plates:
        for sz, ups in p["layout"].items():
            produced[sz] += ups * p["sheets"]

    return plates, dict(produced)

# =========================
# UI — Inputs
# =========================
st.title("🖨️ Pre-Press Plate Planner — Auto Multi-Plate")
st.caption("Tag count, Plate capacity, Tag name + QTY দিন। অ্যাপ নিজে Plate A/B/C… বানাবে এবং প্রতি Plate-এর per-sheet layout ও impressions দেখাবে।")

c1, c2 = st.columns(2)
num_tags = c1.number_input("কতটি Tag QTY ইনপুট দিতে চান?", min_value=1, max_value=50, value=6, step=1)
capacity = c2.number_input("Plate capacity (tags per plate)", min_value=1, max_value=64, value=12, step=1)

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

    plates, produced = auto_multi_plate_plan(demand, capacity)

    if not plates:
        st.warning("পরিকল্পনা তৈরি করা যায়নি। ইনপুটগুলো যাচাই করুন।")
        st.stop()

    # Build per-plate table: columns = Plate, all tags..., Sheets
    cols = ["Plate"] + list(demand.keys()) + ["Sheets (impressions)"]
    rows = []
    for p in plates:
        row = {"Plate": p["name"], "Sheets (impressions)": p["sheets"]}
        for t in demand.keys():
            row[t] = p["layout"].get(t, 0)
        rows.append(row)
    df_plate = pd.DataFrame(rows, columns=cols)

    total_sheets = int(sum(p["sheets"] for p in plates))

    st.markdown("### 🧾 প্রতি Plate-এ সাইজ-আপ (প্রতি শিট) + ইমপ্রেশন")
    st.dataframe(df_plate, use_container_width=True)
    st.success(f"✅ মোট শিট (সব প্লেট মিলিয়ে): {total_sheets}")

    # Excel export
    xout = BytesIO()
    with pd.ExcelWriter(xout, engine="openpyxl") as writer:
        df_plate.to_excel(writer, sheet_name="Per-Plate Layout & Sheets", index=False)
        # Optional: add demand/produced for your records
        pd.DataFrame([{"Tag": k, "Demand": demand[k], "Produced": produced.get(k, 0)} for k in demand])\
            .to_excel(writer, sheet_name="Demand vs Produced", index=False)
    xout.seek(0)
    st.download_button(
        "⬇️ Download Excel",
        data=xout,
        file_name="auto_multi_plate_plan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.caption("💡 সম্পূর্ণ অটো: Plate A/B/C... নিজে তৈরি হয়। প্রতিটি Plate-এর per-sheet layout ও sheet (impressions) সংখ্যা দেখানো হয়।")
