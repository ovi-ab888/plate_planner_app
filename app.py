# app.py
import streamlit as st
import pandas as pd
from collections import Counter
from io import BytesIO

st.set_page_config(page_title="Pre-Press Plate Planner", page_icon="🖨️", layout="wide")

# =========================
# Helper functions
# =========================
def normalize_layout(layout: dict, cap: int):
    """Ensure a plate layout sums to <= cap (12-up). If above, trim smallest first."""
    total = sum(int(v) for v in layout.values() if pd.notna(v))
    # Remove zeros/NaNs
    res = {k: int(v) for k, v in layout.items() if pd.notna(v) and int(v) > 0}
    if total <= cap:
        return res

    # Trim smallest contributors first to reach cap
    overflow = total - cap
    for k, _ in sorted(res.items(), key=lambda kv: kv[1]):  # small -> big
        if overflow <= 0:
            break
        take = min(res[k], overflow)
        res[k] -= take
        if res[k] <= 0:
            del_keys = [kk for kk, vv in res.items() if vv <= 0]
            for dk in del_keys:
                res.pop(dk, None)
        overflow -= take

    # if still above (rare), keep only largest until cap
    cur = sum(res.values())
    if cur > cap:
        out = {}
        used = 0
        for k, v in sorted(res.items(), key=lambda kv: kv[1], reverse=True):
            take = min(v, cap - used)
            if take > 0:
                out[k] = take
                used += take
            if used == cap:
                break
        res = out
    return res


def ratio_single_plate_layout(demand: dict, cap: int):
    """Build 1 plate layout by proportional allocation to demand."""
    total = sum(demand.values())
    if total == 0:
        return {}
    raw = {k: (demand[k] * cap) / total for k in demand}
    floored = {k: int(raw[k]) for k in raw}
    used = sum(floored.values())
    remain = cap - used

    # largest fractional parts get the remaining slots
    fracs = sorted(((k, raw[k] - floored[k]) for k in raw), key=lambda x: x[1], reverse=True)
    layout = dict(floored)
    for i in range(remain):
        if i < len(fracs):
            layout[fracs[i][0]] += 1

    # clean zeros and normalize to cap
    layout = {k: v for k, v in layout.items() if v > 0}
    return normalize_layout(layout, cap)


def greedy_sheet_plan(demand: dict, layouts: list):
    """
    demand: dict size->qty
    layouts: list of dicts (each plate), values = ups per sheet for that size
    Returns:
      - sheet_counts: list of ints per plate
      - remaining: demand left (should be zeros or small)
      - produced: dict size->total produced
      - overage: dict size->overproduction count
    Greedy = each step choose plate that covers the most still-needed units.
    """
    remaining = dict(demand)
    sheet_counts = [0] * len(layouts)

    def covered_by(layout, remaining):
        return sum(min(remaining.get(sz, 0), cnt) for sz, cnt in layout.items())

    # loop until all demands are met or no plate covers anything
    safe_guard = 10_000  # prevent infinite loop
    while any(v > 0 for v in remaining.values()) and safe_guard > 0:
        coverages = [covered_by(layouts[i], remaining) for i in range(len(layouts))]
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

    all_sizes = set(list(demand.keys()) + list(produced.keys()))
    overage = {sz: max(0, produced.get(sz, 0) - demand.get(sz, 0)) for sz in all_sizes}

    return sheet_counts, remaining, dict(produced), overage


def build_excel(auto_objs=None, custom_objs=None):
    """
    auto_objs/custom_objs = dict of dataframes to export.
    Returns: bytes of an .xlsx file.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        if auto_objs:
            for name, df in auto_objs.items():
                df.to_excel(writer, sheet_name=f"Auto - {name[:28]}", index=False)
        if custom_objs:
            for name, df in custom_objs.items():
                df.to_excel(writer, sheet_name=f"Custom - {name[:26]}", index=False)
    output.seek(0)
    return output


# =========================
# UI
# =========================
st.title("🖨️ Pre-Press Plate Planner")
st.caption("Tag size & QTY দিন, নিচে Auto বা Custom plate পরিকল্পনা দেখুন।")
colL, colR = st.columns([2,1])

with colL:
    sizes_text = st.text_input("Tag sizes (comma separated):", "XS,S,M,L,XL,XXL,3XL")
    qtys_text  = st.text_input("Quantities (comma separated):", "100,100,150,150,100,100,50")
with colR:
    capacity = st.number_input("Plate capacity (tags per plate):", min_value=1, max_value=64, value=12, step=1)

sizes = [s.strip() for s in sizes_text.split(",") if s.strip()]
qtys  = [q.strip() for q in qtys_text.split(",") if q.strip()]
if len(qtys) != len(sizes):
    st.error("সাইজ আর QTY সংখ্যায় মিল নেই।")
    st.stop()

try:
    demand = {s: int(q) for s, q in zip(sizes, qtys)}
except:
    st.error("QTY গুলো integer দিন।")
    st.stop()

tabs = st.tabs(["⚙️ Auto (1 plate suggestion)", "🧩 Custom plates (A/B/C...)"])

# =========================
# TAB 1: Auto mode
# =========================
with tabs[0]:
    st.subheader("Auto plate suggestion (single layout)")
    auto_plate = ratio_single_plate_layout(demand, capacity)
    st.write("Proposed 1-plate layout (per sheet):")
    st.dataframe(pd.DataFrame([auto_plate]).fillna(0).astype(int))

    # plan with only this plate
    auto_layouts = [auto_plate]
    auto_counts, auto_remaining, auto_produced, auto_over = greedy_sheet_plan(demand, auto_layouts)

    # Details
    details_rows = [{
        "Plate": "Plate 1",
        "Layout (size:ups)": ", ".join(f"{k}:{v}" for k, v in auto_plate.items()),
        "Sheets (impressions)": auto_counts[0],
    }]
    df_details = pd.DataFrame(details_rows)

    # Balance
    df_balance = pd.DataFrame([{
        "Size": sz,
        "Demand": demand[sz],
        "Produced": auto_produced.get(sz, 0),
        "Over/Short": auto_produced.get(sz, 0) - demand[sz],
    } for sz in sizes])

    # Summary
    df_summary = pd.DataFrame([{
        "Distinct Plates": 1,
        "Total Sheets (all plates)": sum(auto_counts),
    }])

    st.markdown("### 📋 Plate details")
    st.dataframe(df_details)

    st.markdown("### 🔢 Size balance")
    st.dataframe(df_balance)

    st.markdown("### ✅ Summary")
    st.dataframe(df_summary)

    # Excel download
    auto_xlsx = build_excel(
        auto_objs={
            "Summary": df_summary,
            "Plate layout": pd.DataFrame([auto_plate]).fillna(0).astype(int),
            "Plate details": df_details,
            "Size balance": df_balance,
        }
    )
    st.download_button("⬇️ Download Excel (Auto)", data=auto_xlsx, file_name="plate_planner_auto.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# =========================
# TAB 2: Custom plates
# =========================
with tabs[1]:
    st.subheader("Define your plates (A/B/C...)")

    # Make an editable grid where rows are plates and columns are sizes
    # Pre-fill two plates as an example
    template_rows = 2
    init = { "Plate": [ "A", "B" ] + [f"{chr(65+i)}" for i in range(2, template_rows)] }
    # Build a starting DF with zeros
    df_init = pd.DataFrame({"Plate": ["A", "B"]})
    for s in sizes:
        df_init[s] = 0
    df_init.loc[0, sizes[:min(len(sizes), 6)]] = 2  # light prefill for convenience

    st.caption(f"প্রতি প্লেটের সাইজ-আপ (প্রতি শিট) লিখুন। প্রত্যেক প্লেটের sum ≤ {capacity} রাখবেন।")
    edited = st.data_editor(
        df_init,
        num_rows="dynamic",
        use_container_width=True,
        key="plate_editor",
    )

    if st.button("Generate Custom Plan"):
        # Build layouts from edited grid
        layouts = []
        names = []
        for _, row in edited.iterrows():
            plate_name = str(row["Plate"]).strip() if pd.notna(row["Plate"]) and str(row["Plate"]).strip() else f"P{_+1}"
            layout_raw = {s: int(row.get(s, 0)) for s in sizes if pd.notna(row.get(s, 0)) and int(row.get(s, 0)) > 0}
            layout = normalize_layout(layout_raw, capacity)
            if sum(layout.values()) == 0:
                continue
            layouts.append(layout)
            names.append(plate_name)

        if not layouts:
            st.warning("কোনো বৈধ plate layout পাওয়া যায়নি। টেবিলে সংখ্যা দিন।")
            st.stop()

        # Plan
        counts, remaining, produced, over = greedy_sheet_plan(demand, layouts)

        # Details table (per plate)
        rows = []
        for i, layout in enumerate(layouts):
            rows.append({
                "Plate": f"Plate {names[i]}",
                "Layout (size:ups)": ", ".join(f"{k}:{v}" for k, v in layout.items()),
                "Sheets (impressions)": counts[i],
            })
        df_details_c = pd.DataFrame(rows)

        # Balance (per size)
        df_balance_c = pd.DataFrame([{
            "Size": sz,
            "Demand": demand[sz],
            "Produced": produced.get(sz, 0),
            "Over/Short": produced.get(sz, 0) - demand[sz],
        } for sz in sizes])

        # Summary
        df_summary_c = pd.DataFrame([{
            "Distinct Plates": len(layouts),
            "Total Sheets (all plates)": sum(counts),
        }])

        # Pretty show
        st.markdown("### 📋 Plate details (Custom)")
        st.dataframe(df_details_c, use_container_width=True)

        st.markdown("### 🔢 Size balance (Custom)")
        st.dataframe(df_balance_c, use_container_width=True)

        st.markdown("### ✅ Summary (Custom)")
        st.dataframe(df_summary_c, use_container_width=True)

        # Excel
        custom_xlsx = build_excel(
            custom_objs={
                "Summary": df_summary_c,
                "Plate details": df_details_c,
                "Size balance": df_balance_c,
            }
        )
        st.download_button("⬇️ Download Excel (Custom)", data=custom_xlsx, file_name="plate_planner_custom.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Footer help
st.markdown("---")
st.caption("Tip: Auto mode এক প্লেটের প্রোপোরশনাল সাজেশন দেয়। Custom mode-এ Plate A/B/C নিজের মতো দিন—অ্যাপ বলবে কোন প্লেটে কোন সাইজ কতটা থাকবে এবং প্রতিটি প্লেটের শিট (impressions) কত লাগবে।")
