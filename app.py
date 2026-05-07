import streamlit as st
import math
import pandas as pd
import random

st.set_page_config(page_title="Pre-Press Optimizer Pro", layout="wide")

st.title("🎨 Pre-Press Grid & Plate Optimizer (Offline Mode)")
st.caption("⚙️ No API Key Needed - Smart Manual Optimization")

# --- Sidebar ---
st.sidebar.header("⚙️ কনফিগারেশন")
grid_size = st.sidebar.number_input("একটি প্লেটে মোট কয়টি লেবেল (Grid Size)?", min_value=1, value=30)
extra_percent = st.sidebar.number_input("অ্যাড-অন (Extra %) কত হবে?", min_value=0.0, value=0.0)
num_labels = st.sidebar.number_input("মোট কত পদের লেবেল?", min_value=1, value=1, step=1)
num_plates = st.sidebar.number_input("কয়টি প্লেট ব্যবহার করবেন?", min_value=1, value=2, step=1)

# --- Data Input ---
st.subheader("📦 লেবেল কোয়ান্টিটি ইনপুট")
labels_input = []
cols = st.columns(2)

for i in range(int(num_labels)):
    col_idx = i % 2
    with cols[col_idx]:
        c1, c2 = st.columns([2, 1])
        name = c1.text_input(f"নাম {i+1}", value=f"Label {i+1}", key=f"n_{i}")
        qty = c2.number_input(f"QTY", min_value=1, value=100, key=f"q_{i}")
        target = math.ceil(qty * (1 + extra_percent / 100))
        labels_input.append({"Name": name, "Original QTY": qty, "Target QTY": target})

# ============================================
# OPTIMIZED MANUAL ALGORITHM
# ============================================

def optimize_plates_manual(targets, grid_size, num_plates):
    num_items = len(targets)
    total_target = sum(targets)
    
    # Base UPS Calculation (Proportional)
    base_ups = []
    for i in range(num_items):
        ups = max(1, round((targets[i] / total_target) * grid_size))
        base_ups.append(ups)
    
    # Adjust base ups to match grid size
    diff = grid_size - sum(base_ups)
    if diff > 0:
        for _ in range(diff):
            max_idx = max(range(num_items), key=lambda i: targets[i])
            base_ups[max_idx] += 1
            
    # Create variations for each plate
    plate_ups_list = []
    for p in range(num_plates):
        if p == 0:
            plate_ups_list.append(base_ups.copy())
        else:
            new_ups = base_ups.copy()
            # Smart redistribution to create variation
            for _ in range(random.randint(1, 3)):
                candidates = [i for i in range(num_items) if new_ups[i] > 1]
                if candidates:
                    src = random.choice(candidates)
                    tgt = random.randint(0, num_items-1)
                    if new_ups[src] > 1:
                        new_ups[src] -= 1
                        new_ups[tgt] += 1
            
            # Re-adjust sum
            curr_sum = sum(new_ups)
            if curr_sum != grid_size:
                diff2 = grid_size - curr_sum
                for _ in range(abs(diff2)):
                    idx = max(range(num_items), key=lambda i: targets[i])
                    new_ups[idx] += 1 if diff2 > 0 else -1
            plate_ups_list.append(new_ups)
    
    # Calculate Sheets per plate (Variable Sheets)
    remaining = targets.copy()
    sheets_list = []
    final_produced = [0] * num_items
    
    for p in range(num_plates):
        if sum(remaining) <= 0:
            sheets_list.append(0)
            continue
            
        max_sheets = 0
        for i in range(num_items):
            if plate_ups_list[p][i] > 0 and remaining[i] > 0:
                needed = math.ceil(remaining[i] / plate_ups_list[p][i])
                max_sheets = max(max_sheets, needed)
        
        # Optimization: If this is last plate, we might need to adjust
        if p == num_plates - 1 and sum(remaining) > 0:
            max_sheets = max(max_sheets, math.ceil(max(remaining) / max(plate_ups_list[p])))
            
        sheets_list.append(max_sheets)
        
        for i in range(num_items):
            produced = plate_ups_list[p][i] * max_sheets
            final_produced[i] += produced
            remaining[i] = max(0, remaining[i] - produced)
    
    # If still remaining, add one more run to last plate
    if sum(remaining) > 0 and num_plates > 0:
        extra_sheets = 0
        last_plate = num_plates - 1
        for i in range(num_items):
            if remaining[i] > 0 and plate_ups_list[last_plate][i] > 0:
                needed = math.ceil(remaining[i] / plate_ups_list[last_plate][i])
                extra_sheets = max(extra_sheets, needed)
        
        if extra_sheets > sheets_list[last_plate]:
            add_sheets = extra_sheets - sheets_list[last_plate]
            sheets_list[last_plate] = extra_sheets
            for i in range(num_items):
                final_produced[i] += plate_ups_list[last_plate][i] * add_sheets
    
    return plate_ups_list, sheets_list, final_produced


# --- Main Button ---
if st.button("🚀 ক্যালকুলেট করুন"):
    
    targets = [l["Target QTY"] for l in labels_input]
    original_qty = [l["Original QTY"] for l in labels_input]
    
    with st.spinner("Calculating optimal plate ratio..."):
        plate_ups_list, sheets_list, final_produced = optimize_plates_manual(targets, grid_size, int(num_plates))
    
    # Calculate excess
    excess_list = [final_produced[i] - targets[i] for i in range(len(targets))]
    total_excess = sum(max(0, e) for e in excess_list)
    total_original = sum(original_qty)
    total_target = sum(targets)
    total_produced = sum(final_produced)
    
    over_print_pct = round((total_excess / total_original * 100), 2) if total_original > 0 else 0
    
    # --- Report ---
    st.divider()
    st.subheader("📊 Production Report")
    
    # Data Table
    final_data = []
    for i in range(len(labels_input)):
        row = {
            "Name": labels_input[i]["Name"],
            "Original QTY": labels_input[i]["Original QTY"],
            "Target QTY": targets[i]
        }
        for p in range(int(num_plates)):
            row[f"Plate {p+1} (Ups)"] = plate_ups_list[p][i]
        row["Total Produced"] = final_produced[i]
        row["Excess"] = excess_list[i]
        row["Over Print (%)"] = round((excess_list[i] / original_qty[i] * 100), 2) if original_qty[i] > 0 else 0
        final_data.append(row)
    
    df = pd.DataFrame(final_data)
    
    # Total row
    total_row = {"Name": "📊 TOTAL"}
    for col in df.columns:
        if col != "Name":
            if "Ups" in col:
                total_row[col] = df[col].sum()
            elif df[col].dtype in ['int64', 'float64']:
                total_row[col] = df[col].sum()
    
    df_with_total = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    st.dataframe(df_with_total, use_container_width=True)
    
    # Print Instructions
    st.write("### 📝 Printing Instructions")
    cols_info = st.columns(int(num_plates))
    for p in range(int(num_plates)):
        with cols_info[p]:
            st.info(f"""
            **Plate {p+1}:** 
            - {sheets_list[p]} sheets
            - Total UPS: {sum(plate_ups_list[p])}
            """)
    
    # Final Verdict
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Target", f"{total_target:,} pcs")
    with col2:
        st.metric("🏭 Total Produced", f"{total_produced:,} pcs", 
                  delta=f"+{total_excess}" if total_excess > 0 else None)
    with col3:
        st.metric("📊 Over Print", f"{over_print_pct}%")
    
    if over_print_pct <= 3:
        st.success(f"✅ Excellent! Only {over_print_pct}% Over Print!")
        st.balloons()
    elif over_print_pct <= 7:
        st.info(f"📈 {over_print_pct}% Over Print - Acceptable range.")
    else:
        st.warning(f"⚠️ {over_print_pct}% Over Print - Try increasing number of plates.")
