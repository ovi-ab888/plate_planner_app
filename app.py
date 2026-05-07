import streamlit as st
import math
import pandas as pd

st.set_page_config(page_title="Pre-Press Optimizer Pro", layout="wide")

st.title("🎨 Pre-Press Grid & Plate Optimizer")

# --- Sidebar ---
st.sidebar.header("কনফিগারেশন")
grid_size = st.sidebar.number_input("একটি প্লেটে মোট কয়টি লেবেল (Grid Size)?", min_value=1, value=30)
extra_percent = st.sidebar.number_input("অ্যাড-অন (Extra %) কত হবে?", min_value=0.0, value=0.0)
num_labels = st.sidebar.number_input("মোট কত পদের লেবেল?", min_value=1, value=1, step=1)

st.sidebar.subheader("প্লেট সেটিংস")
num_plates = st.sidebar.number_input("কয়টি প্লেট করতে চান?", min_value=1, value=1, step=1)

# --- Data Input ---
st.subheader("📦 লেবেল কোয়ান্টিটি ইনপুট")
labels = []
cols = st.columns(2)
for i in range(int(num_labels)):
    col_idx = i % 2
    with cols[col_idx]:
        c1, c2 = st.columns([2, 1])
        name = c1.text_input(f"নাম {i+1}", value=f"Label {i+1}", key=f"n_{i}")
        qty = c2.number_input(f"QTY", min_value=1, value=100, key=f"q_{i}")
        target = math.ceil(qty * (1 + extra_percent / 100))
        labels.append({"Name": name, "Original QTY": qty, "Target QTY": target, "Remaining": target})

if st.button("ক্যালকুলেট করুন"):
    all_plates_data = []
    
    # --- Multi-Plate Logic ---
    for p in range(int(num_plates)):
        st.write(f"### 📑 প্লেট নম্বর: {p+1}")
        
        current_labels = pd.DataFrame(labels)
        total_rem = current_labels["Remaining"].sum()
        
        if total_rem <= 0:
            st.warning(f"প্লেট {p+1} এর জন্য আর কোনো কোয়ান্টিটি অবশিষ্ট নেই।")
            continue

        # Proportional Ups calculation for this plate
        current_labels["Ups"] = (current_labels["Remaining"] / total_rem * grid_size).round().astype(int)
        
        # Adjust Ups to match grid_size
        diff = grid_size - current_labels["Ups"].sum()
        if diff != 0:
            idx = current_labels["Remaining"].idxmax()
            current_labels.at[idx, "Ups"] += diff
        
        # Calculate Sheets for THIS plate (based on the average or max needed)
        # প্রি-প্রেস লজিক: আমরা একটি নির্দিষ্ট শিট সংখ্যা ধরি (যেমন ৫০০ বা ১০০০)
        # এখানে আমরা অবশিষ্ট কাজের ওপর ভিত্তি করে শিট সংখ্যা নির্ধারণ করছি
        active_ups = current_labels[current_labels["Ups"] > 0]
        if not active_ups.empty:
            plate_sheets = math.ceil(active_ups["Remaining"].sum() / grid_size)
        else:
            plate_sheets = 0

        current_labels["Produced"] = current_labels["Ups"] * plate_sheets
        
        # Update remaining QTY for next plate
        for i in range(len(labels)):
            labels[i]["Remaining"] -= current_labels.at[i, "Produced"]
            if labels[i]["Remaining"] < 0: labels[i]["Remaining"] = 0

        # Display Plate Table
        plate_df = current_labels[current_labels["Ups"] > 0][["Name", "Ups", "Produced"]]
        plate_df.rename(columns={"Produced": f"Produced (Plate {p+1})"}, inplace=True)
        st.table(plate_df)
        st.info(f"এই প্লেটটি **{plate_sheets}** শিট প্রিন্ট করতে হবে।")
        
        all_plates_data.append({"df": current_labels, "sheets": plate_sheets})

    # --- GRAND TOTAL SECTION ---
    st.divider()
    st.subheader("📊 ফাইনাল রিপোর্ট (Grand Total)")
    
    final_rows = []
    for i in range(len(labels)):
        total_produced = sum([p["df"].at[i, "Produced"] for p in all_plates_data])
        orig_qty = labels[i]["Original QTY"]
        over_print = ((total_produced - orig_qty) / orig_qty * 100) if orig_qty > 0 else 0
        
        final_rows.append({
            "Name": labels[i]["Name"],
            "Original QTY": orig_qty,
            "Target QTY": labels[i]["Target QTY"],
            "Total Produced": total_produced,
            "Over Print (%)": round(over_print, 2)
        })
    
    final_df = pd.DataFrame(final_rows)
    
    # Adding Total Row
    total_row = pd.DataFrame([{
        "Name": "TOTAL",
        "Original QTY": final_df["Original QTY"].sum(),
        "Target QTY": final_df["Target QTY"].sum(),
        "Total Produced": final_df["Total Produced"].sum(),
        "Over Print (%)": round(final_df["Over Print (%)"].mean(), 2)
    }])
    
    final_df_with_total = pd.concat([final_df, total_row], ignore_index=True)
    
    st.table(final_df_with_total)

    st.success("✅ ক্যালকুলেশন সম্পন্ন হয়েছে। প্রতিটি প্লেটের জন্য আলাদা শিট ও আপস চেক করে নিন।")
