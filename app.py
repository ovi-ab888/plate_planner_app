import streamlit as st
import math
import pandas as pd

st.set_page_config(page_title="Pre-Press Master Optimizer", layout="wide")

st.title("🎨 Pre-Press Grid & Plate Optimizer (Manual Control)")
st.write("---")

# --- Sidebar Configuration ---
st.sidebar.header("১. কনফিগারেশন")
grid_size = st.sidebar.number_input("গ্রিড সাইজ (যেমন: ৩০)", min_value=1, value=30)
extra_percent = st.sidebar.number_input("অ্যাড-অন (Extra %)", min_value=0.0, value=0.0)
num_labels = st.sidebar.number_input("লেবেল পদ সংখ্যা", min_value=1, value=8)
num_plates = st.sidebar.number_input("প্লেট সংখ্যা", min_value=1, value=1)

# --- Data Input Section ---
st.subheader("📦 ২. লেবেল ডেটা ও ম্যানুয়াল Ups সেটিংস")
labels_data = []
cols = st.columns(2)

for i in range(int(num_labels)):
    col_idx = i % 2
    with cols[col_idx]:
        st.markdown(f"**লেবেল {i+1}**")
        c1, c2, c3 = st.columns([2, 1, 1])
        name = c1.text_input("নাম", value=f"Label {i+1}", key=f"n_{i}")
        qty = c2.number_input("QTY", min_value=1, value=1000, key=f"q_{i}")
        
        # Target with Extra
        target = math.ceil(qty * (1 + extra_percent / 100))
        
        # Manual Ups inputs for each plate
        ups_inputs = []
        for p in range(int(num_plates)):
            u = c3.number_input(f"Plt {p+1} Ups", min_value=0, max_value=grid_size, value=0, key=f"u_{i}_{p}")
            ups_inputs.append(u)
            
        labels_data.append({
            "Name": name, 
            "Original QTY": qty, 
            "Target QTY": target, 
            "Ups": ups_inputs
        })

# --- Sheets Input per Plate ---
st.subheader("📄 ৩. প্রিন্টিং শিট সংখ্যা নির্ধারণ")
plate_sheets = []
ps_cols = st.columns(int(num_plates))
for p in range(int(num_plates)):
    with ps_cols[p]:
        s = st.number_input(f"Plate {p+1} মোট শিট", min_value=0, value=100, key=f"ps_{p}")
        plate_sheets.append(s)

# --- Calculation Engine ---
if st.button("ফাইনাল রিপোর্ট জেনারেট করুন"):
    final_rows = []
    for i, label in enumerate(labels_data):
        row = {
            "Name": label["Name"],
            "Original QTY": label["Original QTY"],
            "Target QTY": label["Target QTY"]
        }
        
        total_produced = 0
        for p in range(int(num_plates)):
            u = label["Ups"][p]
            row[f"Plate {p+1} (Ups)"] = u
            total_produced += (u * plate_sheets[p])
            
        row["Total Produced"] = total_produced
        row["Excess"] = total_produced - label["Target QTY"]
        row["Over Print (%)"] = round((row["Excess"] / label["Original QTY"] * 100), 2) if label["Original QTY"] > 0 else 0
        final_rows.append(row)

    df_final = pd.DataFrame(final_rows)

    # Grand Total Row
    total_row = {"Name": "TOTAL", "Original QTY": df_final["Original QTY"].sum(), "Target QTY": df_final["Target QTY"].sum()}
    for p in range(int(num_plates)):
        total_row[f"Plate {p+1} (Ups)"] = df_final[f"Plate {p+1} (Ups)"].sum()
        # Grid Status Check
        used_ups = df_final[f"Plate {p+1} (Ups)"].sum()
        if used_ups > grid_size:
            st.error(f"❌ Plate {p+1}-এ আপনি {used_ups}টি ঘর ব্যবহার করেছেন, কিন্তু গ্রিড সাইজ মাত্র {grid_size}!")
        elif used_ups < grid_size:
            st.warning(f"⚠️ Plate {p+1}-এ {grid_size - used_ups}টি ঘর খালি আছে।")

    total_row["Total Produced"] = df_final["Total Produced"].sum()
    total_row["Excess"] = df_final["Excess"].sum()
    total_row["Over Print (%)"] = round(df_final["Over Print (%)"].mean(), 2)

    df_with_total = pd.concat([df_final, pd.DataFrame([total_row])], ignore_index=True)
    
    st.divider()
    st.subheader("📊 ফাইনাল রিপোর্ট (Grand Total)")
    st.table(df_with_total)

    # Efficiency Indicators
    if any(df_final["Excess"] < 0):
        st.error("🛑 টার্গেট পূরণ হয়নি! কিছু লেবেল কম উৎপাদন হচ্ছে। শিট বা Ups বাড়ান।")
    else:
        st.success("✅ পারফেক্ট! সব লেবেল টার্গেট পূরণ করেছে।")
