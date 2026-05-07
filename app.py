import streamlit as st
import math
import pandas as pd

st.set_page_config(page_title="Pre-Press Optimizer Pro", layout="wide")

# CSS দিয়ে টেবিল এবং UI আরও সুন্দর করা
st.markdown("""
    <style>
    /* মেইন ব্যাকগ্রাউন্ড কালো করা */
    .stApp {
        background-color: #000000;
    }
    
    /* টেবিলের ব্যাকগ্রাউন্ড কালো এবং বর্ডার সুন্দর করা */
    .stTable {
        background-color: #1e1e1e !important;
        border-radius: 10px;
        color: white !important;
    }

    /* টেবিলের ভেতরকার টেক্সট সাদা করা */
    table {
        color: white !important;
    }
    
    /* ইনপুট বক্সের লেবেলগুলো সাদা করা */
    label, p {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎨 Pre-Press Grid & Plate Optimizer")
st.write("লেবেল কোয়ান্টিটি এবং মাল্টি-প্লেট অপ্টিমাইজেশন টুল।")

# --- Sidebar ---
st.sidebar.header("⚙️ কনফিগারেশন")
grid_size = st.sidebar.number_input("একটি প্লেটে মোট কয়টি লেবেল (Grid Size)?", min_value=1, value=30)
extra_percent = st.sidebar.number_input("অ্যাড-অন (Extra %) কত হবে?", min_value=0.0, value=0.0)
num_labels = st.sidebar.number_input("মোট কত পদের লেবেল?", min_value=1, value=8, step=1)

st.sidebar.subheader("🍽️ প্লেট সেটিংস")
num_plates = st.sidebar.number_input("কয়টি প্লেট করতে চান?", min_value=1, value=1, step=1)

# --- Data Input ---
st.subheader("📦 লেবেল কোয়ান্টিটি ইনপুট")
labels_input = []
cols = st.columns(2)
for i in range(int(num_labels)):
    col_idx = i % 2
    with cols[col_idx]:
        c1, c2 = st.columns([2, 1])
        name = c1.text_input(f"নাম {i+1}", value=f"Label {i+1}", key=f"n_{i}")
        qty = c2.number_input(f"QTY", min_value=1, value=1000, key=f"q_{i}")
        target = math.ceil(qty * (1 + extra_percent / 100))
        labels_input.append({"Name": name, "Original QTY": qty, "Target QTY": target, "Remaining": target})

if st.button("🚀 ক্যালকুলেট করুন"):
    temp_labels = [l.copy() for l in labels_input]
    plate_ups_data = {f"Plate {p+1}": [] for p in range(int(num_plates))}
    plate_sheets_info = []

    # --- Multi-Plate Smart Ratio Calculation ---
    for p in range(int(num_plates)):
        total_rem = sum(l["Remaining"] for l in temp_labels)
        if total_rem <= 0:
            for i in range(len(temp_labels)): plate_ups_data[f"Plate {p+1}"].append(0)
            plate_sheets_info.append(0)
            continue

        # Smart Ratio Logic: 
        # ছোট অর্ডারগুলোকে (Ratio < 0.5) প্রথম প্লেটে না দিয়ে পরের প্লেটের জন্য জমা রাখে
        ups_list = []
        for l in temp_labels:
            if l["Remaining"] <= 0:
                ups_list.append(0)
                continue
            
            actual_ratio = (l["Remaining"] / total_rem) * grid_size
            
            # যদি আরও প্লেট বাকি থাকে এবং অর্ডার খুব ছোট হয়, তবে এই প্লেটে ০ দাও
            if actual_ratio < 0.5 and (p + 1) < int(num_plates):
                ups_list.append(0)
            else:
                ups_list.append(max(1, round(actual_ratio)))

        # Adjusting to grid size
        diff = grid_size - sum(ups_list)
        if diff != 0:
            max_idx = 0
            max_val = -1
            for i, l in enumerate(temp_labels):
                if l["Remaining"] > max_val:
                    max_val = l["Remaining"]; max_idx = i
            
            if ups_list[max_idx] + diff >= 0:
                ups_list[max_idx] += diff

        # Calculate Sheets (Must cover all targets in this plate)
        needed_sheets_list = []
        for i, u in enumerate(ups_list):
            if u > 0:
                needed_sheets_list.append(math.ceil(temp_labels[i]["Remaining"] / u))
        
        current_plate_sheets = max(needed_sheets_list) if needed_sheets_list else 0
        plate_sheets_info.append(current_plate_sheets)

        # Record Ups and Update Remaining
        for i, u in enumerate(ups_list):
            plate_ups_data[f"Plate {p+1}"].append(u)
            produced = u * current_plate_sheets
            temp_labels[i]["Remaining"] = max(0, temp_labels[i]["Remaining"] - produced)

    # --- Build Master Table ---
    st.divider()
    st.subheader("📊 ফাইনাল রিপোর্ট (Grand Total)")
    
    final_data = []
    for i in range(len(labels_input)):
        row = {
            "Name": labels_input[i]["Name"],
            "Original QTY": labels_input[i]["Original QTY"],
            "Target QTY": labels_input[i]["Target QTY"]
        }
        
        total_produced = 0
        for p in range(int(num_plates)):
            ups = plate_ups_data[f"Plate {p+1}"][i]
            row[f"Plate {p+1} (Ups)"] = ups
            total_produced += (ups * plate_sheets_info[p])
            
        row["Total Produced"] = total_produced
        row["Excess"] = total_produced - labels_input[i]["Target QTY"]
        row["Over Print (%)"] = round((row["Excess"] / labels_input[i]["Original QTY"] * 100), 2) if labels_input[i]["Original QTY"] > 0 else 0
        
        final_data.append(row)

    df_final = pd.DataFrame(final_data)

    # Grand Total Row
    total_row = {"Name": "TOTAL", "Original QTY": df_final["Original QTY"].sum(), "Target QTY": df_final["Target QTY"].sum()}
    for p in range(int(num_plates)):
        total_row[f"Plate {p+1} (Ups)"] = df_final[f"Plate {p+1} (Ups)"].sum()
    total_row["Total Produced"] = df_final["Total Produced"].sum()
    total_row["Excess"] = df_final["Excess"].sum()
    total_row["Over Print (%)"] = round(df_final["Over Print (%)"].mean(), 2)
    
    df_with_total = pd.concat([df_final, pd.DataFrame([total_row])], ignore_index=True)
    
    # Display Table
    st.table(df_with_total)

    # Printing Instructions
    st.write("### 📝 প্রিন্টিং ইনস্ট্রাকশন:")
    cols_info = st.columns(int(num_plates))
    for p in range(int(num_plates)):
        with cols_info[p]:
            st.info(f"**Plate {p+1}:** {plate_sheets_info[p]} শিট প্রিন্ট করতে হবে।")

    # Final Summary Status
    if any(df_final["Excess"] < 0):
        st.error("⚠️ সতর্কবার্তা: কিছু লেবেল টার্গেটের চেয়ে কম উৎপাদন হচ্ছে! প্লেট সংখ্যা বাড়িয়ে দেখুন।")
    else:
        st.success("✅ অভিনন্দন! স্মার্ট রেশিও লজিকে টার্গেট কোয়ান্টিটি সম্পূর্ণ পূরণ হয়েছে।")

# Footer
st.markdown("---")
st.caption("Developed for Towhidul Islam Tushar | Pre-Press Smart Optimizer 2026")
