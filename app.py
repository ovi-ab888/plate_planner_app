import streamlit as st
import math
import pandas as pd

st.set_page_config(page_title="Pre-Press Optimizer Pro", layout="wide")

st.title("🎨 Pre-Press Grid & Plate Optimizer")
st.write("---")

# --- Sidebar ---
st.sidebar.header("কনফিগারেশন")
grid_size = st.sidebar.number_input("একটি প্লেটে মোট কয়টি লেবেল (Grid Size)?", min_value=1, value=30)
extra_percent = st.sidebar.number_input("অ্যাড-অন (Extra %) কত হবে?", min_value=0.0, value=0.0)
num_labels = st.sidebar.number_input("মোট কত পদের লেবেল?", min_value=1, value=1, step=1)

st.sidebar.subheader("প্লেট সেটিংস")
num_plates = st.sidebar.number_input("কয়টি প্লেট করতে চান?", min_value=1, value=1, step=1)

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
        labels_input.append({"Name": name, "Original QTY": qty, "Target QTY": target, "Remaining": target})

if st.button("ক্যালকুলেট করুন"):
    temp_labels = [l.copy() for l in labels_input]
    plate_ups_data = {f"Plate {p+1}": [] for p in range(int(num_plates))}
    plate_sheets_info = []

    # --- Multi-Plate Fill-The-Gap Logic ---
    for p in range(int(num_plates)):
        total_rem = sum(l["Remaining"] for l in temp_labels)
        if total_rem <= 0:
            for i in range(len(temp_labels)): plate_ups_data[f"Plate {p+1}"].append(0)
            plate_sheets_info.append(0)
            continue

        # Proportional Ups (Floor use korchi jate extra na hoy)
        ups_list = [math.floor((l["Remaining"] / total_rem) * grid_size) for l in temp_labels]
        
        # 'Fill the Gap' Logic: Jora kore grid_size milabo na.
        # Jodi 0 hoye jay kono label, kintu shetar demand thake, tobe at least 1 up dibo.
        for i, u in enumerate(ups_list):
            if u == 0 and temp_labels[i]["Remaining"] > 0 and sum(ups_list) < grid_size:
                ups_list[i] = 1

        # Calculate Sheets based on current Ups
        needed_sheets = []
        for i, u in enumerate(ups_list):
            if u > 0:
                needed_sheets.append(math.ceil(temp_labels[i]["Remaining"] / u))
        
        current_plate_sheets = max(needed_sheets) if needed_sheets else 0
        plate_sheets_info.append(current_plate_sheets)

        # Record and Update
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

    # Total Row
    total_row = {"Name": "TOTAL", "Original QTY": df_final["Original QTY"].sum(), "Target QTY": df_final["Target QTY"].sum()}
    for p in range(int(num_plates)):
        total_row[f"Plate {p+1} (Ups)"] = df_final[f"Plate {p+1} (Ups)"].sum()
    total_row["Total Produced"] = df_final["Total Produced"].sum()
    total_row["Excess"] = df_final["Excess"].sum()
    total_row["Over Print (%)"] = round(df_final["Over Print (%)"].mean(), 2)
    
    df_with_total = pd.concat([df_final, pd.DataFrame([total_row])], ignore_index=True)
    st.table(df_with_total)

    # Blank Slots Info
    for p in range(int(num_plates)):
        used_ups = df_final[f"Plate {p+1} (Ups)"].sum()
        blank = grid_size - used_ups
        if blank > 0:
            st.warning(f"⚠️ Plate {p+1}-এ {blank}টি ঘর খালি (Blank) থাকবে অপচয় কমানোর জন্য।")

    st.write("### 📝 প্রিন্টিং ইনস্ট্রাকশন:")
    cols_info = st.columns(int(num_plates))
    for p in range(int(num_plates)):
        with cols_info[p]:
            st.info(f"**Plate {p+1}:** {plate_sheets_info[p]} শিট প্রিন্ট করতে হবে।")

    if any(df_final["Excess"] < 0):
        st.error("⚠️ সতর্কবার্তা: কিছু লেবেল টার্গেটের চেয়ে কম উৎপাদন হচ্ছে! অনুগ্রহ করে প্লেট সংখ্যা বা শিট সংখ্যা পুনরায় চেক করুন।")
    else:
        st.success("✅ অভিনন্দন! 'Fill the Gap' লজিকে অপচয় কমিয়ে টার্গেট পূরণ করা হয়েছে।")
