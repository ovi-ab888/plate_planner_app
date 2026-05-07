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
        # এখানে 'Remaining' রাখবে টার্গেট অনুযায়ী
        labels.append({"Name": name, "Original QTY": qty, "Target QTY": target, "Remaining": target})

if st.button("ক্যালকুলেট করুন"):
    all_plates_results = []
    
    # কপি তৈরি করা ক্যালকুলেশনের জন্য
    temp_labels = [label.copy() for label in labels]
    
    for p in range(int(num_plates)):
        st.write(f"### 📑 প্লেট নম্বর: {p+1}")
        
        total_rem = sum(l["Remaining"] for l in temp_labels)
        if total_rem <= 0:
            st.warning(f"প্লেট {p+1} এর জন্য আর কোনো কোয়ান্টিটি অবশিষ্ট নেই।")
            continue

        # Ups calculation using math.ceil for better coverage
        ups_list = []
        for l in temp_labels:
            u = round((l["Remaining"] / total_rem) * grid_size)
            ups_list.append(u)
        
        # Adjusting Ups to match grid_size exactly
        diff = grid_size - sum(ups_list)
        if diff != 0:
            # বড় কোয়ান্টিটিতে গ্যাপ অ্যাড করা
            max_idx = 0
            max_val = -1
            for i, l in enumerate(temp_labels):
                if l["Remaining"] > max_val:
                    max_val = l["Remaining"]
                    max_idx = i
            ups_list[max_idx] += diff

        # প্লেটের জন্য শিট ক্যালকুলেশন - এটিই সবচেয়ে গুরুত্বপূর্ণ
        # আমরা নিশ্চিত করছি যেন অন্তত টার্গেট কাভার হয়
        needed_sheets_list = []
        for i, u in enumerate(ups_list):
            if u > 0:
                needed_sheets_list.append(math.ceil(temp_labels[i]["Remaining"] / u))
        
        # প্লেটের শিট হবে সেই সংখ্যা যা দিয়ে সর্বোচ্চ ডিমান্ড মেটানো যায়
        plate_sheets = max(needed_sheets_list) if needed_sheets_list else 0

        # রেজাল্ট আপডেট
        plate_data = []
        for i, u in enumerate(ups_list):
            produced = u * plate_sheets
            if u > 0:
                plate_data.append({"Name": temp_labels[i]["Name"], "Ups": u, "Produced": produced})
            temp_labels[i]["Remaining"] = max(0, temp_labels[i]["Remaining"] - produced)
        
        st.table(pd.DataFrame(plate_data))
        st.info(f"এই প্লেটটি **{plate_sheets}** শিট প্রিন্ট করতে হবে।")
        all_plates_results.append(plate_data)

    # --- GRAND TOTAL REPORT ---
    st.divider()
    st.subheader("📊 ফাইনাল রিপোর্ট (Grand Total)")
    
    final_report = []
    for i, original_label in enumerate(labels):
        total_produced = 0
        for plate in all_plates_results:
            for item in plate:
                if item["Name"] == original_label["Name"]:
                    total_produced += item["Produced"]
        
        over_print_qty = total_produced - original_label["Target QTY"]
        over_print_per = (over_print_qty / original_label["Original QTY"] * 100) if original_label["Original QTY"] > 0 else 0
        
        final_report.append({
            "Name": original_label["Name"],
            "Original QTY": original_label["Original QTY"],
            "Target QTY": original_label["Target QTY"],
            "Total Produced": total_produced,
            "Short/Excess": over_print_qty,
            "Over Print (%)": round(over_print_per, 2)
        })

    final_df = pd.DataFrame(final_report)
    
    # Total Row calculation
    total_row = pd.DataFrame([{
        "Name": "TOTAL",
        "Original QTY": final_df["Original QTY"].sum(),
        "Target QTY": final_df["Target QTY"].sum(),
        "Total Produced": final_df["Total Produced"].sum(),
        "Short/Excess": final_df["Short/Excess"].sum(),
        "Over Print (%)": round(final_df["Over Print (%)"].mean(), 2)
    }])
    
    final_df_with_total = pd.concat([final_df, total_row], ignore_index=True)
    st.table(final_df_with_total)

    # সতর্কতা চেক
    if any(final_df["Short/Excess"] < 0):
        st.error("⚠️ সতর্কবার্তা: কিছু লেবেল টার্গেটের চেয়ে কম উৎপাদন হচ্ছে! অনুগ্রহ করে প্লেট সংখ্যা বাড়ান।")
    else:
        st.success("✅ অভিনন্দন! সব লেবেল টার্গেট অনুযায়ী অথবা তার বেশি উৎপাদিত হচ্ছে।")
