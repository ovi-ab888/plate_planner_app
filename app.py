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

# NEW: Auto Plate Ratio Finder অপশন
auto_plate_mode = st.sidebar.checkbox("🤖 Auto Plate Ratio Finder (অটোমেটিক প্লেট সংখ্যা নির্বাচন)", value=True)

if auto_plate_mode:
    max_plates_to_try = st.sidebar.slider("সর্বোচ্চ কতটি প্লেট ট্রাই করবেন?", min_value=1, max_value=20, value=5)
    st.sidebar.info("Auto মোড অন থাকায় সিস্টেম নিজেই সেরা প্লেট সংখ্যা বের করবে। নিচের প্লেট সংখ্যা ইনপুট ইগনোর হবে।")
    num_plates = 1  # ডামি ভ্যালু, পরে অটো ওভাররাইড করবে
else:
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

# NEW: ফাংশন যা একটি নির্দিষ্ট প্লেট সংখ্যার জন্য ক্যালকুলেশন করে
def calculate_for_plates(num_plates, labels_input, grid_size):
    temp_labels = [l.copy() for l in labels_input]
    plate_ups_data = {f"Plate {p+1}": [] for p in range(num_plates)}
    plate_sheets_info = []
    
    for p in range(num_plates):
        total_rem = sum(l["Remaining"] for l in temp_labels)
        if total_rem <= 0:
            for i in range(len(temp_labels)):
                plate_ups_data[f"Plate {p+1}"].append(0)
            plate_sheets_info.append(0)
            continue
        
        # Proportional Ups
        ups_list = [round((l["Remaining"] / total_rem) * grid_size) for l in temp_labels]
        
        # Adjusting to grid size
        diff = grid_size - sum(ups_list)
        if diff != 0:
            max_idx = 0
            max_val = -1
            for i, l in enumerate(temp_labels):
                if l["Remaining"] > max_val:
                    max_val = l["Remaining"]
                    max_idx = i
            ups_list[max_idx] += diff
        
        # Calculate Sheets
        needed_sheets = []
        for i, u in enumerate(ups_list):
            if u > 0:
                needed_sheets.append(math.ceil(temp_labels[i]["Remaining"] / u))
        
        current_plate_sheets = max(needed_sheets) if needed_sheets else 0
        plate_sheets_info.append(current_plate_sheets)
        
        # Update Remaining
        for i, u in enumerate(ups_list):
            plate_ups_data[f"Plate {p+1}"].append(u)
            produced = u * current_plate_sheets
            temp_labels[i]["Remaining"] = max(0, temp_labels[i]["Remaining"] - produced)
    
    # মোট Excess বের করা (Optimization এর জন্য)
    total_excess = 0
    total_target = sum(l["Target QTY"] for l in labels_input)
    total_produced_overall = 0
    for i in range(len(labels_input)):
        produced = 0
        for p in range(num_plates):
            produced += plate_ups_data[f"Plate {p+1}"][i] * plate_sheets_info[p]
        total_produced_overall += produced
        excess = produced - labels_input[i]["Target QTY"]
        if excess > 0:
            total_excess += excess
    
    return {
        "num_plates": num_plates,
        "total_excess": total_excess,
        "total_produced": total_produced_overall,
        "total_target": total_target,
        "plate_sheets": plate_sheets_info,
        "plate_ups_data": plate_ups_data,
        "temp_labels": temp_labels
    }

# NEW: Auto Plate নির্বাচন ফাংশন
def find_best_plate_count(labels_input, grid_size, max_plates):
    best_result = None
    best_score = float('inf')
    
    results = []
    for p in range(1, max_plates + 1):
        result = calculate_for_plates(p, labels_input, grid_size)
        # স্কোর: Excess কম ভালো। যদি Excess সমান হয়, তাহলে কম প্লেট ভালো।
        score = result["total_excess"] + (p * 0.01)  # .01 যোগ করলাম যাতে কম প্লেটে priority যায়
        results.append((score, result))
        
        if score < best_score:
            best_score = score
            best_result = result
    
    return best_result, results

# --- Main Calculation Button ---
if st.button("ক্যালকুলেট করুন"):
    
    if auto_plate_mode:
        # NEW: Auto Mode - সেরা প্লেট সংখ্যা বের করো
        with st.spinner("🤖 সেরা প্লেট সংখ্যা বের করা হচ্ছে..."):
            best_result, all_results = find_best_plate_count(labels_input, grid_size, max_plates_to_try)
        
        num_plates_used = best_result["num_plates"]
        plate_sheets_info = best_result["plate_sheets"]
        plate_ups_data = best_result["plate_ups_data"]
        
        # NEW: Auto রিপোর্ট দেখানো
        st.divider()
        st.subheader("🤖 অটো প্লেট রেশিও ফাইন্ডার রিপোর্ট")
        
        # সবার জন্য তুলনামূলক টেবিল
        comparison_data = []
        for score, res in all_results:
            comparison_data.append({
                "প্লেট সংখ্যা": res["num_plates"],
                "মোট প্রোডিউস": res["total_produced"],
                "মোট টার্গেট": res["total_target"],
                "মোট এক্সেস": res["total_excess"],
                "এক্সেস (%)": round((res["total_excess"] / res["total_target"]) * 100, 2) if res["total_target"] > 0 else 0
            })
        
        df_compare = pd.DataFrame(comparison_data)
        st.dataframe(df_compare, use_container_width=True)
        
        st.success(f"✅ **সেরা প্লেট সংখ্যা: {num_plates_used} টি** (সবচেয়ে কম এক্সেস ও অপটিমাইজড প্রোডাকশন)")
        
        # BEST প্লেটের জন্য ইনসাইট
        if best_result["total_excess"] == 0:
            st.balloons()
            st.info("🎯 পারফেক্ট! কোনো এক্সেস প্রিন্ট ছাড়াই টার্গেট পূরণ হয়েছে।")
        else:
            st.warning(f"⚠️ সর্বনিম্ন এক্সেস: {best_result['total_excess']} পিস (টার্গেটের {round((best_result['total_excess']/best_result['total_target'])*100,2)}%)")
    
    else:
        # Manual Mode (আগের মতো)
        num_plates_used = int(num_plates)
        result = calculate_for_plates(num_plates_used, labels_input, grid_size)
        plate_sheets_info = result["plate_sheets"]
        plate_ups_data = result["plate_ups_data"]
    
    # --- Build Master Table (ইউজারকে দেখানোর জন্য) ---
    st.divider()
    st.subheader("📊 ফাইনাল রিপোর্ট (Grand Total)")
    
    final_data = []
    for i in range(len(labels_input)):
        row = {
            "Name": labels_input[i]["Name"],
            "Original QTY": labels_input[i]["Original QTY"],
            "Target QTY": labels_input[i]["Target QTY"]
        }
        
        # Add Plate Ups Columns
        total_produced = 0
        for p in range(num_plates_used):
            ups = plate_ups_data[f"Plate {p+1}"][i]
            row[f"Plate {p+1} (Ups)"] = ups
            total_produced += (ups * plate_sheets_info[p])
            
        row["Total Produced"] = total_produced
        row["Excess"] = total_produced - labels_input[i]["Target QTY"]
        row["Over Print (%)"] = round((row["Excess"] / labels_input[i]["Original QTY"] * 100), 2) if labels_input[i]["Original QTY"] > 0 else 0
        
        final_data.append(row)

    df_final = pd.DataFrame(final_data)

    # Calculate Total Row
    total_row = {"Name": "TOTAL", "Original QTY": df_final["Original QTY"].sum(), "Target QTY": df_final["Target QTY"].sum()}
    for p in range(num_plates_used):
        total_row[f"Plate {p+1} (Ups)"] = df_final[f"Plate {p+1} (Ups)"].sum()
    total_row["Total Produced"] = df_final["Total Produced"].sum()
    total_row["Excess"] = df_final["Excess"].sum()
    total_row["Over Print (%)"] = round(df_final["Over Print (%)"].mean(), 2)
    
    df_with_total = pd.concat([df_final, pd.DataFrame([total_row])], ignore_index=True)
    
    # Display Table
    st.table(df_with_total)

    # Display Sheet Requirements under the table
    st.write("### 📝 প্রিন্টিং ইনস্ট্রাকশন:")
    cols_info = st.columns(num_plates_used)
    for p in range(num_plates_used):
        with cols_info[p]:
            st.info(f"**Plate {p+1}:** {plate_sheets_info[p]} শিট প্রিন্ট করতে হবে।")

    # Final Check
    if any(df_final["Excess"] < 0):
        st.error("⚠️ সতর্কবার্তা: কিছু লেবেল টার্গেটের চেয়ে কম উৎপাদন হচ্ছে!")
    else:
        st.success("✅ অভিনন্দন! টার্গেট কোয়ান্টিটি সম্পূর্ণ পূরণ হয়েছে।")
