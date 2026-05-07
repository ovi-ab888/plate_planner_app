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
auto_plate_mode = st.sidebar.checkbox("🤖 Auto Plate Ratio Finder", value=True)

if auto_plate_mode:
    max_plates_to_try = st.sidebar.slider("সর্বোচ্চ কতটি প্লেট ট্রাই করবেন?", min_value=1, max_value=20, value=5)
    st.sidebar.info("Auto মোড অন - সিস্টেম নিজেই সেরা প্লেট সংখ্যা বের করবে")
    num_plates = 1
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
        labels_input.append({"Name": name, "Original QTY": qty, "Target QTY": target})

# ============================================
# FIXED CALCULATION FUNCTION
# ============================================
def calculate_for_plates(num_plates, labels_input, grid_size):
    """
    প্রতিটি প্লেটে সব লেবেলের UPS বিতরণ করে,
    যাতে কোনো প্লেট খালি না থাকে
    """
    # কপি তৈরি
    remaining = [l["Target QTY"] for l in labels_input]
    num_items = len(labels_input)
    
    # ফলাফল সংরক্ষণের জন্য
    plate_ups = {f"Plate {p+1}": [0] * num_items for p in range(num_plates)}
    plate_sheets = [0] * num_plates
    
    for plate_idx in range(num_plates):
        total_remaining = sum(remaining)
        
        if total_remaining <= 0:
            break
        
        # UPS বরাদ্দ - প্রতিটি লেবেলের জন্য কমপক্ষে 1 থাকবে (যতক্ষণ বাকি থাকে)
        ups_list = []
        for i in range(num_items):
            if remaining[i] > 0:
                # Proportional UPS + minimum guarantee
                proportional = (remaining[i] / total_remaining) * grid_size
                # কমপক্ষে 1, বেশি হলে proportional
                ups = max(1, int(proportional))
                ups_list.append(ups)
            else:
                ups_list.append(0)
        
        # সমষ্টি grid_size এর সমান করতে adjust
        total_ups = sum(ups_list)
        diff = grid_size - total_ups
        
        if diff > 0:
            # যাদের remaining বেশি তাদের বাড়ানো
            indices = [i for i in range(num_items) if remaining[i] > 0]
            indices.sort(key=lambda i: remaining[i], reverse=True)
            for i in range(min(diff, len(indices))):
                ups_list[indices[i]] += 1
        elif diff < 0:
            # যাদের remaining কম তাদের কমানো, কিন্তু 1 এর নিচে নয়
            indices = [i for i in range(num_items) if ups_list[i] > 1]
            indices.sort(key=lambda i: remaining[i])
            for i in range(min(abs(diff), len(indices))):
                ups_list[indices[i]] -= 1
        
        # এই প্লেটের জন্য Sheets বের করা
        needed_sheets = []
        for i in range(num_items):
            if ups_list[i] > 0 and remaining[i] > 0:
                sheets = math.ceil(remaining[i] / ups_list[i])
                needed_sheets.append(sheets)
        
        current_sheets = max(needed_sheets) if needed_sheets else 0
        plate_sheets[plate_idx] = current_sheets
        
        # UPS সংরক্ষণ এবং remaining আপডেট
        for i in range(num_items):
            plate_ups[f"Plate {plate_idx+1}"][i] = ups_list[i]
            produced = ups_list[i] * current_sheets
            remaining[i] = max(0, remaining[i] - produced)
    
    return plate_ups, plate_sheets, remaining


def find_best_plate_count(labels_input, grid_size, max_plates):
    """সবচেয়ে কম Excess যেখানে সেটা বের করে"""
    target_sum = sum(l["Target QTY"] for l in labels_input)
    best_result = None
    best_score = float('inf')
    all_results = []
    
    for p in range(1, max_plates + 1):
        plate_ups, plate_sheets, remaining_after = calculate_for_plates(p, labels_input, grid_size)
        
        # Excess বের করা
        total_produced = 0
        total_excess = 0
        
        for i, label in enumerate(labels_input):
            produced = 0
            for plate_idx in range(p):
                produced += plate_ups[f"Plate {plate_idx+1}"][i] * plate_sheets[plate_idx]
            total_produced += produced
            excess = produced - label["Target QTY"]
            if excess > 0:
                total_excess += excess
        
        score = total_excess + (p * 0.1)  # কম প্লেটে priority
        all_results.append({
            "num_plates": p,
            "total_excess": total_excess,
            "total_produced": total_produced,
            "total_target": target_sum,
            "excess_percent": (total_excess / target_sum * 100) if target_sum > 0 else 0,
            "plate_ups": plate_ups,
            "plate_sheets": plate_sheets
        })
        
        if score < best_score:
            best_score = score
            best_result = all_results[-1]
    
    return best_result, all_results


# --- Main Button ---
if st.button("ক্যালকুলেট করুন"):
    
    if auto_plate_mode:
        with st.spinner("🤖 সেরা প্লেট সংখ্যা বের করা হচ্ছে..."):
            best_result, all_results = find_best_plate_count(labels_input, grid_size, max_plates_to_try)
        
        num_plates_used = best_result["num_plates"]
        plate_sheets_info = best_result["plate_sheets"]
        plate_ups_data = best_result["plate_ups"]
        
        # তুলনামূলক টেবিল
        st.divider()
        st.subheader("🤖 অটো প্লেট রেশিও ফাইন্ডার রিপোর্ট")
        
        comparison_data = []
        for res in all_results:
            comparison_data.append({
                "প্লেট সংখ্যা": res["num_plates"],
                "মোট প্রোডিউস": res["total_produced"],
                "মোট টার্গেট": res["total_target"],
                "মোট এক্সেস": res["total_excess"],
                "এক্সেস (%)": round(res["excess_percent"], 2)
            })
        
        df_compare = pd.DataFrame(comparison_data)
        st.dataframe(df_compare, use_container_width=True)
        
        if best_result["total_excess"] == 0:
            st.success(f"✅ **সেরা প্লেট সংখ্যা: {num_plates_used} টি** (Perfect — কোনো Excess নেই!)")
            st.balloons()
        else:
            st.success(f"✅ **সেরা প্লেট সংখ্যা: {num_plates_used} টি** (Excess: {best_result['total_excess']} পিস, {round(best_result['excess_percent'],2)}%)")
    
    else:
        num_plates_used = int(num_plates)
        plate_ups_data, plate_sheets_info, _ = calculate_for_plates(num_plates_used, labels_input, grid_size)
    
    # --- ফাইনাল রিপোর্ট টেবিল ---
    st.divider()
    st.subheader("📊 ফাইনাল রিপোর্ট")
    
    final_data = []
    for i in range(len(labels_input)):
        row = {
            "Name": labels_input[i]["Name"],
            "Original QTY": labels_input[i]["Original QTY"],
            "Target QTY": labels_input[i]["Target QTY"]
        }
        
        total_produced = 0
        for p in range(num_plates_used):
            ups = plate_ups_data[f"Plate {p+1}"][i]
            row[f"Plate {p+1} (Ups)"] = ups
            total_produced += ups * plate_sheets_info[p]
        
        row["Total Produced"] = total_produced
        row["Excess"] = total_produced - labels_input[i]["Target QTY"]
        row["Over Print (%)"] = round((row["Excess"] / labels_input[i]["Original QTY"] * 100), 2) if labels_input[i]["Original QTY"] > 0 else 0
        
        final_data.append(row)
    
    df_final = pd.DataFrame(final_data)
    
    # Total row
    total_row = {"Name": "TOTAL", "Original QTY": df_final["Original QTY"].sum(), "Target QTY": df_final["Target QTY"].sum()}
    for p in range(num_plates_used):
        total_row[f"Plate {p+1} (Ups)"] = df_final[f"Plate {p+1} (Ups)"].sum()
    total_row["Total Produced"] = df_final["Total Produced"].sum()
    total_row["Excess"] = df_final["Excess"].sum()
    total_row["Over Print (%)"] = round(df_final["Over Print (%)"].mean(), 2)
    
    df_with_total = pd.concat([df_final, pd.DataFrame([total_row])], ignore_index=True)
    st.table(df_with_total)
    
    # শিট ইনস্ট্রাকশন
    st.write("### 📝 প্রিন্টিং ইনস্ট্রাকশন:")
    cols_info = st.columns(num_plates_used)
    for p in range(num_plates_used):
        with cols_info[p]:
            if plate_sheets_info[p] > 0:
                st.info(f"**Plate {p+1}:** {plate_sheets_info[p]} শিট প্রিন্ট করতে হবে।")
            else:
                st.info(f"**Plate {p+1}:** ব্যবহারের প্রয়োজন নেই।")
    
    if any(df_final["Excess"] < 0):
        st.error("⚠️ কিছু লেবেল টার্গেটের চেয়ে কম উৎপাদন হচ্ছে! প্লেট সংখ্যা বাড়ানোর চেষ্টা করুন।")
    else:
        st.success("✅ সব লেবেল টার্গেট পূরণ হয়েছে!")
