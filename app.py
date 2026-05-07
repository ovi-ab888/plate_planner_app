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
else:
    num_plates_manual = st.sidebar.number_input("কয়টি প্লেট করতে চান?", min_value=1, value=2, step=1)

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
# NEW & IMPROVED CALCULATION
# ============================================
def calculate_balanced_plates(num_plates, labels_input, grid_size):
    """
    Balanced algorithm - সব প্লেটে সমান UPS বিতরণ করে
    """
    num_items = len(labels_input)
    targets = [l["Target QTY"] for l in labels_input]
    
    # Step 1: প্রতি প্লেটের জন্য UPS বের করা (সব প্লেটে সমান pattern)
    # UPS হবে proportional to target
    total_target = sum(targets)
    
    if total_target == 0:
        return None, None, None
    
    # বেস UPS হিসাব (একটি প্লেটের জন্য total UPS = grid_size)
    base_ups_per_plate = []
    for i in range(num_items):
        # proportional UPS
        ups = round((targets[i] / total_target) * grid_size)
        base_ups_per_plate.append(max(1, ups))  # কমপক্ষে 1
    
    # Grid size adjust
    diff = grid_size - sum(base_ups_per_plate)
    if diff != 0:
        # যাদের target বেশি তাদের বাড়াই
        indices = list(range(num_items))
        indices.sort(key=lambda i: targets[i], reverse=True)
        for j in range(abs(diff)):
            idx = indices[j % num_items]
            if diff > 0:
                base_ups_per_plate[idx] += 1
            else:
                if base_ups_per_plate[idx] > 1:
                    base_ups_per_plate[idx] -= 1
    
    # Step 2: সব প্লেটের জন্য একই UPS pattern থাকবে
    # কিন্তু শিট সংখ্যা ভিন্ন হতে পারে (প্রতি প্লেটে আলাদা শিট)
    
    # Step 3: প্রতি প্লেটের জন্য কত শিট লাগবে?
    # আমরা চাই সব প্লেট মিলিয়ে টার্গেট পূরণ হোক
    # এবং সব প্লেটের শিট সংখ্যা যতটা সম্ভব সমান থাকুক
    
    # মোট প্রয়োজনীয় শিট (প্রতি লেবেলের জন্য)
    sheets_per_label = []
    for i in range(num_items):
        if base_ups_per_plate[i] > 0:
            sheets_needed = math.ceil(targets[i] / (base_ups_per_plate[i] * num_plates))
            sheets_per_label.append(sheets_needed)
        else:
            sheets_per_label.append(0)
    
    # সব লেবেলের জন্য সমান শিট সংখ্যা নিতে হবে (এক প্লেটে একবার রান)
    # তাই সবচেয়ে বেশি যেটা লাগে সেটাই হবে শিট সংখ্যা
    sheets_per_plate = max(sheets_per_label) if sheets_per_label else 0
    
    # Step 4: হিসাব করে দেখা প্রতিটি লেবেল কত উৎপাদন হবে
    plate_ups_data = {}
    for p in range(num_plates):
        plate_ups_data[f"Plate {p+1}"] = base_ups_per_plate.copy()
    
    plate_sheets_info = [sheets_per_plate] * num_plates
    
    # উৎপাদিত পরিমাণ ও excess বের করা
    total_produced = [0] * num_items
    for i in range(num_items):
        total_produced[i] = base_ups_per_plate[i] * sheets_per_plate * num_plates
    
    excess_list = [total_produced[i] - targets[i] for i in range(num_items)]
    
    return plate_ups_data, plate_sheets_info, {
        "total_produced": total_produced,
        "excess": excess_list,
        "sheets_per_plate": sheets_per_plate
    }


def find_best_plate_count_balanced(labels_input, grid_size, max_plates):
    """Balanced algorithm দিয়ে সেরা প্লেট সংখ্যা বের করা"""
    targets = [l["Target QTY"] for l in labels_input]
    total_target = sum(targets)
    best_result = None
    best_score = float('inf')
    all_results = []
    
    for p in range(1, max_plates + 1):
        plate_ups_data, plate_sheets_info, calc_data = calculate_balanced_plates(p, labels_input, grid_size)
        
        if calc_data is None:
            continue
        
        total_produced_sum = sum(calc_data["total_produced"])
        total_excess = sum(max(0, e) for e in calc_data["excess"])
        
        # Excess % বের করা
        excess_percent = (total_excess / total_target * 100) if total_target > 0 else 0
        
        # Score: কম excess ভালো, কম প্লেট ভালো
        score = total_excess + (p * 10)  # প্লেট প্রতি 10 পয়েন্ট যোগ
        
        all_results.append({
            "num_plates": p,
            "total_excess": total_excess,
            "total_produced": total_produced_sum,
            "total_target": total_target,
            "excess_percent": round(excess_percent, 2),
            "plate_ups_data": plate_ups_data,
            "plate_sheets_info": plate_sheets_info,
            "calc_data": calc_data
        })
        
        if score < best_score:
            best_score = score
            best_result = all_results[-1]
    
    return best_result, all_results


# --- Main Button ---
if st.button("ক্যালকুলেট করুন"):
    
    if auto_plate_mode:
        with st.spinner("🤖 সেরা প্লেট সংখ্যা বের করা হচ্ছে..."):
            best_result, all_results = find_best_plate_count_balanced(labels_input, grid_size, max_plates_to_try)
        
        if best_result is None:
            st.error("ক্যালকুলেশন করতে সমস্যা হয়েছে!")
            st.stop()
        
        num_plates_used = best_result["num_plates"]
        plate_sheets_info = best_result["plate_sheets_info"]
        plate_ups_data = best_result["plate_ups_data"]
        calc_data = best_result["calc_data"]
        
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
                "এক্সেস (%)": res["excess_percent"]
            })
        
        df_compare = pd.DataFrame(comparison_data)
        st.dataframe(df_compare, use_container_width=True)
        
        if best_result["total_excess"] == 0:
            st.success(f"✅ **সেরা প্লেট সংখ্যা: {num_plates_used} টি** (Perfect — কোনো Excess নেই!)")
            st.balloons()
        else:
            st.success(f"✅ **সেরা প্লেট সংখ্যা: {num_plates_used} টি** (Excess: {best_result['total_excess']} পিস, {best_result['excess_percent']}%)")
    
    else:
        # Manual mode
        num_plates_used = num_plates_manual
        plate_ups_data, plate_sheets_info, calc_data = calculate_balanced_plates(num_plates_used, labels_input, grid_size)
        
        if plate_ups_data is None:
            st.error("ক্যালকুলেশন করতে সমস্যা হয়েছে!")
            st.stop()
    
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
        
        for p in range(num_plates_used):
            ups = plate_ups_data[f"Plate {p+1}"][i]
            row[f"Plate {p+1} (Ups)"] = ups
        
        row["Total Produced"] = calc_data["total_produced"][i]
        row["Excess"] = calc_data["excess"][i]
        row["Over Print (%)"] = round((calc_data["excess"][i] / labels_input[i]["Original QTY"] * 100), 2) if labels_input[i]["Original QTY"] > 0 else 0
        
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
    st.dataframe(df_with_total, use_container_width=True)  # table থেকে dataframe ভালো দেখায়
    
    # শিট ইনস্ট্রাকশন
    st.write("### 📝 প্রিন্টিং ইনস্ট্রাকশন:")
    cols_info = st.columns(num_plates_used)
    for p in range(num_plates_used):
        with cols_info[p]:
            st.info(f"**Plate {p+1}:** {plate_sheets_info[p]} শিট প্রিন্ট করতে হবে। (প্রতি শিটে {grid_size}টি লেবেল)")
    
    # চেক
    if any(calc_data["excess"][i] < 0 for i in range(len(labels_input))):
        st.error("⚠️ কিছু লেবেল টার্গেটের চেয়ে কম উৎপাদন হচ্ছে! প্লেট সংখ্যা বাড়ানোর চেষ্টা করুন।")
    else:
        st.success("✅ সব লেবেল টার্গেট পূরণ হয়েছে!")
