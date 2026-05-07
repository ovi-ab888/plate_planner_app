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
    max_plates_to_try = st.sidebar.slider("সর্বোচ্চ কতটি প্লেট ট্রাই করবেন?", min_value=1, max_value=10, value=4)
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
# AS YOUR MANUAL STRATEGY - Proper Optimization
# ============================================

def optimize_plates(labels_input, grid_size, num_plates):
    """
    আপনার ম্যানুয়াল কৌশল অনুসারে অপটিমাইজেশন
    - প্রতিটি প্লেটের UPS ভিন্ন হতে পারে
    - প্রতিটি প্লেটের শিট সংখ্যা ভিন্ন হতে পারে
    """
    num_items = len(labels_input)
    targets = [l["Target QTY"] for l in labels_input]
    
    # Best সমাধান খোঁজার জন্য ট্রায়াল
    best_solution = None
    best_excess = float('inf')
    
    # বিভিন্ন UPS কম্বিনেশন ট্রাই করা (অপটিমাইজড)
    # আমরা প্রতি প্লেটের জন্য UPS ভাগ করে দেব
    
    # Step 1: মোট টার্গেটের অনুপাতে বেস UPS
    total_target = sum(targets)
    base_ups = []
    for i in range(num_items):
        ups = max(1, round((targets[i] / total_target) * grid_size))
        base_ups.append(ups)
    
    # Adjust to grid size
    diff = grid_size - sum(base_ups)
    if diff > 0:
        for i in range(diff):
            max_idx = max(range(num_items), key=lambda i: targets[i])
            base_ups[max_idx] += 1
    elif diff < 0:
        for i in range(-diff):
            min_idx = min(range(num_items), key=lambda i: base_ups[i] if base_ups[i] > 1 else 999999)
            if base_ups[min_idx] > 1:
                base_ups[min_idx] -= 1
    
    # Step 2: বিভিন্ন প্লেটের জন্য UPS ভ্যারিয়েশন ট্রাই
    # আপনার ম্যানুয়াল সলিউশনের মতো - কিছু প্লেটে বেশি, কিছুতে কম
    
    best_ups_per_plate = []
    best_sheets_per_plate = []
    
    # মোট UPS per plate হবে grid_size, কিন্তু বিতরণ ভিন্ন হতে পারে
    # আমরা try করব বিভিন্ন distribution
    
    for attempt in range(50):  # 50 বার ট্রাই
        ups_per_plate = []
        for p in range(num_plates):
            if p == 0:
                # প্রথম প্লেটে base_ups
                current_ups = base_ups.copy()
            else:
                # পরবর্তী প্লেটে সামান্য পরিবর্তন
                current_ups = base_ups.copy()
                # র‍্যান্ডম অ্যাডজাস্টমেন্ট (অপটিমাইজেশনের জন্য)
                import random
                for _ in range(random.randint(-3, 3)):
                    idx = random.randint(0, num_items-1)
                    if current_ups[idx] > 1:
                        current_ups[idx] -= 1
                        current_ups[random.randint(0, num_items-1)] += 1
            
            # Grid size adjust
            diff2 = grid_size - sum(current_ups)
            if diff2 > 0:
                for _ in range(diff2):
                    max_idx = max(range(num_items), key=lambda i: targets[i])
                    current_ups[max_idx] += 1
            ups_per_plate.append(current_ups)
        
        # এখন শিট সংখ্যা বের করি
        # আমরা চাই সব প্লেট মিলিয়ে টার্গেট পূরণ হোক
        # এবং শিট সংখ্যা যতটা সম্ভব কম হোক
        
        remaining = targets.copy()
        sheets_per_plate = []
        total_sheets = 0
        
        for p in range(num_plates):
            if sum(remaining) <= 0:
                sheets_per_plate.append(0)
                continue
            
            # এই প্লেটের জন্য কত শিট লাগবে?
            max_sheets_needed = 0
            for i in range(num_items):
                if ups_per_plate[p][i] > 0 and remaining[i] > 0:
                    sheets = math.ceil(remaining[i] / ups_per_plate[p][i])
                    max_sheets_needed = max(max_sheets_needed, sheets)
            
            sheets_per_plate.append(max_sheets_needed)
            total_sheets += max_sheets_needed
            
            # Remaining আপডেট
            for i in range(num_items):
                produced = ups_per_plate[p][i] * max_sheets_needed
                remaining[i] = max(0, remaining[i] - produced)
        
        # Excess বের করি
        remaining_final = remaining
        excess = sum(remaining_final)  # যতটুকু কম পড়েছে
        overproduced = 0
        for i in range(num_items):
            produced = 0
            for p in range(num_plates):
                produced += ups_per_plate[p][i] * sheets_per_plate[p]
            if produced > targets[i]:
                overproduced += (produced - targets[i])
        
        total_excess = excess + overproduced
        
        if total_excess < best_excess:
            best_excess = total_excess
            best_ups_per_plate = ups_per_plate
            best_sheets_per_plate = sheets_per_plate
    
    # Final calculation with best solution
    final_remaining = targets.copy()
    final_produced = [0] * num_items
    
    for p in range(num_plates):
        sheets = best_sheets_per_plate[p]
        for i in range(num_items):
            produced = best_ups_per_plate[p][i] * sheets
            final_produced[i] += produced
            final_remaining[i] = max(0, final_remaining[i] - produced)
    
    excess_list = [final_produced[i] - targets[i] for i in range(num_items)]
    
    # Prepare return data
    plate_ups_data = {}
    for p in range(num_plates):
        plate_ups_data[f"Plate {p+1}"] = best_ups_per_plate[p]
    
    return plate_ups_data, best_sheets_per_plate, {
        "total_produced": final_produced,
        "excess": excess_list,
        "total_excess_sum": sum(max(0, e) for e in excess_list)
    }


def find_best_plate_count(labels_input, grid_size, max_plates):
    """বিভিন্ন প্লেট সংখ্যা ট্রাই করে সেরাটা বের করা"""
    targets = [l["Target QTY"] for l in labels_input]
    total_target = sum(targets)
    best_result = None
    best_score = float('inf')
    all_results = []
    
    for p in range(1, max_plates + 1):
        plate_ups_data, sheets_info, calc_data = optimize_plates(labels_input, grid_size, p)
        
        total_produced_sum = sum(calc_data["total_produced"])
        total_excess = calc_data["total_excess_sum"]
        excess_percent = (total_excess / total_target * 100) if total_target > 0 else 0
        
        score = total_excess + (p * 10)
        
        all_results.append({
            "num_plates": p,
            "total_excess": total_excess,
            "total_produced": total_produced_sum,
            "total_target": total_target,
            "excess_percent": round(excess_percent, 2),
            "plate_ups_data": plate_ups_data,
            "plate_sheets_info": sheets_info,
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
            best_result, all_results = find_best_plate_count(labels_input, grid_size, max_plates_to_try)
        
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
        
        st.info(f"📊 **মোট উৎপাদন:** {best_result['total_produced']} পিস | **টার্গেট:** {best_result['total_target']} পিস")
    
    else:
        # Manual mode
        num_plates_used = num_plates_manual
        plate_ups_data, plate_sheets_info, calc_data = optimize_plates(labels_input, grid_size, num_plates_used)
    
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
    st.dataframe(df_with_total, use_container_width=True)
    
    # শিট ইনস্ট্রাকশন
    st.write("### 📝 প্রিন্টিং ইনস্ট্রাকশন:")
    cols_info = st.columns(num_plates_used)
    for p in range(num_plates_used):
        with cols_info[p]:
            if plate_sheets_info[p] > 0:
                st.info(f"**Plate {p+1}:** {plate_sheets_info[p]} শিট প্রিন্ট করতে হবে। (প্রতি শিটে {grid_size}টি লেবেল)")
            else:
                st.info(f"**Plate {p+1}:** ব্যবহারের প্রয়োজন নেই।")
    
    # Final verdict
    total_excess = sum(max(0, e) for e in calc_data["excess"])
    if total_excess == 0:
        st.success("✅ পারফেক্ট! সব লেবেল ঠিক টার্গেট অনুযায়ী উৎপাদন হয়েছে।")
    else:
        st.warning(f"📈 মোট {total_excess} পিস অতিরিক্ত প্রিন্ট হয়েছে (Over Print)। এটি ইন্ডাস্ট্রি স্ট্যান্ডার্ডের মধ্যে থাকলে গ্রহণযোগ্য।")
