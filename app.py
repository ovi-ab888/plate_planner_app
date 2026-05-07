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
    max_plates_to_try = st.sidebar.slider("সর্বোচ্চ কতটি প্লেট ট্রাই করবেন?", min_value=1, max_value=6, value=3)
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
# YOUR EXACT MANUAL STRATEGY
# ============================================

def calculate_exact_manual_strategy(labels_input, grid_size, num_plates):
    """
    আপনার ম্যানুয়াল রিপোর্টের কৌশলটি প্রোগ্রামেটিকভাবে করা
    """
    num_items = len(labels_input)
    targets = [l["Target QTY"] for l in labels_input]
    original_qty = [l["Original QTY"] for l in labels_input]
    
    # Step 1: UPS নির্ধারণ (আনুপাতিক পদ্ধতিতে)
    total_target = sum(targets)
    
    # বেস UPS বের করা
    base_ups = []
    for i in range(num_items):
        ups = max(1, round((targets[i] / total_target) * grid_size))
        base_ups.append(ups)
    
    # Grid size adjust
    diff = grid_size - sum(base_ups)
    if diff > 0:
        # যাদের target বেশি তাদের বাড়াই
        for _ in range(diff):
            max_idx = max(range(num_items), key=lambda i: targets[i])
            base_ups[max_idx] += 1
    elif diff < 0:
        # যাদের ups বেশি তাদের কমানো
        for _ in range(-diff):
            candidates = [i for i in range(num_items) if base_ups[i] > 1]
            if candidates:
                min_idx = min(candidates, key=lambda i: targets[i])
                base_ups[min_idx] -= 1
    
    # Step 2: শিট সংখ্যা ক্যালকুলেশন
    # আমরা চাই total_shit × total_ups_per_plate × num_plates ≈ targets
    total_ups_all_plates = sum(base_ups) * num_plates  # = grid_size × num_plates
    
    # প্রয়োজনীয় মোট শিট (আনুমানিক)
    approx_total_sheets = math.ceil(total_target / total_ups_all_plates)
    
    # Step 3: UPS ভ্যারিয়েশন তৈরি (যাতে প্রতি প্লেটে ভিন্ন UPS থাকে)
    # আপনার ম্যানুয়াল রিপোর্টের মতো করে
    plate_ups_list = []
    
    for p in range(num_plates):
        if p == 0:
            # প্রথম প্লেটে base_ups
            current_ups = base_ups.copy()
        else:
            # পরবর্তী প্লেটে UPS রি-ডিস্ট্রিবিউট
            current_ups = [0] * num_items
            
            # মোট জায়গা grid_size
            remaining_slots = grid_size
            
            # প্রথমে সব লেবেলের জন্য 1 করে দিই
            for i in range(num_items):
                current_ups[i] = 1
                remaining_slots -= 1
            
            # বাকি স্লটগুলো টার্গেট অনুপাতে ভাগ করি
            while remaining_slots > 0:
                max_idx = max(range(num_items), key=lambda i: targets[i] / (current_ups[i] + 1) if current_ups[i] + 1 > 0 else 0)
                current_ups[max_idx] += 1
                remaining_slots -= 1
        
        plate_ups_list.append(current_ups)
    
    # Step 4: শিট সংখ্যা নির্ধারণ (প্রতি প্লেটের জন্য আলাদা)
    remaining_targets = targets.copy()
    sheets_per_plate = [0] * num_plates
    final_produced = [0] * num_items
    
    for p in range(num_plates):
        if sum(remaining_targets) <= 0:
            break
        
        current_ups = plate_ups_list[p]
        
        # এই প্লেটের জন্য কত শিট লাগবে?
        max_sheets = 0
        for i in range(num_items):
            if current_ups[i] > 0 and remaining_targets[i] > 0:
                needed = math.ceil(remaining_targets[i] / current_ups[i])
                max_sheets = max(max_sheets, needed)
        
        # শিট সংখ্যা (অপটিমাইজড)
        sheets_per_plate[p] = max_sheets
        
        # উৎপাদন ও রিমেইনিং আপডেট
        for i in range(num_items):
            produced = current_ups[i] * max_sheets
            final_produced[i] += produced
            remaining_targets[i] = max(0, remaining_targets[i] - produced)
    
    # Step 5: যদি কিছু রিমেইনিং থেকে যায়, শেষ প্লেটে বাড়াই
    if sum(remaining_targets) > 0 and num_plates > 0:
        # শেষ প্লেটে বাড়তি শিট যোগ করি
        last_plate = num_plates - 1
        extra_sheets = 0
        for i in range(num_items):
            if remaining_targets[i] > 0 and plate_ups_list[last_plate][i] > 0:
                needed = math.ceil(remaining_targets[i] / plate_ups_list[last_plate][i])
                extra_sheets = max(extra_sheets, needed)
        
        if extra_sheets > 0:
            sheets_per_plate[last_plate] += extra_sheets
            for i in range(num_items):
                final_produced[i] += plate_ups_list[last_plate][i] * extra_sheets
    
    # Excess বের করা
    excess_list = [final_produced[i] - targets[i] for i in range(num_items)]
    
    # ডাটা প্যাক করা
    plate_ups_data = {}
    for p in range(num_plates):
        plate_ups_data[f"Plate {p+1}"] = plate_ups_list[p]
    
    return plate_ups_data, sheets_per_plate, {
        "total_produced": final_produced,
        "excess": excess_list
    }


def find_best_plate_count_exact(labels_input, grid_size, max_plates):
    """বিভিন্ন প্লেট সংখ্যা ট্রাই করে সবচেয়ে কম Excess যেটায় হয় সেটা বের করা"""
    targets = [l["Target QTY"] for l in labels_input]
    total_target = sum(targets)
    best_result = None
    best_excess = float('inf')
    all_results = []
    
    for p in range(1, max_plates + 1):
        plate_ups_data, sheets_info, calc_data = calculate_exact_manual_strategy(labels_input, grid_size, p)
        
        total_produced = sum(calc_data["total_produced"])
        total_excess = sum(max(0, e) for e in calc_data["excess"])
        total_shortage = sum(max(0, -e) for e in calc_data["excess"])
        
        excess_percent = (total_excess / total_target * 100) if total_target > 0 else 0
        
        # স্কোর: Excess + (Shortage × 5) + (প্লেট সংখ্যা × 5)
        score = total_excess + (total_shortage * 5) + (p * 5)
        
        all_results.append({
            "num_plates": p,
            "total_excess": total_excess,
            "total_shortage": total_shortage,
            "total_produced": total_produced,
            "total_target": total_target,
            "excess_percent": round(excess_percent, 2),
            "plate_ups_data": plate_ups_data,
            "plate_sheets_info": sheets_info,
            "calc_data": calc_data
        })
        
        if score < best_excess:
            best_excess = score
            best_result = all_results[-1]
    
    return best_result, all_results


# --- Main Button ---
if st.button("ক্যালকুলেট করুন"):
    
    if auto_plate_mode:
        with st.spinner("🤖 আপনার ম্যানুয়াল কৌশল অনুসরণ করে সেরা প্লেট সংখ্যা বের করা হচ্ছে..."):
            best_result, all_results = find_best_plate_count_exact(labels_input, grid_size, max_plates_to_try)
        
        if best_result is None:
            st.error("ক্যালকুলেশন ব্যর্থ!")
            st.stop()
        
        num_plates_used = best_result["num_plates"]
        plate_ups_data = best_result["plate_ups_data"]
        plate_sheets_info = best_result["plate_sheets_info"]
        calc_data = best_result["calc_data"]
        
        # তুলনামূলক টেবিল
        st.divider()
        st.subheader("🤖 অটো প্লেট রেশিও ফাইন্ডার রিপোর্ট")
        
        comparison_data = []
        for res in all_results:
            comparison_data.append({
                "প্লেট সংখ্যা": res["num_plates"],
                "মোট উৎপাদন": res["total_produced"],
                "মোট টার্গেট": res["total_target"],
                "এক্সেস": res["total_excess"],
                "শর্টেজ": res["total_shortage"],
                "এক্সেস (%)": res["excess_percent"]
            })
        
        df_compare = pd.DataFrame(comparison_data)
        st.dataframe(df_compare, use_container_width=True)
        
        if best_result["total_shortage"] > 0:
            st.error(f"⚠️ {best_result['total_shortage']} পিস কম উৎপাদন হয়েছে! প্লেট সংখ্যা বাড়ানোর চেষ্টা করুন।")
        elif best_result["total_excess"] == 0:
            st.success(f"✅ **সেরা প্লেট সংখ্যা: {num_plates_used} টি** (Perfect — কোনো Excess নেই!)")
            st.balloons()
        else:
            st.success(f"✅ **সেরা প্লেট সংখ্যা: {num_plates_used} টি**")
            st.info(f"📊 **এক্সেস:** {best_result['total_excess']} পিস ({best_result['excess_percent']}%) | **উৎপাদন:** {best_result['total_produced']} / {best_result['total_target']}")
    
    else:
        # Manual mode
        num_plates_used = num_plates_manual
        plate_ups_data, plate_sheets_info, calc_data = calculate_exact_manual_strategy(labels_input, grid_size, num_plates_used)
    
    # --- ফাইনাল রিপোর্ট ---
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
        
        over_print_pct = round((calc_data["excess"][i] / labels_input[i]["Original QTY"] * 100), 2) if labels_input[i]["Original QTY"] > 0 else 0
        row["Over Print (%)"] = over_print_pct
        
        final_data.append(row)
    
    df_final = pd.DataFrame(final_data)
    
    # Total row
    total_row = {"Name": "TOTAL"}
    for col in df_final.columns:
        if col != "Name" and "Ups" not in col:
            if df_final[col].dtype in ['int64', 'float64']:
                total_row[col] = df_final[col].sum()
        elif "Ups" in col:
            total_row[col] = df_final[col].sum()
    
    df_with_total = pd.concat([df_final, pd.DataFrame([total_row])], ignore_index=True)
    st.dataframe(df_with_total, use_container_width=True)
    
    # শিট ইনস্ট্রাকশন
    st.write("### 📝 প্রিন্টিং ইনস্ট্রাকশন:")
    cols_info = st.columns(num_plates_used)
    for p in range(num_plates_used):
        with cols_info[p]:
            ups_total = sum(plate_ups_data[f"Plate {p+1}"])
            st.info(f"**Plate {p+1}:** {plate_sheets_info[p]} শিট প্রিন্ট করতে হবে। (প্রতি শিটে {ups_total}টি লেবেল)")
    
    # Final verdict
    total_original = sum(l["Original QTY"] for l in labels_input)
    total_excess = sum(max(0, e) for e in calc_data["excess"])
    total_shortage = sum(max(0, -e) for e in calc_data["excess"])
    final_over_print_pct = round((total_excess / total_original * 100), 2) if total_original > 0 else 0
    
    if total_shortage > 0:
        st.error(f"❌ {total_shortage} পিস কম উৎপাদন হয়েছে! অনুগ্রহ করে প্লেট সংখ্যা বাড়ান।")
    elif total_excess == 0:
        st.success("✅ পারফেক্ট! 0% Over Print — অভিনন্দন!")
    elif final_over_print_pct <= 2:
        st.success(f"✅ খুব ভালো! মাত্র {final_over_print_pct}% Over Print — ইন্ডাস্ট্রি স্ট্যান্ডার্ডের মধ্যে।")
    elif final_over_print_pct <= 5:
        st.warning(f"⚠️ {final_over_print_pct}% Over Print — গ্রহণযোগ্য কিন্তু আরও কমানো সম্ভব।")
    else:
        st.error(f"❌ {final_over_print_pct}% Over Print — খুব বেশি! প্লেট সংখ্যা বা UPS কনফিগারেশন পরিবর্তন করুন।")
