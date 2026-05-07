import streamlit as st
import math
import pandas as pd
import random

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
# CORRECT ALGORITHM - Based on Your Manual Strategy
# ============================================

def distribute_ups_variable(targets, grid_size, num_plates):
    """
    প্রতিটি প্লেটের জন্য আলাদা UPS ডিস্ট্রিবিউশন তৈরি করে
    যাতে মোট UPS সব প্লেট মিলিয়ে num_plates * grid_size হয়
    কিন্তু প্রতিটি প্লেটের UPS pattern ভিন্ন
    """
    num_items = len(targets)
    total_target = sum(targets)
    
    # বেস UPS (আনুপাতিক)
    base_ups = []
    for i in range(num_items):
        ups = max(1, round((targets[i] / total_target) * grid_size))
        base_ups.append(ups)
    
    # গ্রিড সাইজ অ্যাডজাস্ট
    diff = grid_size - sum(base_ups)
    if diff > 0:
        for _ in range(diff):
            max_idx = max(range(num_items), key=lambda i: targets[i])
            base_ups[max_idx] += 1
    
    # বিভিন্ন প্লেটের জন্য UPS ভ্যারিয়েশন তৈরি
    all_plate_ups = []
    
    for plate in range(num_plates):
        if plate == 0:
            all_plate_ups.append(base_ups.copy())
        else:
            # আগের প্লেট থেকে ভিন্ন UPS তৈরি
            new_ups = base_ups.copy()
            
            # কিছু UPS কমিয়ে, অন্যগুলো বাড়িয়ে
            # যাতে মোট যোগফল grid_size থাকে
            for _ in range(random.randint(1, grid_size // 2)):
                # কমাতে হবে যেখানে >1
                candidates_dec = [i for i, v in enumerate(new_ups) if v > 1]
                if candidates_dec:
                    dec_idx = random.choice(candidates_dec)
                    new_ups[dec_idx] -= 1
                    
                    # বাড়াতে হবে
                    inc_idx = random.randint(0, num_items - 1)
                    new_ups[inc_idx] += 1
            
            # ফাইনাল অ্যাডজাস্ট
            diff2 = grid_size - sum(new_ups)
            if diff2 > 0:
                for _ in range(diff2):
                    max_idx = max(range(num_items), key=lambda i: targets[i])
                    new_ups[max_idx] += 1
            elif diff2 < 0:
                for _ in range(-diff2):
                    min_idx = min(range(num_items), key=lambda i: new_ups[i] if new_ups[i] > 1 else 999999)
                    if new_ups[min_idx] > 1:
                        new_ups[min_idx] -= 1
            
            all_plate_ups.append(new_ups)
    
    return all_plate_ups


def calculate_sheets_variable(all_plate_ups, targets):
    """
    প্রতিটি প্লেটের জন্য আলাদা শিট সংখ্যা বের করে
    """
    num_plates = len(all_plate_ups)
    num_items = len(targets)
    
    remaining = targets.copy()
    sheets_per_plate = []
    final_produced = [0] * num_items
    
    for p in range(num_plates):
        ups = all_plate_ups[p]
        
        # এই প্লেটের জন্য কত শিট লাগবে?
        max_sheets = 0
        for i in range(num_items):
            if ups[i] > 0 and remaining[i] > 0:
                needed = math.ceil(remaining[i] / ups[i])
                max_sheets = max(max_sheets, needed)
        
        # যদি সব শেষ হয়ে যায়, তাহলে শিট 0
        if sum(remaining) == 0:
            sheets_per_plate.append(0)
            continue
        
        # শিট সংখ্যা এডজাস্ট (অপটিমাইজেশনের জন্য)
        # কখনো কখনো একটু বেশি শিট নিলে পরের প্লেট কম লাগে
        sheets_per_plate.append(max_sheets)
        
        # উৎপাদন ও রিমেইনিং আপডেট
        for i in range(num_items):
            produced = ups[i] * max_sheets
            final_produced[i] += produced
            remaining[i] = max(0, remaining[i] - produced)
    
    return sheets_per_plate, final_produced, remaining


def optimize_plates_smart(labels_input, grid_size, num_plates, iterations=100):
    """
    স্মার্ট অপটিমাইজেশন - বিভিন্ন UPS কম্বিনেশন ট্রাই করে
    সবচেয়ে কম Excess ওভারঅল যেটায় হয় সেটা বেছে নেয়
    """
    targets = [l["Target QTY"] for l in labels_input]
    best_solution = None
    best_excess = float('inf')
    best_details = None
    
    for iteration in range(iterations):
        # আলাদা UPS ডিস্ট্রিবিউশন তৈরি
        all_plate_ups = distribute_ups_variable(targets, grid_size, num_plates)
        
        # শিট সংখ্যা বের করা
        sheets_per_plate, final_produced, remaining = calculate_sheets_variable(all_plate_ups, targets)
        
        # Excess বের করা
        excess = 0
        for i in range(len(targets)):
            if final_produced[i] > targets[i]:
                excess += (final_produced[i] - targets[i])
            elif final_produced[i] < targets[i]:
                # কম উৎপাদন হলে penalty বেশি
                excess += (targets[i] - final_produced[i]) * 2
        
        # Shortage check
        shortage = sum(remaining)
        
        # মোট স্কোর
        score = excess + shortage
        
        if score < best_excess:
            best_excess = score
            best_solution = {
                "plate_ups": all_plate_ups,
                "sheets": sheets_per_plate,
                "produced": final_produced,
                "excess_list": [final_produced[i] - targets[i] for i in range(len(targets))],
                "total_excess": excess,
                "shortage": shortage
            }
    
    return best_solution


def find_best_plate_count_smart(labels_input, grid_size, max_plates):
    """বিভিন্ন প্লেট সংখ্যার মধ্যে সেরাটা বের করা"""
    targets = [l["Target QTY"] for l in labels_input]
    total_target = sum(targets)
    best_result = None
    best_score = float('inf')
    all_results = []
    
    for p in range(1, max_plates + 1):
        solution = optimize_plates_smart(labels_input, grid_size, p, iterations=80)
        
        if not solution:
            continue
        
        total_produced = sum(solution["produced"])
        total_excess = solution["total_excess"]
        shortage = solution["shortage"]
        
        excess_percent = (total_excess / total_target * 100) if total_target > 0 else 0
        
        # স্কোর: কম excess ভালো, কম shortage ভালো, কম প্লেট ভালো
        score = total_excess + (shortage * 3) + (p * 10)
        
        all_results.append({
            "num_plates": p,
            "total_excess": total_excess,
            "total_produced": total_produced,
            "total_target": total_target,
            "excess_percent": round(excess_percent, 2),
            "shortage": shortage,
            "solution": solution
        })
        
        if score < best_score:
            best_score = score
            best_result = all_results[-1]
    
    return best_result, all_results


# --- Main Button ---
if st.button("ক্যালকুলেট করুন"):
    
    if auto_plate_mode:
        with st.spinner("🤖 সেরা প্লেট সংখ্যা বের করা হচ্ছে (৮০টি ভিন্ন কৌশল ট্রাই করছি)..."):
            best_result, all_results = find_best_plate_count_smart(labels_input, grid_size, max_plates_to_try)
        
        if best_result is None:
            st.error("ক্যালকুলেশন ব্যর্থ! আবার চেষ্টা করুন।")
            st.stop()
        
        num_plates_used = best_result["num_plates"]
        solution = best_result["solution"]
        plate_ups_data = {}
        for p in range(num_plates_used):
            plate_ups_data[f"Plate {p+1}"] = solution["plate_ups"][p]
        plate_sheets_info = solution["sheets"]
        calc_data = {
            "total_produced": solution["produced"],
            "excess": solution["excess_list"]
        }
        
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
                "এক্সেস (%)": res["excess_percent"],
                "শর্টেজ": res["shortage"]
            })
        
        df_compare = pd.DataFrame(comparison_data)
        st.dataframe(df_compare, use_container_width=True)
        
        if best_result["shortage"] > 0:
            st.error(f"⚠️ {best_result['shortage']} পিস কম উৎপাদন হয়েছে! প্লেট সংখ্যা বাড়ানোর চেষ্টা করুন।")
        elif best_result["total_excess"] == 0:
            st.success(f"✅ **সেরা প্লেট সংখ্যা: {num_plates_used} টি** (Perfect — কোনো Excess নেই!)")
            st.balloons()
        else:
            st.success(f"✅ **সেরা প্লেট সংখ্যা: {num_plates_used} টি** (Excess: {best_result['total_excess']} পিস, {best_result['excess_percent']}%)")
        
        st.info(f"📊 **মোট উৎপাদন:** {best_result['total_produced']} পিস | **টার্গেট:** {best_result['total_target']} পিস")
    
    else:
        # Manual mode - আপনার নিজের সেট করা প্লেট সংখ্যা
        num_plates_used = num_plates_manual
        solution = optimize_plates_smart(labels_input, grid_size, num_plates_used, iterations=100)
        
        if solution is None:
            st.error("ক্যালকুলেশন ব্যর্থ!")
            st.stop()
        
        plate_ups_data = {}
        for p in range(num_plates_used):
            plate_ups_data[f"Plate {p+1}"] = solution["plate_ups"][p]
        plate_sheets_info = solution["sheets"]
        calc_data = {
            "total_produced": solution["produced"],
            "excess": solution["excess_list"]
        }
    
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
            shits = plate_sheets_info[p]
            ups_total = sum(plate_ups_data[f"Plate {p+1}"])
            st.info(f"**Plate {p+1}:** {shits} শিট প্রিন্ট করতে হবে।\n\n(প্রতি শিটে {ups_total}টি লেবেল, গ্রিড {grid_size})")
    
    # Final verdict
    total_excess = sum(max(0, e) for e in calc_data["excess"])
    total_shortage = sum(max(0, -e) for e in calc_data["excess"])
    
    if total_shortage > 0:
        st.error(f"❌ {total_shortage} পিস কম উৎপাদন হয়েছে! প্লেট সংখ্যা বাড়ান বা UPS রেশিও পরিবর্তন করুন।")
    elif total_excess == 0:
        st.success("✅ পারফেক্ট! সব লেবেল ঠিক টার্গেট অনুযায়ী উৎপাদন হয়েছে।")
        st.balloons()
    else:
        st.warning(f"📈 মোট {total_excess} পিস অতিরিক্ত প্রিন্ট হয়েছে ({round(total_excess/sum(l['Original QTY'] for l in labels_input)*100, 1)}% Over Print)। এটি গ্রহণযোগ্য কিনা যাচাই করুন।")
