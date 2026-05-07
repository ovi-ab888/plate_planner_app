import streamlit as st
import math
import pandas as pd
import itertools

st.set_page_config(page_title="Pre-Press Optimizer Pro", layout="wide")

st.title("🎨 Pre-Press Grid & Plate Optimizer")
st.caption("🤖 AI-Inspired Plate Ratio Optimization")

# --- Sidebar ---
st.sidebar.header("কনফিগারেশন")
grid_size = st.sidebar.number_input("একটি প্লেটে মোট কয়টি লেবেল (Grid Size)?", min_value=1, value=30)
extra_percent = st.sidebar.number_input("অ্যাড-অন (Extra %) কত হবে?", min_value=0.0, value=0.0)
num_labels = st.sidebar.number_input("মোট কত পদের লেবেল?", min_value=1, value=1, step=1)

st.sidebar.subheader("প্লেট সেটিংস")
auto_plate_mode = st.sidebar.checkbox("🤖 AI Auto Plate Ratio Finder", value=True)

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
# AI-INSPIRED INTELLIGENT ALGORITHM
# ============================================

def ai_inspired_optimization(targets, grid_size, num_plates):
    """
    AI-ইন্সপায়ার্ড লজিক:
    1. প্রথমে সম্ভাব্য সব UPS কম্বিনেশন জেনারেট করে (ইন্টেলিজেন্টভাবে)
    2. প্রতিটি কম্বিনেশনের জন্য শিট সংখ্যা ক্যালকুলেট করে
    3. সবচেয়ে কম Excess যেটায় হয় সেটা বেছে নেয়
    """
    num_items = len(targets)
    total_target = sum(targets)
    
    # Step 1: বেস UPS বের করা (প্রপোরশনাল)
    base_ups = []
    for i in range(num_items):
        ups = max(1, round((targets[i] / total_target) * grid_size))
        base_ups.append(ups)
    
    # Adjust to grid size
    diff = grid_size - sum(base_ups)
    if diff > 0:
        for _ in range(diff):
            max_idx = max(range(num_items), key=lambda i: targets[i])
            base_ups[max_idx] += 1
    
    # Step 2: বিভিন্ন UPS ভ্যারিয়েশন তৈরি করা
    # এখানে আমরা Genetic Algorithm এর মতো কাজ করি
    best_solution = None
    best_excess = float('inf')
    best_details = None
    
    # প্লেটের সংখ্যা অনুযায়ী UPS ভ্যারিয়েশন
    ups_variations = [base_ups.copy()]
    
    # ভিন্ন ভিন্ন UPS তৈরি করা
    for _ in range(num_plates * 10):
        new_ups = base_ups.copy()
        # র‍্যান্ডম শাফল
        for __ in range(random.randint(1, grid_size // 3)):
            i, j = random.sample(range(num_items), 2)
            if new_ups[i] > 1:
                new_ups[i] -= 1
                new_ups[j] += 1
        
        # Grid size adjust
        diff2 = grid_size - sum(new_ups)
        if diff2 != 0:
            if diff2 > 0:
                for _ in range(diff2):
                    max_idx = max(range(num_items), key=lambda i: targets[i])
                    new_ups[max_idx] += 1
            else:
                for _ in range(-diff2):
                    candidates = [i for i in range(num_items) if new_ups[i] > 1]
                    if candidates:
                        min_idx = min(candidates, key=lambda i: targets[i])
                        new_ups[min_idx] -= 1
        
        if new_ups not in ups_variations:
            ups_variations.append(new_ups)
    
    # Step 3: প্রতিটি কম্বিনেশন চেষ্টা করা
    # আমরা প্রতি প্লেটের জন্য আলাদা UPS কম্বিনেশন ব্যবহার করব
    
    # সব সম্ভাব্য কম্বিনেশন (প্লেট数量 পর্যন্ত)
    from itertools import combinations_with_replacement
    
    best_score = float('inf')
    
    # আমরা ট্রাই করব বিভিন্ন কম্বিনেশন
    for p in range(1, num_plates + 1):
        # p প্লেটের জন্য UPS কম্বিনেশন বাছাই
        for combo in combinations_with_replacement(ups_variations, p):
            # ডুপ্লিকেট কম্বিনেশন এড়িয়ে যাও
            if len(set(str(c) for c in combo)) == 1 and p > 1:
                # সব প্লেটের UPS একই হলে skip (কারণ এটা অপটিমাল না)
                if p > 1:
                    continue
            
            # এই কম্বিনেশনের জন্য শিট সংখ্যা বের করা
            remaining = targets.copy()
            sheets_list = []
            final_produced = [0] * num_items
            
            for plate_idx, ups in enumerate(combo):
                if sum(remaining) <= 0:
                    sheets_list.append(0)
                    continue
                
                # কত শিট লাগবে?
                max_sheets = 0
                for i in range(num_items):
                    if ups[i] > 0 and remaining[i] > 0:
                        needed = math.ceil(remaining[i] / ups[i])
                        max_sheets = max(max_sheets, needed)
                
                sheets_list.append(max_sheets)
                
                for i in range(num_items):
                    produced = ups[i] * max_sheets
                    final_produced[i] += produced
                    remaining[i] = max(0, remaining[i] - produced)
            
            # স্কোর বের করা
            total_produced = sum(final_produced)
            total_excess = sum(max(0, final_produced[i] - targets[i]) for i in range(num_items))
            total_shortage = sum(max(0, targets[i] - final_produced[i]) for i in range(num_items))
            
            # স্কোর = excess + (shortage × 10) + (প্লেট সংখ্যা × 5)
            score = total_excess + (total_shortage * 10) + (p * 5)
            
            if score < best_score:
                best_score = score
                best_solution = {
                    "num_plates": p,
                    "ups_list": combo,
                    "sheets": sheets_list,
                    "produced": final_produced,
                    "excess_list": [final_produced[i] - targets[i] for i in range(num_items)],
                    "total_excess": total_excess,
                    "total_shortage": total_shortage,
                    "total_produced": total_produced
                }
    
    return best_solution


def find_best_with_ai(labels_input, grid_size, max_plates):
    """AI পদ্ধতিতে সেরা প্লেট সংখ্যা বের করা"""
    targets = [l["Target QTY"] for l in labels_input]
    total_target = sum(targets)
    best_result = None
    best_score = float('inf')
    all_results = []
    
    for p in range(1, max_plates + 1):
        st.progress(p / max_plates, text=f"AI চিন্তা করছে... {p}/{max_plates} প্লেট ট্রাই করছি")
        
        solution = ai_inspired_optimization(targets, grid_size, p)
        
        if not solution:
            continue
        
        excess_percent = (solution["total_excess"] / total_target * 100) if total_target > 0 else 0
        
        all_results.append({
            "num_plates": p,
            "total_excess": solution["total_excess"],
            "total_shortage": solution["total_shortage"],
            "total_produced": solution["total_produced"],
            "total_target": total_target,
            "excess_percent": round(excess_percent, 2),
            "solution": solution
        })
        
        # স্কোর: কম excess + কম shortage
        score = solution["total_excess"] + (solution["total_shortage"] * 10)
        if score < best_score:
            best_score = score
            best_result = all_results[-1]
    
    return best_result, all_results


# --- Main Button ---
if st.button("🤖 AI দিয়ে ক্যালকুলেট করুন"):
    
    targets = [l["Target QTY"] for l in labels_input]
    
    if auto_plate_mode:
        with st.spinner("🧠 AI বিভিন্ন প্লেট রেশিও অ্যানালাইসিস করছে..."):
            best_result, all_results = find_best_with_ai(labels_input, grid_size, max_plates_to_try)
        
        if best_result is None:
            st.error("AI ফলাফল দিতে ব্যর্থ! আবার চেষ্টা করুন।")
            st.stop()
        
        num_plates_used = best_result["num_plates"]
        solution = best_result["solution"]
        
        # ফলাফল দেখানো
        st.divider()
        st.subheader("🤖 AI অ্যানালাইসিস রিপোর্ট")
        
        comparison_data = []
        for res in all_results:
            comparison_data.append({
                "প্লেট সংখ্যা": res["num_plates"],
                "মোট উৎপাদন": res["total_produced"],
                "টার্গেট": res["total_target"],
                "এক্সেস": res["total_excess"],
                "শর্টেজ": res["total_shortage"],
                "এক্সেস (%)": res["excess_percent"]
            })
        
        df_compare = pd.DataFrame(comparison_data)
        st.dataframe(df_compare, use_container_width=True)
        
        if best_result["total_shortage"] > 0:
            st.error(f"⚠️ {best_result['total_shortage']} পিস কম উৎপাদন!")
        else:
            st.success(f"✅ **AI সেরা প্লেট সংখ্যা বের করেছে: {num_plates_used} টি**")
            st.info(f"📊 **এক্সেস:** {best_result['total_excess']} পিস ({best_result['excess_percent']}%)")
            
            if best_result['excess_percent'] <= 2:
                st.balloons()
                st.success("🎉 অসাধারণ! 2% এর নিচে Over Print!")
        
        # ডিটেইল্ড ফলাফল তৈরি
        plate_ups_data = {}
        for p in range(num_plates_used):
            plate_ups_data[f"Plate {p+1}"] = solution["ups_list"][p]
        
        calc_data = {
            "total_produced": solution["produced"],
            "excess": solution["excess_list"]
        }
        
    else:
        # Manual mode
        num_plates_used = num_plates_manual
        solution = ai_inspired_optimization(targets, grid_size, num_plates_used)
        
        if solution is None:
            st.error("ক্যালকুলেশন ব্যর্থ!")
            st.stop()
        
        plate_ups_data = {}
        for p in range(num_plates_used):
            plate_ups_data[f"Plate {p+1}"] = solution["ups_list"][p]
        
        calc_data = {
            "total_produced": solution["produced"],
            "excess": solution["excess_list"]
        }
        plate_sheets_info = solution["sheets"]
    
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
    if 'solution' in locals():
        st.write("### 📝 প্রিন্টিং ইনস্ট্রাকশন (AI সাজেস্টেড):")
        cols_info = st.columns(num_plates_used)
        for p in range(num_plates_used):
            with cols_info[p]:
                ups_total = sum(plate_ups_data[f"Plate {p+1}"])
                st.info(f"**Plate {p+1}:** {solution['sheets'][p]} শিট প্রিন্ট করতে হবে।\n\n(প্রতি শিটে {ups_total}টি লেবেল)")
    
    # Final verdict
    total_original = sum(l["Original QTY"] for l in labels_input)
    total_excess = sum(max(0, e) for e in calc_data["excess"])
    final_over_print_pct = round((total_excess / total_original * 100), 2) if total_original > 0 else 0
    
    if final_over_print_pct <= 2:
        st.success(f"✅ অসাধারণ! মাত্র {final_over_print_pct}% Over Print — AI সফল হয়েছে!")
        st.balloons()
    elif final_over_print_pct <= 5:
        st.info(f"📈 {final_over_print_pct}% Over Print — গ্রহণযোগ্য স্তরে।")
    else:
        st.error(f"❌ {final_over_print_pct}% Over Print — এখনও বেশি। AI আরও ট্রায়াল দরকার।")
