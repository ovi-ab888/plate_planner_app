import streamlit as st
import math
import pandas as pd
import requests
import json
import time

st.set_page_config(page_title="Pre-Press Optimizer Pro - AI Edition", layout="wide")

st.title("🎨 Pre-Press Grid & Plate Optimizer")
st.caption("🤖 Powered by DeepSeek AI - Intelligent Plate Ratio Optimization")

# --- Sidebar ---
st.sidebar.header("⚙️ কনফিগারেশন")

# API Key input
st.sidebar.subheader("🔑 DeepSeek API Configuration")
api_key = st.sidebar.text_input("DeepSeek API Key", type="password", 
                                 help="platform.deepseek.com থেকে API key নিন")
use_ai = st.sidebar.checkbox("🤖 AI Mode (DeepSeek চালু করুন)", value=True)

st.sidebar.divider()

grid_size = st.sidebar.number_input("একটি প্লেটে মোট কয়টি লেবেল (Grid Size)?", min_value=1, value=30)
extra_percent = st.sidebar.number_input("অ্যাড-অন (Extra %) কত হবে?", min_value=0.0, value=0.0)
num_labels = st.sidebar.number_input("মোট কত পদের লেবেল?", min_value=1, value=1, step=1)

st.sidebar.subheader("প্লেট সেটিংস")
if not use_ai:
    num_plates_manual = st.sidebar.number_input("কয়টি প্লেট করতে চান?", min_value=1, value=2, step=1)
else:
    max_plates_to_try = st.sidebar.slider("সর্বোচ্চ কতটি প্লেট ট্রাই করবেন?", min_value=1, max_value=6, value=3)
    st.sidebar.info("🤖 AI Mode অন থাকায় প্লেট সংখ্যা অটোমেটিক ডিটার্মাইন হবে")

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
# DEEPSEEK AI INTEGRATION
# ============================================

def call_deepseek_for_optimization(targets, grid_size, num_plates, api_key):
    """
    DeepSeek API কে কল করে প্লেট রেশিও বের করা
    """
    total_target = sum(targets)
    
    # Prompt তৈরি করা
    prompt = f"""You are a Pre-Press optimization expert. I need the optimal plate ratio calculation.

LABEL QUANTITIES (Target): {targets}
GRID SIZE per plate: {grid_size} (total slots per plate)
NUMBER OF PLATES: {num_plates}

CALCULATION RULES:
1. Each plate must have exactly {grid_size} total UPS (sum of all label ups per plate)
2. Each label should get at least 1 UPS per plate
3. Different plates can have different UPS distribution
4. Calculate sheets per plate independently
5. Goal: Minimize overprint (excess) while ensuring all targets are met

Return ONLY valid JSON in this exact format:
{{
    "plate_ups": [
        [ups_for_label1, ups_for_label2, ...],  // Plate 1
        [ups_for_label1, ups_for_label2, ...],  // Plate 2
        ...
    ],
    "sheets": [sheets_for_plate1, sheets_for_plate2, ...]
}}

The UPS per plate must sum to {grid_size}. Make the optimization smart - different plates can have different UPS patterns to minimize excess.

Calculate carefully. Return ONLY the JSON, no explanation."""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a precise pre-press optimization expert. Always return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,  # কম temperature = বেশি precise output
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"]
            
            # JSON extract করার চেষ্টা
            # Sometimes AI extra text add করে, তাই JSON অংশ বের করা
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                ai_json = json.loads(json_match.group())
                return ai_json
            else:
                return None
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"API Call Failed: {str(e)}")
        return None


def calculate_with_ai_ups(plate_ups_list, sheets_list, targets):
    """
    AI থেকে পাওয়া UPS এবং Sheets দিয়ে উৎপাদন ক্যালকুলেশন
    """
    num_plates = len(plate_ups_list)
    num_items = len(targets)
    
    final_produced = [0] * num_items
    
    for p in range(num_plates):
        ups = plate_ups_list[p]
        sheets = sheets_list[p] if p < len(sheets_list) else 0
        
        for i in range(num_items):
            final_produced[i] += ups[i] * sheets
    
    excess_list = [final_produced[i] - targets[i] for i in range(num_items)]
    
    return {
        "total_produced": final_produced,
        "excess": excess_list,
        "sheets": sheets_list
    }


def manual_optimization_algorithm(targets, grid_size, num_plates):
    """
    Fallback algorithm - যখন API কাজ করে না তখন ব্যবহার হবে
    """
    num_items = len(targets)
    total_target = sum(targets)
    
    # Base UPS (প্রপোরশনাল)
    base_ups = []
    for i in range(num_items):
        ups = max(1, round((targets[i] / total_target) * grid_size))
        base_ups.append(ups)
    
    # Grid size adjust
    diff = grid_size - sum(base_ups)
    if diff > 0:
        for _ in range(diff):
            max_idx = max(range(num_items), key=lambda i: targets[i])
            base_ups[max_idx] += 1
    
    # Different UPS variations for different plates
    plate_ups_list = []
    for p in range(num_plates):
        if p == 0:
            plate_ups_list.append(base_ups.copy())
        else:
            # Slight variation
            new_ups = base_ups.copy()
            # Shift some weight
            for _ in range(2):
                candidates = [i for i in range(num_items) if new_ups[i] > 1]
                if candidates:
                    from_idx = min(candidates, key=lambda i: targets[i])
                    to_idx = max(range(num_items), key=lambda i: targets[i])
                    if new_ups[from_idx] > 1:
                        new_ups[from_idx] -= 1
                        new_ups[to_idx] += 1
            plate_ups_list.append(new_ups)
    
    # Calculate sheets
    remaining = targets.copy()
    sheets_list = []
    
    for p in range(num_plates):
        if sum(remaining) <= 0:
            sheets_list.append(0)
            continue
        
        max_sheets = 0
        for i in range(num_items):
            if plate_ups_list[p][i] > 0 and remaining[i] > 0:
                needed = math.ceil(remaining[i] / plate_ups_list[p][i])
                max_sheets = max(max_sheets, needed)
        
        sheets_list.append(max_sheets)
        
        for i in range(num_items):
            remaining[i] = max(0, remaining[i] - (plate_ups_list[p][i] * max_sheets))
    
    return plate_ups_list, sheets_list


def find_best_plate_with_ai(labels_input, grid_size, max_plates, api_key):
    """
    AI দিয়ে সব প্লেট কম্বিনেশন ট্রাই করে সেরাটা বের করা
    """
    targets = [l["Target QTY"] for l in labels_input]
    total_target = sum(targets)
    best_result = None
    best_excess = float('inf')
    all_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for p in range(1, max_plates + 1):
        status_text.text(f"🤖 AI analysing with {p} plate(s)...")
        progress_bar.progress(p / max_plates)
        
        # Try AI first
        ai_result = None
        if api_key:
            ai_result = call_deepseek_for_optimization(targets, grid_size, p, api_key)
        
        if ai_result and "plate_ups" in ai_result and "sheets" in ai_result:
            plate_ups_list = ai_result["plate_ups"]
            sheets_list = ai_result["sheets"]
            calc_data = calculate_with_ai_ups(plate_ups_list, sheets_list, targets)
        else:
            # Fallback to manual algorithm
            if not api_key:
                status_text.text(f"⚠️ No API key, using manual algorithm for {p} plate(s)...")
            else:
                status_text.text(f"🔄 AI response invalid, using fallback for {p} plate(s)...")
            plate_ups_list, sheets_list = manual_optimization_algorithm(targets, grid_size, p)
            calc_data = calculate_with_ai_ups(plate_ups_list, sheets_list, targets)
        
        total_produced = sum(calc_data["total_produced"])
        total_excess = sum(max(0, e) for e in calc_data["excess"])
        total_shortage = sum(max(0, -e) for e in calc_data["excess"])
        excess_percent = (total_excess / total_target * 100) if total_target > 0 else 0
        
        all_results.append({
            "num_plates": p,
            "total_excess": total_excess,
            "total_shortage": total_shortage,
            "total_produced": total_produced,
            "total_target": total_target,
            "excess_percent": round(excess_percent, 2),
            "plate_ups_list": plate_ups_list,
            "sheets_list": sheets_list,
            "calc_data": calc_data
        })
        
        # Score: কম excess + কম shortage + কম প্লেট
        score = total_excess + (total_shortage * 10) + (p * 5)
        if score < best_excess:
            best_excess = score
            best_result = all_results[-1]
    
    progress_bar.empty()
    status_text.empty()
    
    return best_result, all_results


# --- Main Button ---
if st.button("🚀 ক্যালকুলেট করুন (AI চালু)"):
    
    targets = [l["Target QTY"] for l in labels_input]
    total_original = sum(l["Original QTY"] for l in labels_input)
    
    if use_ai:
        if not api_key:
            st.warning("⚠️ API Key দেওয়া নেই! AI Mode বন্ধ করে Manual Mode ব্যবহার করুন অথবা API Key দিন।")
            st.info("🔑 API Key পেতে: platform.deepseek.com এ রেজিস্ট্রেশন করে API Keys সেকশন থেকে নিতে পারেন।")
            
            # Fallback to manual with warning
            with st.spinner("Manual Algorithm দিয়ে ক্যালকুলেশন চলছে..."):
                best_result, all_results = find_best_plate_with_ai(labels_input, grid_size, max_plates_to_try, None)
        else:
            with st.spinner("🧠 DeepSeek AI চিন্তা করছে... অপটিমাল প্লেট রেশিও বের করছে..."):
                best_result, all_results = find_best_plate_with_ai(labels_input, grid_size, max_plates_to_try, api_key)
    else:
        # Manual mode
        with st.spinner("ক্যালকুলেশন চলছে..."):
            num_plates_used = num_plates_manual
            plate_ups_list, sheets_list = manual_optimization_algorithm(targets, grid_size, num_plates_used)
            calc_data = calculate_with_ai_ups(plate_ups_list, sheets_list, targets)
            
            total_produced = sum(calc_data["total_produced"])
            total_excess = sum(max(0, e) for e in calc_data["excess"])
            total_shortage = sum(max(0, -e) for e in calc_data["excess"])
            excess_percent = (total_excess / total_original * 100) if total_original > 0 else 0
            
            best_result = {
                "num_plates": num_plates_used,
                "total_excess": total_excess,
                "total_shortage": total_shortage,
                "total_produced": total_produced,
                "total_target": sum(targets),
                "excess_percent": round(excess_percent, 2),
                "plate_ups_list": plate_ups_list,
                "sheets_list": sheets_list,
                "calc_data": calc_data
            }
            all_results = [best_result]
    
    if best_result is None:
        st.error("❌ ক্যালকুলেশন ব্যর্থ! আবার চেষ্টা করুন।")
        st.stop()
    
    num_plates_used = best_result["num_plates"]
    plate_ups_list = best_result["plate_ups_list"]
    sheets_list = best_result["sheets_list"]
    calc_data = best_result["calc_data"]
    
    # --- তুলনামূলক রিপোর্ট (AI Mode এ) ---
    if use_ai and len(all_results) > 1:
        st.divider()
        st.subheader("🤖 DeepSeek AI - প্লেট রেশিও অ্যানালাইসিস রিপোর্ট")
        
        comparison_data = []
        for res in all_results:
            comparison_data.append({
                "প্লেট সংখ্যা": f"{res['num_plates']} 🏷️",
                "মোট উৎপাদন": res["total_produced"],
                "টার্গেট": res["total_target"],
                "এক্সেস (পিস)": res["total_excess"],
                "শর্টেজ (পিস)": res["total_shortage"],
                "Over Print (%)": res["excess_percent"]
            })
        
        df_compare = pd.DataFrame(comparison_data)
        st.dataframe(df_compare, use_container_width=True)
        
        # Highlight best
        if best_result["total_shortage"] > 0:
            st.error(f"⚠️ {best_result['total_shortage']} পিস কম উৎপাদন হয়েছে! AI আরও প্লেট সুজেস্ট করতে পারে।")
        else:
            st.success(f"✅ **DeepSeek AI সেরা প্লেট সংখ্যা বের করেছে: {num_plates_used} টি**")
            
            over_print = best_result["excess_percent"]
            if over_print <= 2:
                st.balloons()
                st.success(f"🎉 অসাধারণ! মাত্র {over_print}% Over Print — AI সফল হয়েছে!")
            elif over_print <= 5:
                st.info(f"📈 {over_print}% Over Print — ইন্ডাস্ট্রি স্ট্যান্ডার্ডের মধ্যে।")
            else:
                st.warning(f"⚠️ {over_print}% Over Print — AI এখনও শিখছে, manual adjustment লাগতে পারে।")
    
    # --- Plate UPS Data তৈরি ---
    plate_ups_data = {}
    for p in range(num_plates_used):
        plate_ups_data[f"Plate {p+1}"] = plate_ups_list[p]
    
    # --- ফাইনাল রিপোর্ট টেবিল ---
    st.divider()
    st.subheader("📊 ফাইনাল রিপোর্ট (প্রোডাকশন প্ল্যান)")
    
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
    total_row = {"Name": "📊 TOTAL"}
    for col in df_final.columns:
        if col != "Name":
            if "Ups" not in col:
                if df_final[col].dtype in ['int64', 'float64']:
                    total_row[col] = df_final[col].sum()
            else:
                total_row[col] = df_final[col].sum()
    
    df_with_total = pd.concat([df_final, pd.DataFrame([total_row])], ignore_index=True)
    st.dataframe(df_with_total, use_container_width=True)
    
    # --- প্রিন্টিং ইনস্ট্রাকশন ---
    st.write("### 📝 প্রিন্টিং ইনস্ট্রাকশন (AI সাজেস্টেড):")
    cols_info = st.columns(num_plates_used)
    for p in range(num_plates_used):
        with cols_info[p]:
            ups_total = sum(plate_ups_data[f"Plate {p+1}"])
            st.info(f"""
            **Plate {p+1}:** 
            - {sheets_list[p]} শিট প্রিন্ট করতে হবে
            - প্রতি শিটে {ups_total}টি লেবেল (Grid: {grid_size})
            - প্রিন্ট করুন → {ups_total * sheets_list[p]} পিস উৎপাদন হবে
            """)
    
    # --- Final Verdict ---
    total_original_sum = sum(l["Original QTY"] for l in labels_input)
    total_excess_sum = sum(max(0, e) for e in calc_data["excess"])
    final_over_print = round((total_excess_sum / total_original_sum * 100), 2) if total_original_sum > 0 else 0
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 টার্গেট", f"{best_result['total_target']:,} পিস")
    with col2:
        st.metric("🏭 মোট উৎপাদন", f"{best_result['total_produced']:,} পিস", 
                  delta=f"+{best_result['total_excess']}" if best_result['total_excess'] > 0 else None)
    with col3:
        st.metric("📊 Over Print", f"{final_over_print}%", 
                  delta="Good" if final_over_print <= 2 else "High" if final_over_print > 5 else "OK",
                  delta_color="normal" if final_over_print <= 5 else "inverse")
    
    if final_over_print <= 2:
        st.success("✅ **পারফেক্ট!** DeepSeek AI সফলভাবে অপটিমাল প্লেট রেশিও বের করেছে!")
    elif final_over_print <= 5:
        st.info("📈 **গ্রহণযোগ্য স্তরে** Over Print — প্রোডাকশন রান করতে পারেন।")
    else:
        st.warning("⚠️ **Over Print বেশি** — API Key চেক করুন বা প্লেট সংখ্যা ম্যানুয়ালি সেট করে ট্রাই করুন।")
