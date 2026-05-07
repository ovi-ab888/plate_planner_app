import streamlit as st
import math
import pandas as pd

# অ্যাপ কনফিগারেশন
st.set_page_config(page_title="Zero-Waste AI Pre-Press", layout="wide")

# ডার্ক থিম কাস্টমাইজেশন
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; }
    .stTable { background-color: #161b22 !important; border-radius: 10px; color: white !important; }
    table { color: white !important; border: 1px solid #30363d; }
    thead tr th { background-color: #21262d !important; color: #58a6ff !important; }
    label, p, h3 { color: #c9d1d9 !important; }
    .stInfo { background-color: #091e42; color: #e6edf3; border: 1px solid #1f6feb; }
    .stSuccess { background-color: #0f2d1e; color: #aff5b4; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎨 AI Pre-Press Optimizer (Final Zero-Waste)")
st.write("এটি আপনার ম্যানুয়াল রিপোর্টের লজিক ব্যবহার করে অপচয় ১-২% এর নিচে নামিয়ে আনবে।")

# --- Sidebar: Configuration ---
st.sidebar.header("⚙️ মাস্টার সেটিংস")
grid_size = st.sidebar.number_input("গ্রিড সাইজ (Grid Size)", min_value=1, value=30)
extra_percent = st.sidebar.number_input("অ্যাড-অন (Extra %)", min_value=0.0, value=0.0)
num_labels = st.sidebar.number_input("মোট লেবেল পদ", min_value=1, value=8)
num_plates = st.sidebar.number_input("প্লেট সংখ্যা", min_value=1, value=2)

# --- Data Input Section ---
st.subheader("📦 লেবেল কোয়ান্টিটি ইনপুট")
labels_input = []
cols = st.columns(2)
for i in range(int(num_labels)):
    col_idx = i % 2
    with cols[col_idx]:
        c1, c2 = st.columns([2, 1])
        l_name = c1.text_input(f"নাম {i+1}", value=f"Label {i+1}", key=f"name_{i}")
        l_qty = c2.number_input(f"QTY", min_value=1, value=500, key=f"qty_{i}")
        target = math.ceil(l_qty * (1 + extra_percent / 100))
        labels_input.append({"Name": l_name, "Original QTY": l_qty, "Target QTY": target})

# --- AI Optimization Engine ---
if st.button("🚀 স্মার্ট ক্যালকুলেশন রান করুন"):
    best_config = None
    min_total_excess = float('inf')
    
    # সার্চ রেঞ্জ নির্ধারণ (ম্যানুয়াল রিপোর্টের স্টাইল)
    total_q = sum(l['Target QTY'] for l in labels_input)
    avg_s = total_q / (grid_size * num_plates)
    
    # s1 হবে বড় শিট (বড় অর্ডারের জন্য), s2 হবে ছোট শিট (অ্যাডজাস্টমেন্টের জন্য)
    s1_range = range(max(10, int(avg_s * 0.5)), int(avg_s * 1.5) + 100, 2)
    s2_list = [38, 50, 75, 100, 150, 200] if num_plates > 1 else [0]

    # প্রোগ্রেস বার
    progress_bar = st.progress(0)
    
    for idx, s1 in enumerate(s1_range):
        for s2 in s2_list:
            if num_plates == 1: s2 = 0
            
            current_ups = {"P1": [0]*len(labels_input), "P2": [0]*len(labels_input)}
            possible_run = True
            
            for i, l in enumerate(labels_input):
                best_l_excess = float('inf')
                best_u_pair = (0, 0)
                
                # ঘর বণ্টন চেক
                for u1 in range(grid_size + 1):
                    for u2 in range(grid_size + 1) if num_plates > 1 else [0]:
                        if (u1 + u2) > (grid_size * num_plates): continue
                        
                        prod = (u1 * s1) + (u2 * s2)
                        if prod >= l["Target QTY"]:
                            excess = prod - l["Target QTY"]
                            if excess < best_l_excess:
                                best_l_excess = excess
                                best_u_pair = (u1, u2)
                            break # টার্গেট পূরণ হলে পরবর্তী লেবেলে যাও
                
                current_ups["P1"][i] = best_u_pair[0]
                current_ups["P2"][i] = best_u_pair[1]
            
            # গ্রিড সাইজ চেক (৩০ এর বেশি যেন না হয়)
            if sum(current_ups["P1"]) <= grid_size and sum(current_ups["P2"]) <= grid_size:
                total_excess = 0
                for i, l in enumerate(labels_input):
                    prod = (current_ups["P1"][i] * s1) + (current_ups["P2"][i] * s2)
                    total_excess += (prod - l["Target QTY"])
                
                if total_excess < min_total_excess:
                    min_total_excess = total_excess
                    best_config = {"s1": s1, "s2": s2, "ups": current_ups}
                    if min_total_excess == 0: break
        
        if idx % 10 == 0: progress_bar.progress(idx / len(s1_range))
    
    progress_bar.empty()

    if best_config:
        # --- ঘর ৩০ পূর্ণ করার লজিক (Grid Filling) ---
        for p_key in ["P1", "P2"]:
            p_name = "Plate 1" if p_key == "P1" else "Plate 2"
            used = sum(best_config["ups"][p_key])
            if used < grid_size:
                diff = grid_size - used
                # সবচেয়ে বড় টার্গেট যে লেবেলের তাকে বাকি ঘর দাও
                max_idx = 0
                for i in range(len(labels_input)):
                    if labels_input[i]["Target QTY"] > labels_input[max_idx]["Target QTY"]:
                        max_idx = i
                best_config["ups"][p_key][max_idx] += diff

        # --- রিপোর্ট টেবিল জেনারেশন ---
        final_data = []
        for i, l in enumerate(labels_input):
            u1 = best_config["ups"]["P1"][i]
            u2 = best_config["ups"]["P2"][i]
            prod = (u1 * best_config["s1"]) + (u2 * best_config["s2"])
            final_data.append({
                "Name": l["Name"],
                "Target QTY": l["Target QTY"],
                "Plate 1 (Ups)": u1,
                "Plate 2 (Ups)": u2,
                "Total Produced": prod,
                "Excess": prod - l["Target QTY"],
                "Over Print (%)": round(((prod - l["Target QTY"])/l["Target QTY"]*100), 4)
            })

        df_final = pd.DataFrame(final_data)
        
        # TOTAL Row
        total_row = {
            "Name": "TOTAL",
            "Target QTY": df_final["Target QTY"].sum(),
            "Plate 1 (Ups)": df_final["Plate 1 (Ups)"].sum(),
            "Plate 2 (Ups)": df_final["Plate 2 (Ups)"].sum(),
            "Total Produced": df_final["Total Produced"].sum(),
            "Excess": df_final["Excess"].sum(),
            "Over Print (%)": round((df_final["Excess"].sum() / df_final["Target QTY"].sum() * 100), 4)
        }
        
        st.subheader("📊 ফাইনাল রিপোর্ট (Cross-Plate AI Optimized)")
        st.table(pd.concat([df_final, pd.DataFrame([total_row])], ignore_index=True))

        # প্রিন্টিং ইনস্ট্রাকশন
        c1, c2 = st.columns(2)
        with c1: st.info(f"📍 **Plate 1:** {best_config['s1']} শিট প্রিন্ট করুন।")
        with c2: st.info(f"📍 **Plate 2:** {best_config['s2']} শিট প্রিন্ট করুন।")
        
        st.success(f"✅ AI সলিউশন পাওয়া গেছে! মোট অপচয় মাত্র {int(min_total_excess)} পিস।")
    else:
        st.error("দুঃখিত, এই কনফিগারেশনে কোনো রেশিও মেলানো সম্ভব হয়নি।")
