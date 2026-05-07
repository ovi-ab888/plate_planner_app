import streamlit as st
import math
import pandas as pd

st.set_page_config(page_title="AI Pre-Press Optimizer Pro", layout="wide")

# ডার্ক মোড ডিজাইন
st.markdown("""
    <style>
    .stApp { background-color: #000000; }
    .stTable { background-color: #1e1e1e !important; border-radius: 10px; color: white !important; }
    table { color: white !important; }
    label, p, h3 { color: white !important; }
    .stInfo { background-color: #0e2f44; color: #ffffff; border: 1px solid #17a2b8; }
    .stSuccess { background-color: #0b2e13; color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 AI-Powered Pre-Press Optimizer (Final)")
st.write("এই টুলটি AI লজিক ব্যবহার করে প্লেট প্রতি আলাদা শিট সংখ্যা এবং Ups নির্ধারণ করে অপচয় (Excess) সর্বনিম্ন রাখবে।")

# --- Sidebar ---
st.sidebar.header("⚙️ মাস্টার কনফিগারেশন")
grid_size = st.sidebar.number_input("গ্রিড সাইজ (Grid Size)", min_value=1, value=30)
extra_percent = st.sidebar.number_input("অ্যাড-অন (Extra %)", min_value=0.0, value=0.0)
num_labels = st.sidebar.number_input("মোট লেবেল পদ", min_value=1, value=8, step=1)
num_plates = st.sidebar.number_input("প্লেট সংখ্যা", min_value=1, value=2, step=1)

# --- Data Input ---
st.subheader("📦 লেবেল কোয়ান্টিটি ইনপুট")
labels_input = []
cols = st.columns(2)
for i in range(int(num_labels)):
    col_idx = i % 2
    with cols[col_idx]:
        c1, c2 = st.columns([2, 1])
        name = c1.text_input(f"নাম {i+1}", value=f"Label {i+1}", key=f"n_{i}")
        qty = c2.number_input(f"QTY", min_value=1, value=1000, key=f"q_{i}")
        target = math.ceil(qty * (1 + extra_percent / 100))
        labels_input.append({"Name": name, "Original QTY": qty, "Target QTY": target})

if st.button("🤖 AI অপ্টিমাইজেশন রান করুন"):
    # --- AI Engine: Iterative Sheet & Ups Search ---
    best_overall_excess = float('inf')
    best_config = None
    
    # AI লজিক: এটি বিভিন্ন শিট কম্বিনেশন ট্রাই করবে (ম্যানুয়াল রিপোর্টের মতো আলাদা শিট)
    # আমরা প্লেট ১ এবং প্লেট ২ এর জন্য একটি রেঞ্জ স্ক্যান করবো
    total_target_all = sum(l["Target QTY"] for l in labels_input)
    avg_s = total_target_all / (grid_size * num_plates)
    
    # শিট সার্চ রেঞ্জ (আপনার ম্যানুয়াল রিপোর্টের শিট সংখ্যার আশেপাশে)
    s_range = range(max(1, int(avg_s * 0.5)), int(avg_s * 1.5) + 100, 5)

    # সিম্পল হিউরিস্টিক অপ্টিমাইজার
    for s1 in s_range:
        for s2 in [s1 if num_plates == 1 else s1 // 2, s1, s1 * 2]: # প্লেট ২ এর শিট ভিন্ন হতে পারে
            if num_plates == 1 and s2 != s1: continue
            
            p_sheets = [s1, s2] if num_plates >= 2 else [s1]
            if len(p_sheets) < num_plates: p_sheets.extend([s1] * (num_plates - len(p_sheets)))

            # প্রতিটি লেবেলের জন্য বেস্ট Ups বণ্টন (Cross-Plate)
            plate_ups = {f"Plate {p+1}": [] for p in range(num_plates)}
            total_excess_this_run = 0
            possible_run = True
            
            for l in labels_input:
                # টার্গেট মেলাতে সর্বনিম্ন কত ঘর লাগবে এই শিট সংখ্যায়
                # Equation: (U1 * S1) + (U2 * S2) >= Target
                found_ups = False
                for u1 in range(grid_size + 1):
                    for u2 in range(grid_size + 1) if num_plates >= 2 else [0]:
                        prod = (u1 * p_sheets[0]) + (u2 * (p_sheets[1] if num_plates >= 2 else 0))
                        if prod >= l["Target QTY"]:
                            l_excess = prod - l["Target QTY"]
                            # পেনাল্টি: অতিরিক্ত প্রোডাকশন কমানো
                            total_excess_this_run += l_excess
                            plate_ups["Plate 1"].append(u1)
                            if num_plates >= 2: plate_ups["Plate 2"].append(u2)
                            found_ups = True
                            break
                    if found_ups: break
                if not found_ups: possible_run = False; break
            
            # গ্রিড সাইজ চেক
            if possible_run:
                valid_grid = True
                for p in range(num_plates):
                    if sum(plate_ups[f"Plate {p+1}"]) > grid_size:
                        valid_grid = False; break
                
                if valid_grid and total_excess_this_run < best_overall_excess:
                    best_overall_excess = total_excess_this_run
                    best_config = {
                        "sheets": p_sheets,
                        "ups": plate_ups,
                        "excess": total_excess_this_run
                    }
                    if best_overall_excess < 50: break # খুব ভালো সলিউশন পেলে থেমে যাও

    # --- ফলাফল প্রদর্শন ---
    if best_config:
        st.divider()
        st.subheader("📊 AI অপ্টিমাইজড ফাইনাল রিপোর্ট")
        
        final_rows = []
        for i, l in enumerate(labels_input):
            row = {"Name": l["Name"], "Original QTY": l["Original QTY"], "Target QTY": l["Target QTY"]}
            total_produced = 0
            for p in range(num_plates):
                u = best_config["ups"][f"Plate {p+1}"][i]
                row[f"Plate {p+1} (Ups)"] = u
                total_produced += (u * best_config["sheets"][p])
            
            row["Total Produced"] = total_produced
            row["Excess"] = total_produced - l["Target QTY"]
            row["Over Print (%)"] = round((row["Excess"] / l["Original QTY"] * 100), 2) if l["Original QTY"] > 0 else 0
            final_rows.append(row)

        df_final = pd.DataFrame(final_rows)
        
        # TOTAL Row
        total_row = {"Name": "TOTAL", "Original QTY": df_final["Original QTY"].sum(), "Target QTY": df_final["Target QTY"].sum()}
        for p in range(num_plates):
            total_row[f"Plate {p+1} (Ups)"] = df_final[f"Plate {p+1} (Ups)"].sum()
        total_row["Total Produced"] = df_final["Total Produced"].sum()
        total_row["Excess"] = df_final["Excess"].sum()
        total_row["Over Print (%)"] = round(df_final["Over Print (%)"].mean(), 2)
        
        df_with_total = pd.concat([df_final, pd.DataFrame([total_row])], ignore_index=True)
        st.table(df_with_total)

        # প্রিন্টিং ইনস্ট্রাকশন
        st.write("### 📝 প্রিন্টিং ইনস্ট্রাকশন:")
        inst_cols = st.columns(num_plates)
        for p in range(num_plates):
            with inst_cols[p]:
                st.info(f"**Plate {p+1}:** {best_config['sheets'][p]} শিট প্রিন্ট করতে হবে।")
        
        st.success(f"✅ AI সলিউশন পাওয়া গেছে! মোট অপচয় মাত্র {int(best_config['excess'])} পিস।")
    else:
        st.error("❌ এই কনফিগারেশনে নিখুঁত কোনো রেশিও পাওয়া যায়নি। অনুগ্রহ করে প্লেট সংখ্যা বাড়িয়ে আবার চেষ্টা করুন।")

st.markdown("---")
st.caption("AI Pre-Press Optimization Engine | 2026 Update")
