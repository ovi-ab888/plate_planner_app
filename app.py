import streamlit as st
import math
import pandas as pd

st.set_page_config(page_title="AI Pre-Press Optimizer Pro", layout="wide")

# ডার্ক মোড এবং সুন্দর UI ডিজাইন
st.markdown("""
    <style>
    .stApp { background-color: #000000; }
    .stTable { background-color: #1e1e1e !important; border-radius: 10px; color: white !important; }
    table { color: white !important; }
    label, p, h3 { color: white !important; }
    .stInfo { background-color: #0e2f44; color: #ffffff; border: 1px solid #17a2b8; }
    .stSuccess { background-color: #0b2e13; color: #ffffff; }
    .stWarning { background-color: #332b00; color: #ffffff; border: 1px solid #ffc107; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 AI Pre-Press Optimizer (V3 - Best Fit)")
st.write("এই ভার্সনটি Force Distribution লজিক ব্যবহার করে প্লেট ১ এবং প্লেট ২-এর মধ্যে ঘর (Ups) বণ্টন নিশ্চিত করবে।")

# --- Sidebar Configuration ---
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
        qty = c2.number_input(f"QTY", min_value=1, value=100, key=f"q_{i}")
        target = math.ceil(qty * (1 + extra_percent / 100))
        labels_input.append({"Name": name, "Original QTY": qty, "Target QTY": target})

if st.button("🚀 স্মার্ট অপ্টিমাইজেশন রান করুন"):
    best_overall_excess = float('inf')
    best_config = None
    
    # AI লজিক: গড় শিট সংখ্যা থেকে একটি স্মার্ট রেঞ্জ স্ক্যান করা
    total_target_all = sum(l["Target QTY"] for l in labels_input)
    avg_s = total_target_all / (grid_size * num_plates)
    s_range = range(max(10, int(avg_s * 0.4)), int(avg_s * 1.6) + 150, 2)

    for s1 in s_range:
        # প্লেট ২-এর শিট সংখ্যা প্লেট ১-এর সমান বা তার কম হওয়ার সম্ভাবনা চেক করা (ম্যানুয়াল রিপোর্টের মতো)
        potential_s2 = [s1, s1 // 2, s1 // 4, 38, 50] # সাধারণ কিছু প্রিন্টিং শিট প্যাটার্ন
        for s2 in potential_s2:
            if num_plates == 1: s2 = 0
            p_sheets = [s1, s2] if num_plates >= 2 else [s1]

            current_plate_ups = {"Plate 1": [], "Plate 2": []}
            total_excess_this_run = 0
            possible_run = True
            
            for l in labels_input:
                best_l_excess = float('inf')
                best_u_pair = (0, 0)
                
                # ঘর বণ্টন লজিক: (u1*s1) + (u2*s2) যেন টার্গেটের সবচেয়ে কাছে থাকে
                # এখানে Force করা হচ্ছে যেন u1 এবং u2 মিলিয়ে অন্তত ঘর থাকে
                for u1 in range(grid_size + 1):
                    limit_u2 = (grid_size - u1) if num_plates >= 2 else 0
                    for u2 in range(limit_u2 + 1):
                        prod = (u1 * s1) + (u2 * s2)
                        if prod >= l["Target QTY"]:
                            excess = prod - l["Target QTY"]
                            if excess < best_l_excess:
                                best_l_excess = excess
                                best_u_pair = (u1, u2)
                            break # টার্গেট পূরণ হলে এই জোড়া চেক করা বন্ধ
                
                if best_l_excess == float('inf'):
                    possible_run = False; break
                else:
                    total_excess_this_run += best_l_excess
                    current_plate_ups["Plate 1"].append(best_u_pair[0])
                    if num_plates >= 2: current_plate_ups["Plate 2"].append(best_u_pair[1])

            # গ্রিড সাইজ চেক এবং বেস্ট সলিউশন সেভ করা
            if possible_run:
                valid_grid = True
                for p in range(num_plates):
                    if sum(current_plate_ups[f"Plate {p+1}"]) > grid_size:
                        valid_grid = False; break
                
                # অপচয় যদি আগের চেয়ে কম হয়, তবে এটিই নতুন সেরা কনফিগারেশন
                if valid_grid and total_excess_this_run < best_overall_excess:
                    best_overall_excess = total_excess_this_run
                    best_config = {
                        "sheets": p_sheets,
                        "ups": current_plate_ups,
                        "excess": total_excess_this_run
                    }
                    if best_overall_excess < 20: break # খুব ভালো রেজাল্ট

    # --- ফলাফল প্রদর্শন ---
    if best_config:
        st.divider()
        st.subheader("📊 AI অপ্টিমাইজড ফাইনাল রিপোর্ট (Cross-Plate)")
        
        final_rows = []
        for i, l in enumerate(labels_input):
            row = {"Name": l["Name"], "Original QTY": l["Original QTY"], "Target QTY": l["Target QTY"]}
            total_produced = 0
            for p in range(int(num_plates)):
                u = best_config["ups"][f"Plate {p+1}"][i]
                row[f"Plate {p+1} (Ups)"] = u
                total_produced += (u * best_config["sheets"][p])
            
            row["Total Produced"] = total_produced
            row["Excess"] = total_produced - l["Target QTY"]
            row["Over Print (%)"] = round((row["Excess"] / l["Original QTY"] * 100), 4) if l["Original QTY"] > 0 else 0
            final_rows.append(row)

        df_final = pd.DataFrame(final_rows)
        
        # TOTAL Row
        total_row = {"Name": "TOTAL", "Original QTY": df_final["Original QTY"].sum(), "Target QTY": df_final["Target QTY"].sum()}
        for p in range(int(num_plates)):
            total_row[f"Plate {p+1} (Ups)"] = df_final[f"Plate {p+1} (Ups)"].sum()
        total_row["Total Produced"] = df_final["Total Produced"].sum()
        total_row["Excess"] = df_final["Excess"].sum()
        total_row["Over Print (%)"] = round((total_row["Excess"] / total_row["Original QTY"] * 100), 4)
        
        df_with_total = pd.concat([df_final, pd.DataFrame([total_row])], ignore_index=True)
        st.table(df_with_total)

        # প্রিন্টিং ইনস্ট্রাকশন
        st.write("### 📝 প্রিন্টিং ইনস্ট্রাকশন:")
        inst_cols = st.columns(int(num_plates))
        for p in range(int(num_plates)):
            with inst_cols[p]:
                st.info(f"**Plate {p+1}:** {best_config['sheets'][p]} শিট প্রিন্ট করতে হবে।")
        
        st.success(f"✅ AI সলিউশন পাওয়া গেছে! মোট অপচয় মাত্র {int(best_config['excess'])} পিস।")
    else:
        st.error("❌ কোনো সমাধান পাওয়া যায়নি। অনুগ্রহ করে গ্রিড সাইজ বা প্লেট সংখ্যা বাড়িয়ে দেখুন।")

# Footer
st.markdown("---")
st.caption("Developed for Towhidul Islam Tushar | Advanced Pre-Press AI 2026")
