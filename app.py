import streamlit as st
import math
import pandas as pd

st.set_page_config(page_title="AI Pre-Press Optimizer Pro", layout="wide")

# CSS for UI
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

st.title("🧠 AI Pre-Press Optimizer (Full Grid Utilization)")

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

if st.button("🚀 অপ্টিমাইজেশন রান করুন"):
    best_overall_excess = float('inf')
    best_config = None
    
    total_target_all = sum(l["Target QTY"] for l in labels_input)
    avg_s = total_target_all / (grid_size * num_plates)
    s_range = range(max(10, int(avg_s * 0.4)), int(avg_s * 1.6) + 150, 2)

    for s1 in s_range:
        # সম্ভাব্য শিট সংখ্যার কম্বিনেশন (ম্যানুয়াল রিপোর্ট লজিক)
        potential_s2 = [s1, s1 // 2, 38, 50, 100] if num_plates >= 2 else [0]
        for s2 in potential_s2:
            current_plate_ups = {"Plate 1": [], "Plate 2": []}
            possible_run = True
            
            for l in labels_input:
                best_l_excess = float('inf')
                best_u_pair = (0, 0)
                # টার্গেট পূরণের জন্য প্রয়োজনীয় সর্বনিম্ন Ups বের করা
                for u1 in range(grid_size + 1):
                    for u2 in range(grid_size + 1):
                        prod = (u1 * s1) + (u2 * s2)
                        if prod >= l["Target QTY"]:
                            excess = prod - l["Target QTY"]
                            if excess < best_l_excess:
                                best_l_excess = excess
                                best_u_pair = (u1, u2)
                            break
                    if best_l_excess != float('inf'): break
                
                current_plate_ups["Plate 1"].append(best_u_pair[0])
                current_plate_ups["Plate 2"].append(best_u_pair[1])

            # --- গ্রিড পূর্ণ করার লজিক (Fill to 30) ---
            for p in range(int(num_plates)):
                p_name = f"Plate {p+1}"
                used_ups = sum(current_plate_ups[p_name])
                if used_ups < grid_size:
                    # সবচেয়ে বড় অর্ডারটিকে খালি ঘরগুলো দিয়ে দাও
                    diff = grid_size - used_ups
                    max_idx = 0
                    for i, l in enumerate(labels_input):
                        if l["Target QTY"] > labels_input[max_idx]["Target QTY"]:
                            max_idx = i
                    current_plate_ups[p_name][max_idx] += diff

            # ফাইনাল অপচয় চেক
            total_produced_all = 0
            for i in range(len(labels_input)):
                total_produced_all += (current_plate_ups["Plate 1"][i] * s1) + (current_plate_ups["Plate 2"][i] * s2)
            
            total_excess = total_produced_all - total_target_all
            
            # গ্রিড সাইজ লিমিট ক্রস না করলে এবং অপচয় কম হলে সেভ করো
            if all(sum(current_plate_ups[f"Plate {p+1}"]) <= grid_size for p in range(int(num_plates))):
                if total_excess < best_overall_excess:
                    best_overall_excess = total_excess
                    best_config = {"sheets": [s1, s2], "ups": current_plate_ups}

    # --- রিপোর্ট প্রদর্শন ---
    if best_config:
        st.divider()
        st.subheader(f"📊 AI ফাইনাল রিপোর্ট (Grid Size: {grid_size} Utilized)")
        
        final_rows = []
        for i, l in enumerate(labels_input):
            row = {"Name": l["Name"], "Original QTY": l["Original QTY"], "Target QTY": l["Target QTY"]}
            prod = 0
            for p in range(int(num_plates)):
                u = best_config["ups"][f"Plate {p+1}"][i]
                row[f"Plate {p+1} (Ups)"] = u
                prod += (u * best_config["sheets"][p])
            row["Total Produced"] = prod
            row["Excess"] = prod - l["Target QTY"]
            row["Over Print (%)"] = round((row["Excess"] / l["Original QTY"] * 100), 2)
            final_rows.append(row)

        df = pd.DataFrame(final_rows)
        # Total Row
        t_row = {"Name": "TOTAL", "Original QTY": df["Original QTY"].sum(), "Target QTY": df["Target QTY"].sum()}
        for p in range(int(num_plates)): t_row[f"Plate {p+1} (Ups)"] = df[f"Plate {p+1} (Ups)"].sum()
        t_row["Total Produced"] = df["Total Produced"].sum()
        t_row["Excess"] = df["Excess"].sum()
        t_row["Over Print (%)"] = round((t_row["Excess"] / t_row["Original QTY"] * 100), 2)
        
        st.table(pd.concat([df, pd.DataFrame([t_row])], ignore_index=True))
        
        st.write("### 📝 প্রিন্টিং ইনস্ট্রাকশন:")
        for p in range(int(num_plates)):
            st.info(f"**Plate {p+1}:** {best_config['sheets'][p]} শিট (Total Ups: {sum(best_config['ups'][f'Plate {p+1}'])}/{grid_size})")
    else:
        st.error("সঠিক কম্বিনেশন পাওয়া যায়নি।")
