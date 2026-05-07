import streamlit as st
import math
import pandas as pd

st.set_page_config(page_title="AI Pre-Press Optimizer Zero-Waste", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #000000; }
    .stTable { background-color: #1e1e1e !important; border-radius: 10px; color: white !important; }
    table { color: white !important; }
    label, p, h3 { color: white !important; }
    .stInfo { background-color: #0e2f44; color: #ffffff; border: 1px solid #17a2b8; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 AI Pre-Press Optimizer (Zero-Waste Edition)")

# --- Sidebar ---
st.sidebar.header("⚙️ কনফিগারেশন")
grid_size = st.sidebar.number_input("গ্রিড সাইজ", min_value=1, value=30)
num_labels = st.sidebar.number_input("মোট লেবেল পদ", min_value=1, value=8)
num_plates = st.sidebar.number_input("প্লেট সংখ্যা", min_value=1, value=2)

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
        labels_input.append({"Name": name, "Original QTY": qty, "Target QTY": qty})

if st.button("🚀 ক্যালকুলেট অপ্টিমাইজড রেশিও"):
    best_config = None
    min_total_excess = float('inf')
    
    # AI Search: শিট সংখ্যার বিভিন্ন কম্বিনেশন ট্রাই করা (আপনার ম্যানুয়াল রিপোর্টের লজিক)
    # প্লেট ১ সাধারণত বড় অর্ডারের জন্য (বেশি শিট)
    # প্লেট ২ সাধারণত ছোট অর্ডারের জন্য (কম শিট)
    
    total_qty = sum(l['Target QTY'] for l in labels_input)
    s1_range = range(10, 500, 5) # বড় শিট রেঞ্জ
    s2_range = [0, 10, 20, 38, 50, 75, 100] # ছোট শিট রেঞ্জ
    
    for s1 in s1_range:
        for s2 in s2_range:
            if num_plates == 1: s2 = 0
            
            current_ups = {"Plate 1": [0]*len(labels_input), "Plate 2": [0]*len(labels_input)}
            possible = True
            
            for i, l in enumerate(labels_input):
                best_l_excess = float('inf')
                best_u = (0, 0)
                
                # এমন (u1, u2) জোড়া খোঁজা যা অপচয় সর্বনিম্ন রাখে
                for u1 in range(grid_size + 1):
                    for u2 in range(grid_size + 1):
                        prod = (u1 * s1) + (u2 * s2)
                        if prod >= l["Target QTY"]:
                            excess = prod - l["Target QTY"]
                            if excess < best_l_excess:
                                best_l_excess = excess
                                best_u = (u1, u2)
                            break
                    if best_l_excess == 0: break # ০ অপচয় পেলে আর খোঁজার দরকার নেই
                
                current_ups["Plate 1"][i] = best_u[0]
                current_ups["Plate 2"][i] = best_u[1]
            
            # গ্রিড সাইজ চেক
            if sum(current_ups["Plate 1"]) <= grid_size and sum(current_ups["Plate 2"]) <= grid_size:
                # টোটাল অপচয় ক্যালকুলেট
                current_excess = 0
                for i, l in enumerate(labels_input):
                    prod = (current_ups["Plate 1"][i] * s1) + (current_ups["Plate 2"][i] * s2)
                    current_excess += (prod - l["Target QTY"])
                
                if current_excess < min_total_excess:
                    min_total_excess = current_excess
                    best_config = {"s1": s1, "s2": s2, "ups": current_ups}
                    if min_total_excess == 0: break
        if min_total_excess == 0: break

    if best_config:
        # --- Grid Filling: বাকি খালি ঘরগুলো বড় অর্ডারকে দিয়ে গ্রিড ৩০ করা ---
        for p in ["Plate 1", "Plate 2"]:
            rem_ups = grid_size - sum(best_config["ups"][p])
            if rem_ups > 0:
                # সবচেয়ে বড় QTY এর লেবেলকে খালি ঘরগুলো দাও
                max_idx = 0
                for i in range(len(labels_input)):
                    if labels_input[i]["Target QTY"] > labels_input[max_idx]["Target QTY"]:
                        max_idx = i
                best_config["ups"][p][max_idx] += rem_ups

        # --- রিপোর্ট টেবিল ---
        final_data = []
        for i, l in enumerate(labels_input):
            p1_u = best_config["ups"]["Plate 1"][i]
            p2_u = best_config["ups"]["Plate 2"][i]
            prod = (p1_u * best_config["s1"]) + (p2_u * best_config["s2"])
            final_data.append({
                "Name": l["Name"],
                "Target QTY": l["Target QTY"],
                "Plate 1 (Ups)": p1_u,
                "Plate 2 (Ups)": p2_u,
                "Total Produced": prod,
                "Excess": prod - l["Target QTY"],
                "Over Print (%)": round(((prod - l["Target QTY"])/l["Target QTY"])*100, 2)
            })
            
        df = pd.DataFrame(final_data)
        st.table(df)
        st.info(f"প্রিন্টিং ইনস্ট্রাকশন: Plate 1: {best_config['s1']} শিট, Plate 2: {best_config['s2']} শিট।")
        st.success(f"অভিনন্দন! মোট অপচয় মাত্র {min_total_excess} পিস।")
