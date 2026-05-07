import streamlit as st
import math
import pandas as pd

st.set_page_config(page_title="Pre-Press Optimizer Pro", layout="wide")

# CSS for Dark Theme
st.markdown("""
    <style>
    .stApp { background-color: #000000; }
    .stTable { background-color: #1e1e1e !important; border-radius: 10px; color: white !important; }
    table { color: white !important; }
    label, p { color: white !important; }
    .stInfo { background-color: #0e2f44; color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎨 Pre-Press Grid & Plate Optimizer (V2)")
st.write("Cross-Plate Distribution লজিক ব্যবহার করে অপচয় কমানোর টুল।")

# --- Sidebar ---
st.sidebar.header("⚙️ কনফিগারেশন")
grid_size = st.sidebar.number_input("গ্রিড সাইজ (Grid Size)", min_value=1, value=30)
num_labels = st.sidebar.number_input("মোট লেবেল পদ", min_value=1, value=8, step=1)
num_plates = st.sidebar.number_input("কয়টি প্লেট করবেন?", min_value=1, value=2, step=1)

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
        labels_input.append({"Name": name, "Target QTY": qty})

if st.button("🚀 অপ্টিমাইজড ক্যালকুলেশন করুন"):
    # --- Advanced Optimization Logic ---
    # ১. মোট কতটি 'ঘর' (Total Ups across all plates) পাওয়া যাবে
    total_available_ups = grid_size * num_plates
    total_target_qty = sum(l["Target QTY"] for l in labels_input)
    
    # ২. প্রতিটি লেবেলের জন্য আদর্শ ঘর (Global Ideal Ups) বের করা
    # (Label QTY / Total QTY) * Total Available Ups
    global_ups_list = []
    for l in labels_input:
        ideal_ups = (l["Target QTY"] / total_target_qty) * total_available_ups
        global_ups_list.append(ideal_ups)

    # ৩. ঘরগুলোকে প্লেট অনুযায়ী ভাগ করা (Cross-Plate Distribution)
    # প্লেট ১ এ কিছু ঘর যাবে, বাকি ঘর প্লেট ২ এ যাবে
    plate_ups_matrix = {f"Plate {p+1}": [] for p in range(int(num_plates))}
    
    for i, total_ups in enumerate(global_ups_list):
        remaining_ups = total_ups
        for p in range(int(num_plates)):
            if p == int(num_plates) - 1: # শেষ প্লেট
                plate_ups_matrix[f"Plate {p+1}"].append(round(remaining_ups))
            else:
                # প্রতিটি প্লেটে গ্রিড সাইজের আনুপাতিক অংশ দেওয়া
                p_ups = round(total_ups / num_plates)
                plate_ups_matrix[f"Plate {p+1}"].append(p_ups)
                remaining_ups -= p_ups

    # ৪. গ্রিড সাইজ অ্যাডজাস্টমেন্ট (৩০ এর বেশি যেন না হয়)
    for p in range(int(num_plates)):
        current_sum = sum(plate_ups_matrix[f"Plate {p+1}"])
        diff = grid_size - current_sum
        if diff != 0:
            # সবচেয়ে বড় লেবেলে ডিফারেন্স যোগ করা
            max_idx = global_ups_list.index(max(global_ups_list))
            plate_ups_matrix[f"Plate {p+1}"][max_idx] += diff

    # ৫. অপ্টিমাইজড শিট সংখ্যা বের করা
    # আমরা এমন একটি শিট সংখ্যা খুঁজছি যা সব প্লেটের জন্য সমান (অথবা কাছাকাছি)
    # আপনার ম্যানুয়াল রিপোর্ট অনুযায়ী প্লেট ১ ও ২ এর শিট আলাদা হতে পারে।
    # এখানে আমরা প্রতিটি প্লেটের সর্বোচ্চ ডিমান্ড অনুযায়ী শিট ধরছি।
    plate_sheets = []
    for p in range(int(num_plates)):
        sheets_needed = []
        for i, u in enumerate(plate_ups_matrix[f"Plate {p+1}"]):
            if u > 0:
                # যেহেতু ঘর ভাগ হয়েছে, তাই টার্গেটের একটি অংশ এই প্লেটে আসবে
                target_share = (plate_ups_matrix[f"Plate {p+1}"][i] / sum(row[i] for row in plate_ups_matrix.values())) * labels_input[i]["Target QTY"]
                sheets_needed.append(math.ceil(target_share / u))
        plate_sheets.append(max(sheets_needed) if sheets_needed else 0)

    # --- রিপোর্ট জেনারেশন ---
    final_data = []
    for i in range(len(labels_input)):
        row = {
            "Name": labels_input[i]["Name"],
            "Target QTY": labels_input[i]["Target QTY"]
        }
        total_produced = 0
        for p in range(int(num_plates)):
            ups = plate_ups_matrix[f"Plate {p+1}"][i]
            row[f"Plate {p+1} (Ups)"] = ups
            total_produced += (ups * plate_sheets[p])
        
        row["Total Produced"] = total_produced
        row["Excess"] = total_produced - labels_input[i]["Target QTY"]
        row["Over Print (%)"] = round((row["Excess"] / labels_input[i]["Target QTY"] * 100), 2)
        final_data.append(row)

    df_final = pd.DataFrame(final_data)
    
    # Total Row
    total_row = {"Name": "TOTAL", "Target QTY": df_final["Target QTY"].sum()}
    for p in range(int(num_plates)):
        total_row[f"Plate {p+1} (Ups)"] = df_final[f"Plate {p+1} (Ups)"].sum()
    total_row["Total Produced"] = df_final["Total Produced"].sum()
    total_row["Excess"] = df_final["Excess"].sum()
    total_row["Over Print (%)"] = round(df_final["Over Print (%)"].mean(), 2)

    df_with_total = pd.concat([df_final, pd.DataFrame([total_row])], ignore_index=True)
    
    st.divider()
    st.subheader("📊 ফাইনাল রিপোর্ট (Cross-Plate Optimized)")
    st.table(df_with_total)

    # Printing Instructions
    st.write("### 📝 প্রিন্টিং ইনস্ট্রাকশন:")
    cols_info = st.columns(int(num_plates))
    for p in range(int(num_plates)):
        with cols_info[p]:
            st.info(f"**Plate {p+1}:** {plate_sheets[p]} শিট প্রিন্ট করতে হবে।")

    st.success("✅ এই ক্যালকুলেশনটি ম্যানুয়াল রিপোর্টের মতো প্রতিটি লেবেলকে সব প্লেটে বণ্টন করেছে, যা অপচয় কমিয়ে আনে।")
