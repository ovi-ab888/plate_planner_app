import streamlit as st
import math
import pandas as pd

st.set_page_config(page_title="Pre-Press Grid Optimizer", layout="wide")

st.title("🎨 Pre-Press Grid & Plate Optimizer")

# --- Sidebar Inputs ---
st.sidebar.header("কনফিগারেশন")
grid_size = st.sidebar.number_input("একটি গ্রিডে (প্লেটে) মোট কয়টি লেবেল ধরে?", min_value=1, value=30)
extra_percent = st.sidebar.number_input("অ্যাড-অন (Extra %) কত হবে?", min_value=0.0, value=2.0)
num_labels = st.sidebar.number_input("মোট কত পদের (Types) লেবেল?", min_value=1, value=1, step=1)

# --- New: Plate Control Feature ---
st.sidebar.subheader("প্লেট সেটিংস")
is_auto = st.sidebar.checkbox("অটো প্লেট ক্যালকুলেশন", value=True)
if is_auto:
    num_plates = 1 # Initial base
else:
    num_plates = st.sidebar.number_input("প্লেট সংখ্যা সেট করুন", min_value=1, value=1, step=1)

# --- Data Input Section ---
st.subheader("📦 লেবেল এবং কোয়ান্টিটি ইনপুট দিন")
label_data = []

# Multiple columns for better UI
cols = st.columns(2)
for i in range(int(num_labels)):
    col_idx = i % 2
    with cols[col_idx]:
        c1, c2 = st.columns([2, 1])
        name = c1.text_input(f"লেবেল {i+1} নাম", value=f"Label {i+1}", key=f"name_{i}")
        qty = c2.number_input(f"QTY", min_value=1, value=100, key=f"qty_{i}")
    
    target_qty = math.ceil(qty * (1 + extra_percent / 100))
    label_data.append({"Name": name, "Original QTY": qty, "Target QTY": target_qty})

if st.button("ক্যালকুলেট করুন"):
    df = pd.DataFrame(label_data)
    total_target_qty = df["Target QTY"].sum()
    
    # --- Optimization Logic for Multiple Plates ---
    # Total 'Ups' available = grid_size * num_plates
    total_slots = grid_size * num_plates
    
    # Calculate proportional Ups
    df["Ups (Calculated)"] = (df["Target QTY"] / total_target_qty) * total_slots
    df["Ups (Actual)"] = df["Ups (Calculated)"].apply(lambda x: round(x))
    
    # Adjust to fit total slots
    current_total_ups = df["Ups (Actual)"].sum()
    if current_total_ups > total_slots:
        df.loc[df["Ups (Actual)"].idxmax(), "Ups (Actual)"] -= (current_total_ups - total_slots)
    elif current_total_ups < total_slots:
        df.loc[df["Ups (Actual)"].idxmax(), "Ups (Actual)"] += (total_slots - current_total_ups)
    
    # Sheets Calculation: Based on the highest single requirement per plate
    sheets_needed = math.ceil(total_target_qty / total_slots) if total_slots > 0 else 0
    
    # Refine sheets needed based on actual Ups distribution
    max_sheets = 0
    for idx, row in df.iterrows():
        if row["Ups (Actual)"] > 0:
            req = math.ceil(row["Target QTY"] / row["Ups (Actual)"])
            if req > max_sheets:
                max_sheets = req
    sheets_needed = max_sheets

    df["Total Produced"] = df["Ups (Actual)"] * sheets_needed
    df["Over Print (%)"] = ((df["Total Produced"] - df["Original QTY"]) / df["Original QTY"] * 100).round(2)

    # --- Calculations for Total Row ---
    total_orig = df["Original QTY"].sum()
    total_target = df["Target QTY"].sum()
    total_ups = df["Ups (Actual)"].sum()
    total_prod = df["Total Produced"].sum()
    avg_overprint = ((total_prod - total_orig) / total_orig * 100)

    st.divider()
    st.subheader("📊 ক্যালকুলেশন রেজাল্ট")

    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("মোট প্লেট", f"{num_plates} টি")
    res_col2.metric("প্রতি প্লেটে প্রিন্ট শিট", f"{sheets_needed} টি")
    res_col3.metric("ব্যবহৃত ঘর (Total Ups)", f"{total_ups} / {total_slots}")

    st.write("### প্লেট সেটআপ ডিটেইলস")
    
    # Displaying the table
    st.table(df[["Name", "Original QTY", "Target QTY", "Ups (Actual)", "Total Produced", "Over Print (%)"]])

    # --- New: Total Row UI ---
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
        <table style="width:100%; font-weight: bold; text-align: left;">
            <tr>
                <td>Total Labels: {len(df)}</td>
                <td>Total Original QTY: {total_orig}</td>
                <td>Total Target QTY: {total_target}</td>
                <td>Total Produced: {total_prod}</td>
                <td>Average Over-print: {avg_overprint:.2f}%</td>
            </tr>
        </table>
    </div>
    <br>
    """, unsafe_allow_html=True)

    st.info(f"💡 পরামর্শ: এই সেটআপে আপনি যদি প্রতি প্লেটে **{sheets_needed}** টি করে শিট প্রিন্ট করেন, তবে আপনার লক্ষ্যমাত্রা পূরণ হবে।")
