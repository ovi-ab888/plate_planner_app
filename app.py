import streamlit as st
import math
import pandas as pd

st.set_page_config(page_title="Pre-Press Grid Optimizer", layout="wide")

st.title("🎨 Pre-Press Grid & Plate Optimizer")
st.write("লেবেল কোয়ান্টিটি এবং গ্রিড সাইজ অনুযায়ী প্লেট সেটআপ বের করার টুল।")

# --- Sidebar Inputs ---
st.sidebar.header("কনফিগারেশন")
grid_size = st.sidebar.number_input("একটি গ্রিডে (প্লেটে) মোট কয়টি লেবেল ধরে?", min_value=1, value=30)
extra_percent = st.sidebar.number_input("অ্যাড-অন (Extra %) কত হবে?", min_value=0.0, value=2.0)
num_labels = st.sidebar.number_input("মোট কত পদের (Types) লেবেল?", min_value=1, value=1, step=1)

# --- Data Input Section ---
st.subheader("📦 লেবেল এবং কোয়ান্টিটি ইনপুট দিন")
label_data = []

col1, col2 = st.columns(2)
for i in range(int(num_labels)):
    with col1:
        name = st.text_input(f"লেবেল {i+1} এর নাম", value=f"Label {i+1}", key=f"name_{i}")
    with col2:
        qty = st.number_input(f"{name} এর QTY (পিস)", min_value=1, value=100, key=f"qty_{i}")
    
    # Target calculation with extra percentage
    target_qty = math.ceil(qty * (1 + extra_percent / 100))
    label_data.append({"Name": name, "Original QTY": qty, "Target QTY": target_qty})

# --- Calculation Logic ---
if st.button("ক্যালকুলেট করুন"):
    df = pd.DataFrame(label_data)
    total_target_qty = df["Target QTY"].sum()
    
    # প্রাথমিক হিসাব: কয়টি প্লেট লাগতে পারে
    # আমরা ভারসাম্য বজায় রাখার জন্য একটি 'Base Print Run' ধরে আগাবো
    avg_qty = df["Target QTY"].mean()
    estimated_sheets = math.ceil(avg_qty / (grid_size / num_labels)) if num_labels > 0 else 0
    
    st.divider()
    st.subheader("📊 ক্যালকুলেশন রেজাল্ট")

    # প্লেট ডিস্ট্রিবিউশন (Simplified proportional logic)
    df["Ups (Calculated)"] = (df["Target QTY"] / total_target_qty) * grid_size
    df["Ups (Actual)"] = df["Ups (Calculated)"].apply(lambda x: round(x))
    
    # নিশ্চিত করা যে মোট Ups যেন গ্রিড সাইজের বেশি না হয়
    current_total_ups = df["Ups (Actual)"].sum()
    if current_total_ups > grid_size:
        # যদি বেশি হয়, সবচেয়ে বড় Ups থেকে কমিয়ে দেয়া
        df.loc[df["Ups (Actual)"].idxmax(), "Ups (Actual)"] -= (current_total_ups - grid_size)
    elif current_total_ups < grid_size:
        # যদি কম হয়, খালি ঘরগুলো সবচেয়ে বড় কোয়ান্টিটিতে যোগ করা
        df.loc[df["Ups (Actual)"].idxmax(), "Ups (Actual)"] += (grid_size - current_total_ups)

    # শিট সংখ্যা নির্ধারণ (সবচেয়ে বেশি যেটা লাগবে)
    sheets_needed = 0
    for index, row in df.iterrows():
        if row["Ups (Actual)"] > 0:
            needed = math.ceil(row["Target QTY"] / row["Ups (Actual)"])
            if needed > sheets_needed:
                sheets_needed = needed

    df["Total Produced"] = df["Ups (Actual)"] * sheets_needed
    df["Over Print (%)"] = ((df["Total Produced"] - df["Original QTY"]) / df["Original QTY"] * 100).round(2)

    # --- Display Outputs ---
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("মোট প্লেট লাগবে", "1 টি" if sheets_needed < 5000 else "একাধিক (ভলিউম অনুযায়ী)")
    res_col2.metric("মোট প্রিন্ট শিট", f"{sheets_needed} টি")
    res_col3.metric("ব্যবহৃত গ্রিড", f"{df['Ups (Actual)'].sum()} / {grid_size}")

    st.write("### প্লেট সেটআপ ডিটেইলস")
    st.table(df[["Name", "Original QTY", "Target QTY", "Ups (Actual)", "Total Produced", "Over Print (%)"]])

    st.info(f"💡 পরামর্শ: এই সেটআপে আপনি যদি **{sheets_needed}** টি শিট প্রিন্ট করেন, তবে আপনার সব লেবেল টার্গেট কোয়ান্টিটি অনুযায়ী পূরণ হবে।")

# --- Footer ---
st.markdown("---")
st.caption("Developed for Pre-press Optimization | User: Towhidul Islam Tushar")
