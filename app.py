import streamlit as st
import pandas as pd

st.set_page_config(page_title="Manual Pre-Press Controller", layout="wide")

# ডার্ক থিম স্টাইল
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: white; }
    .stNumberInput input { background-color: #161b22 !important; color: white !important; }
    .stTable { background-color: #161b22 !important; }
    table { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛠️ Manual Pre-Press Controller")
st.write("এখানে আপনি নিজের ইচ্ছামতো ঘর (Ups) এবং শিট বসিয়ে অপচয় চেক করতে পারবেন।")

# --- Configuration ---
st.sidebar.header("⚙️ সেটিংস")
grid_size = st.sidebar.number_input("গ্রিড সাইজ", value=30)
num_labels = st.sidebar.number_input("মোট লেবেল পদ", value=8)

# --- Sheet Input ---
st.subheader("📄 শিট সংখ্যা ইনপুট (Plate-wise)")
col_s1, col_s2 = st.columns(2)
s1 = col_s1.number_input("Plate 1 Sheets", min_value=0, value=276)
s2 = col_s2.number_input("Plate 2 Sheets", min_value=0, value=38)

# --- Labels & Ups Input ---
st.subheader("📦 লেবেল এবং ঘর (Ups) সেট করুন")
labels_data = []

# টেবিল হেডার
header = st.columns([2, 2, 1, 1])
header[0].write("**লেবেল নাম**")
header[1].write("**টার্গেট QTY**")
header[2].write("**Plate 1 Ups**")
header[3].write("**Plate 2 Ups**")

for i in range(int(num_labels)):
    c = st.columns([2, 2, 1, 1])
    name = c[0].text_input(f"Name {i+1}", value=f"Label {i+1}", key=f"n_{i}", label_visibility="collapsed")
    target = c[1].number_input(f"Target {i+1}", value=1000, key=f"t_{i}", label_visibility="collapsed")
    u1 = c[2].number_input(f"P1 U {i+1}", min_value=0, value=0, key=f"u1_{i}", label_visibility="collapsed")
    u2 = c[3].number_input(f"P2 U {i+1}", min_value=0, value=0, key=f"u2_{i}", label_visibility="collapsed")
    labels_data.append({"Name": name, "Target": target, "U1": u1, "U2": u2})

# --- Calculation & Report ---
if st.button("📊 ক্যালকুলেট রিপোর্ট"):
    final_rows = []
    total_u1 = 0
    total_u2 = 0
    
    for l in labels_data:
        prod = (l["U1"] * s1) + (l["U2"] * s2)
        excess = prod - l["Target"]
        over_print = (excess / l["Target"] * 100) if l["Target"] > 0 else 0
        
        final_rows.append({
            "Name": l["Name"],
            "Target QTY": l["Target"],
            "Plate 1 (Ups)": l["U1"],
            "Plate 2 (Ups)": l["U2"],
            "Total Produced": prod,
            "Excess": excess,
            "Over Print (%)": round(over_print, 2)
        })
        total_u1 += l["U1"]
        total_u2 += l["U2"]

    df = pd.DataFrame(final_rows)
    
    # গ্র্যান্ড টোটাল
    total_target = df["Target QTY"].sum()
    total_prod = df["Total Produced"].sum()
    total_excess = total_prod - total_target
    
    st.divider()
    st.subheader("📋 ফাইনাল রিপোর্ট")
    st.table(df)
    
    # সামারি কার্ডস
    c1, c2, c3 = st.columns(3)
    c1.metric("Plate 1 Total Ups", f"{total_u1} / {grid_size}", delta=total_u1-grid_size, delta_color="inverse")
    c2.metric("Plate 2 Total Ups", f"{total_u2} / {grid_size}", delta=total_u2-grid_size, delta_color="inverse")
    c3.metric("Total Excess", f"{total_excess} Pcs", f"{round((total_excess/total_target*100),2) if total_target > 0 else 0}%")

    if total_u1 > grid_size or total_u2 > grid_size:
        st.error(f"⚠️ সাবধান! আপনার ঘর সংখ্যা গ্রিড সাইজ ({grid_size}) অতিক্রম করেছে।")
    elif total_u1 == grid_size and total_u2 == grid_size:
        st.success("✅ পারফেক্ট! গ্রিড পুরোপুরি পূর্ণ হয়েছে।")
