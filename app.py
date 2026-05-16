""")

return quantities if quantities else None

except Exception as e:
st.error(f"PDF read error: {str(e)}")
return None

# ================================================================
#  EXCEL/CSV TEMPLATE GENERATOR
# ================================================================

def generate_template():
"""Generate Excel template for bulk upload"""
template_data = pd.DataFrame({
"Tag Name": ["Example Product 1", "Example Product 2"],
"Quantity": [100, 250]
})

output = BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
template_data.to_excel(writer, sheet_name="Template", index=False)
output.seek(0)
return output

# ================================================================
#  PLATE NAME GENERATOR
# ================================================================

def plate_name(n):
n -= 1
chars = string.ascii_uppercase
out = ""
while True:
out = chars[n % 26] + out
n = n // 26 - 1
if n < 0:
break
return out

# ================================================================
#  SMART BALANCED UPS - RATIO BASED
# ================================================================

def smart_layout(demand, cap):
total = sum(demand.values())
if total == 0:
return {}
floor_vals = {}
remainders = {}
for k, v in demand.items():
ratio = (v / total) * cap
floor_vals[k] = floor(ratio)
remainders[k] = ratio - floor_vals[k]
layout = dict(floor_vals)
for k in layout:
if layout[k] == 0:
layout[k] = 1
while sum(layout.values()) > cap:
biggest = max(layout, key=layout.get)
if layout[biggest] > 1:
layout[biggest] -= 1
else:
break
remaining_cap = cap - sum(layout.values())
while remaining_cap > 0:
best = max(remainders, key=remainders.get)
layout[best] += 1
remainders[best] = 0
remaining_cap -= 1
return layout

# ================================================================
#  AUTO PLAN
# ================================================================

def auto_plan(demand, cap, max_plates):
remaining = demand.copy()
plates = []
produced = Counter()
for i in range(max_plates):
if not any(v > 0 for v in remaining.values()):
break
layout = smart_layout(remaining, cap)
if not layout:
break
possible = [
ceil(remaining[k] / v)
for k, v in layout.items()
if v > 0
]
sheets = max(1, min(possible))
for k, v in layout.items():
produced_qty = v * sheets
remaining[k] = max(0, remaining[k] - produced_qty)
produced[k] += produced_qty
plates.append({
"name": plate_name(len(plates) + 1),
"layout": layout,
"sheets": sheets
})
if any(v > 0 for v in remaining.values()) and plates:
last = plates[-1]
for k in remaining:
if remaining[k] > 0:
    per_sheet = max(1, last["layout"].get(k, 1))
    add_sheets = ceil(remaining[k] / per_sheet)
    last["sheets"] += add_sheets
    produced[k] += add_sheets * per_sheet
    remaining[k] = 0
return plates, dict(produced)

# ================================================================
#  MAIN UI
# ================================================================

# Custom CSS for main app
st.markdown("""
<style>
.stApp { background: black !important; }
.main-header {
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
padding: 2rem;
border-radius: 15px;
margin-bottom: 2rem;
text-align: center;
}
.main-header h1 { color: white; font-size: 2.5rem; margin-bottom: 0.5rem; font-weight: 600; }
.main-header p { color: rgba(255,255,255,0.9); font-size: 1.1rem; }
.card {
background: #1a1a1a;
border-radius: 12px;
padding: 1.5rem;
margin-bottom: 1.5rem;
border: 1px solid #333;
}
.card:hover { border-color: #667eea; }
.card-title {
font-size: 1.3rem;
font-weight: 600;
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
margin-bottom: 1rem;
padding-bottom: 0.5rem;
border-bottom: 2px solid #667eea;
display: inline-block;
}
.metric-card {
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
border-radius: 10px;
padding: 1rem;
color: white;
text-align: center;
transition: transform 0.3s ease;
}
.metric-card:hover { transform: translateY(-5px); }
.metric-value { font-size: 2rem; font-weight: bold; }
.metric-label { font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem; }
.stButton > button {
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
color: white;
border: none;
padding: 0.75rem 2rem;
font-size: 1rem;
font-weight: 600;
border-radius: 8px;
width: 100%;
}
.stButton > button:hover {
transform: translateY(-2px);
box-shadow: 0 5px 15px rgba(102,126,234,0.4);
}
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
border-radius: 8px;
border: 1px solid #333;
padding: 0.5rem;
background: #1a1a1a;
color: white;
}
.footer {
text-align: center;
padding: 2rem;
background: #1a1a1a;
border-radius: 15px;
margin-top: 2rem;
border: 1px solid #333;
}
.footer p { color: #ccc; font-size: 0.9rem; margin: 0.5rem 0; }
.designer-credit {
font-size: 1rem;
font-weight: 600;
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
}
.badge {
display: inline-block;
padding: 0.25rem 0.75rem;
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
border-radius: 20px;
font-size: 0.85rem;
color: white;
}
.upload-section {
background: #0a0a0a;
border-radius: 10px;
padding: 1rem;
margin-bottom: 1rem;
border: 1px dashed #667eea;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
<h1>📊 Plate Ratio System</h1>
<p>Professional UPS Ratio Optimization | Low Waste + Smart Distribution</p>
</div>
""", unsafe_allow_html=True)

# Configuration Panel
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">⚙️ Production Configuration</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
n = st.number_input("🏷️ Tag Count", 1, 50, 6)

with col2:
cap = st.number_input("📀 Plate Capacity", 1, 64, 12)

with col3:
maxp = st.number_input("🎨 Max Plates", 1, 50, 2)

with col4:
addon = st.number_input("📈 Add-on %", 0.0, 50.0, 0.0, step=0.5)

st.markdown('</div>', unsafe_allow_html=True)

# QTY Input Section with multiple upload options
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">📦 Tag Quantity Details</div>', unsafe_allow_html=True)

# Option selection
input_method = st.radio(
"Select Input Method:",
["✏️ Manual Input", "📄 Upload PDF File", "📊 Upload Excel/CSV"],
horizontal=True,
help="Choose how to enter quantities"
)

tags = []
qty = []
uploaded_data = None

if input_method == "📊 Upload Excel/CSV":
st.markdown('<div class="upload-section">', unsafe_allow_html=True)

# Template download
template_file = generate_template()
st.download_button(
"📥 Download Excel Template",
data=template_file,
file_name="quantity_template.xlsx",
mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

uploaded_file = st.file_uploader(
"Choose Excel/CSV File", 
type=['xlsx', 'xls', 'csv'],
help="Upload file with 'Tag Name' and 'Quantity' columns"
)

if uploaded_file is not None:
try:
if uploaded_file.name.endswith('.csv'):
    df_upload = pd.read_csv(uploaded_file)
else:
    df_upload = pd.read_excel(uploaded_file)

st.success(f"✅ File loaded successfully! Found {len(df_upload)} rows.")

# Display preview
st.dataframe(df_upload.head(10), use_container_width=True)

# Check required columns
if 'Tag Name' in df_upload.columns and 'Quantity' in df_upload.columns:
    uploaded_data = dict(zip(df_upload['Tag Name'], df_upload['Quantity']))
    st.success(f"✅ Loaded {len(uploaded_data)} items!")
elif df_upload.shape[1] >= 2:
    # Assume first column is Tag, second is Quantity
    uploaded_data = dict(zip(df_upload.iloc[:, 0], df_upload.iloc[:, 1]))
    st.info(f"Using first column as Tag Name and second as Quantity. Loaded {len(uploaded_data)} items.")
else:
    st.error("File must have at least 2 columns (Tag Name and Quantity)")
except Exception as e:
st.error(f"Error reading file: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

elif input_method == "📄 Upload PDF File":
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.info("📌 Upload PDF file containing product codes and quantities.")

uploaded_file = st.file_uploader(
"Choose PDF File", 
type=['pdf'],
help="Upload PDF with product names/codes and quantities"
)

if uploaded_file is not None:
with st.spinner("📖 Reading PDF and extracting quantities..."):
extracted_qty = extract_qty_from_pdf(uploaded_file)

if extracted_qty and len(extracted_qty) > 0:
    st.success(f"✅ Successfully extracted {len(extracted_qty)} items from PDF!")
    
    # Display extracted data
    st.markdown("**Extracted Data:**")
    extracted_df = pd.DataFrame([
        {"Product Code/Tag": k, "Quantity": v} 
        for k, v in extracted_qty.items()
    ])
    st.dataframe(extracted_df, use_container_width=True, hide_index=True)
    
    # Allow editing of extracted data
    st.markdown("**Edit extracted data (optional):**")
    edited_df = st.data_editor(extracted_df, use_container_width=True, num_rows="dynamic")
    
    if st.button("✅ Use This Data for Planning"):
        uploaded_data = dict(zip(edited_df['Product Code/Tag'], edited_df['Quantity']))
        st.success(f"Data loaded successfully! {len(uploaded_data)} items ready for planning.")
else:
    st.warning("Could not extract quantities from PDF. Please use Manual Input or Excel/CSV option.")
    
    # Show manual entry option
    st.markdown("### Or enter manually:")
    manual_tags = []
    manual_qty = []
    for i in range(min(5, n)):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input(f"Tag {i+1}", key=f"manual_tag_{i}")
        with col2:
            q = st.number_input(f"Qty", 0, key=f"manual_qty_{i}")
        manual_tags.append(name)
        manual_qty.append(q)
    
    if st.button("Use Manual Entry"):
        uploaded_data = {t: q for t, q in zip(manual_tags, manual_qty) if q > 0}

st.markdown('</div>', unsafe_allow_html=True)

else:  # Manual Input
tags = []
qty = []

for i in range(n):
col1, col2 = st.columns(2)
with col1:
name = st.text_input(f"Tag {i+1} Name", f"Tag {i+1}", key=f"tag_{i}")
with col2:
q = st.number_input(f"Quantity", 0, step=10, key=f"qty_{i}")
tags.append(name)
qty.append(q)

if any(q > 0 for q in qty):
uploaded_data = {t: int(q) for t, q in zip(tags, qty) if q > 0}

st.markdown('</div>', unsafe_allow_html=True)

# Data processing
if uploaded_data:
original_qty = uploaded_data
demand = {t: ceil(int(q) * (1 + addon / 100)) for t, q in uploaded_data.items()}
st.info(f"📊 Loaded {len(demand)} tags for planning")
else:
original_qty = {}
demand = {}

# Generate Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
generate_clicked = st.button("🚀 Generate Optimized Plan", use_container_width=True)

if generate_clicked:
if not demand:
st.error("⚠️ Please enter or upload at least one tag with quantity greater than 0")
st.stop()

with st.spinner("🔄 Optimizing production plan..."):
plates, produced = auto_plan(demand, cap, maxp)

rows = []
for tag in demand:
row = {"Tag": tag, "Original QTY": original_qty[tag], "Produced (+Add-on)": demand[tag]}
total_produced = 0
for p in plates:
ups = p["layout"].get(tag, 0)
row[f"Plate {p['name']}"] = ups
total_produced += (ups * p["sheets"])
excess = total_produced - demand[tag]
excess_percent = round((excess / demand[tag]) * 100, 2) if demand[tag] else 0
row["Total Produced QTY"] = total_produced
row["Excess"] = excess
row["Excess %"] = excess_percent
rows.append(row)

df = pd.DataFrame(rows)

total_row = {
"Tag": "📊 TOTAL",
"Original QTY": df["Original QTY"].sum(),
"Produced (+Add-on)": df["Produced (+Add-on)"].sum(),
}
for p in plates:
total_row[f"Plate {p['name']}"] = df[f"Plate {p['name']}"].sum()
total_row["Total Produced QTY"] = df["Total Produced QTY"].sum()
total_row["Excess"] = df["Excess"].sum()
total_row["Excess %"] = round((total_row["Excess"] / total_row["Produced (+Add-on)"]) * 100, 2) if total_row["Produced (+Add-on)"] > 0 else 0

df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

total_sheets = sum(p["sheets"] for p in plates)
total_excess = df.iloc[:-1]["Excess"].sum()
waste_percentage = (total_excess / df.iloc[:-1]["Produced (+Add-on)"].sum() * 100) if df.iloc[:-1]["Produced (+Add-on)"].sum() > 0 else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
st.markdown(f'<div class="metric-card"><div class="metric-value">{len(plates)}</div><div class="metric-label">Total Plates</div></div>', unsafe_allow_html=True)
with col2:
st.markdown(f'<div class="metric-card"><div class="metric-value">{total_sheets}</div><div class="metric-label">Total Sheets</div></div>', unsafe_allow_html=True)
with col3:
st.markdown(f'<div class="metric-card"><div class="metric-value">{total_excess:,}</div><div class="metric-label">Total Excess</div></div>', unsafe_allow_html=True)
with col4:
st.markdown(f'<div class="metric-card"><div class="metric-value">{waste_percentage:.1f}%</div><div class="metric-label">Waste Rate</div></div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("## 📊 Production Summary")
st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("## 🧾 Plate Configuration Details")
plate_rows = []
for p in plates:
plate_rows.append({
"Plate ID": p["name"],
"Sheets Required": p["sheets"],
"Total UPS": sum(p["layout"].values()),
"Layout": ", ".join([f"{k}:{v}" for k, v in p["layout"].items()])
})
plate_df = pd.DataFrame(plate_rows)
st.dataframe(plate_df, use_container_width=True, hide_index=True)

col1, col2 = st.columns(2)
with col1:
st.success(f"✅ Production plan optimized successfully! Total sheets: {total_sheets}")

bio = BytesIO()
with pd.ExcelWriter(bio, engine="openpyxl") as writer:
df.to_excel(writer, sheet_name="Production Summary", index=False)
plate_df.to_excel(writer, sheet_name="Plate Details", index=False)
bio.seek(0)

with col2:
st.download_button("⬇️ Download Excel Report", data=bio, file_name="plate_ratio_plan.xlsx", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
<p>📊 Plate Ratio System — Smart UPS Ratio Optimization + Zero Waste Planning</p>
<p class="badge">Version 3.0 | Enterprise Ready</p>
<p class="designer-credit">✨ Design & Developed by <strong style="color:#764ba2">Md Ovi</strong> ✨</p>
<p style="font-size:0.8rem; opacity:0.7;">© 2026 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
