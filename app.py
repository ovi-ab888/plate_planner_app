# ================================================================
# PDF QTY UPLOAD SYSTEM
# ================================================================

import fitz
import re
import pandas as pd
from math import ceil


# ================================================================
# PDF EXTRACT FUNCTION
# ================================================================

def extract_qty_from_pdf(uploaded_pdf):

    # ============================================================
    # OPEN PDF
    # ============================================================

    doc = fitz.open(
        stream=uploaded_pdf.read(),
        filetype="pdf"
    )

    # ============================================================
    # EXTRACT ALL TEXT
    # ============================================================

    full_text = ""

    for page in doc:

        full_text += page.get_text()

    # ============================================================
    # OPTIONAL DEBUG
    # ============================================================

    # st.text(full_text)

    # ============================================================
    # REGEX PATTERN
    # ============================================================
    #
    # Example PDF Line:
    #
    # 1-1½; YRS N/A 12 Nightwear 320.00
    #
    # Extract:
    # Tag = 1-1½; YRS
    # Qty = 320
    #
    # ============================================================

    pattern = r'([0-9A-Za-z½;\-\\s]+YRS).*?(\\d+\\.\\d+)'

    matches = re.findall(
        pattern,
        full_text
    )

    # ============================================================
    # BUILD DATA
    # ============================================================

    data = []

    for tag, qty in matches:

        try:

            qty = int(float(qty))

            if qty > 0:

                data.append({
                    "Tag": tag.strip(),
                    "Qty": qty
                })

        except:
            pass

    # ============================================================
    # CREATE DATAFRAME
    # ============================================================

    df = pd.DataFrame(data)

    # ============================================================
    # REMOVE DUPLICATES
    # ============================================================

    if not df.empty:

        df = (
            df.groupby("Tag")["Qty"]
            .sum()
            .reset_index()
        )

    return df


# ================================================================
# INPUT METHOD SELECTOR
# ================================================================

st.sidebar.markdown("## 📥 Input Method")

input_mode = st.sidebar.radio(
    "Select Input Source",
    [
        "Manual Entry",
        "Upload PDF"
    ]
)


# ================================================================
# GLOBAL VARIABLES
# ================================================================

tags = []
qtys = []

original_qty = {}
demand = {}

addon = 0


# ================================================================
# MANUAL ENTRY SYSTEM
# ================================================================

if input_mode == "Manual Entry":

    st.markdown("## ✍️ Manual Qty Entry")

    col1, col2 = st.columns(2)

    n = col1.number_input(
        "Tag Count",
        min_value=1,
        max_value=100,
        value=5
    )

    addon = col2.number_input(
        "Add-on %",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.5
    )

    left, right = st.columns(2)

    for i in range(n):

        tag = left.text_input(
            f"Tag {i+1}",
            key=f"tag_{i}"
        )

        qty = right.number_input(
            f"Qty {i+1}",
            min_value=0,
            value=0,
            step=100,
            key=f"qty_{i}"
        )

        if tag and qty > 0:

            tags.append(tag)

            qtys.append(qty)

    # ============================================================
    # ORIGINAL QTY
    # ============================================================

    original_qty = {
        t: int(q)
        for t, q in zip(tags, qtys)
    }

    # ============================================================
    # DEMAND WITH ADDON
    # ============================================================

    demand = {
        t: ceil(
            int(q)
            *
            (1 + addon / 100)
        )
        for t, q in zip(tags, qtys)
    }


# ================================================================
# PDF UPLOAD SYSTEM
# ================================================================

if input_mode == "Upload PDF":

    st.markdown("## 📄 Upload Work Order PDF")

    # ============================================================
    # ADDON INPUT
    # ============================================================

    addon = st.number_input(
        "Add-on %",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.5
    )

    # ============================================================
    # FILE UPLOAD
    # ============================================================

    uploaded_pdf = st.file_uploader(
        "Upload Work Order PDF",
        type=["pdf"]
    )

    # ============================================================
    # PROCESS PDF
    # ============================================================

    if uploaded_pdf:

        # ========================================================
        # EXTRACT PDF DATA
        # ========================================================

        df_pdf = extract_qty_from_pdf(
            uploaded_pdf
        )

        # ========================================================
        # VALIDATION
        # ========================================================

        if df_pdf.empty:

            st.error(
                "❌ No Qty Found In PDF"
            )

            st.stop()

        # ========================================================
        # SUCCESS MESSAGE
        # ========================================================

        st.success(
            "✅ PDF Extracted Successfully"
        )

        # ========================================================
        # PREVIEW TABLE
        # ========================================================

        st.markdown("### 📋 Extracted Qty Preview")

        st.dataframe(
            df_pdf,
            use_container_width=True
        )

        # ========================================================
        # TOTAL QTY
        # ========================================================

        total_qty = int(
            df_pdf["Qty"].sum()
        )

        st.metric(
            "Total Order Qty",
            f"{total_qty:,}"
        )

        # ========================================================
        # CONVERT TO LIST
        # ========================================================

        tags = (
            df_pdf["Tag"]
            .astype(str)
            .tolist()
        )

        qtys = (
            df_pdf["Qty"]
            .astype(int)
            .tolist()
        )

        # ========================================================
        # ORIGINAL QTY
        # ========================================================

        original_qty = {
            t: int(q)
            for t, q in zip(tags, qtys)
        }

        # ========================================================
        # DEMAND WITH ADDON
        # ========================================================

        demand = {
            t: ceil(
                int(q)
                *
                (1 + addon / 100)
            )
            for t, q in zip(tags, qtys)
        }

        # ========================================================
        # OPTIONAL DEBUG
        # ========================================================

        # st.json(original_qty)
        # st.json(demand)


# ================================================================
# CHECK BEFORE RUNNING ALGORITHM
# ================================================================

if st.button("🚀 Generate Plan"):

    if not demand:

        st.error(
            "❌ Please Input Qty First"
        )

        st.stop()

    st.success(
        "✅ Qty Ready For Optimization"
    )

    # ============================================================
    # CALL YOUR V3 / V4 / V5 / V6 / V7 ALGORITHM HERE
    # ============================================================

    # Example:
    #
    # result = optimizer_v7(
    #     demand,
    #     plate_capacity,
    #     max_plates
    # )
