# ================================================================
# PDF QTY UPLOAD SYSTEM
# ================================================================

# Required Libraries
import fitz  # pymupdf
import re
import pandas as pd
from io import BytesIO

# ================================================================
# PDF EXTRACT FUNCTION
# ================================================================

def extract_qty_from_pdf(uploaded_pdf):

    doc = fitz.open(
        stream=uploaded_pdf.read(),
        filetype="pdf"
    )

    full_text = ""

    for page in doc:

        full_text += page.get_text()

    # ============================================================
    # EXAMPLE PATTERN
    # ============================================================
    # XS 12000
    # S  15000
    # M  18000
    #
    # Adjust regex according to your actual PDF format
    # ============================================================

    pattern = r'([A-Za-z0-9\\-\\/]+)\\s+(\\d{2,})'

    matches = re.findall(
        pattern,
        full_text
    )

    data = []

    for tag, qty in matches:

        try:

            qty = int(qty)

            if qty > 0:

                data.append({
                    "Tag": tag.strip(),
                    "Qty": qty
                })

        except:
            pass

    # ============================================================
    # REMOVE DUPLICATES
    # ============================================================

    df = pd.DataFrame(data)

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

input_mode = st.radio(
    "Select Input Method",
    [
        "Manual Entry",
        "Upload PDF"
    ]
)

# ================================================================
# MANUAL ENTRY MODE
# ================================================================

if input_mode == "Manual Entry":

    # KEEP YOUR EXISTING MANUAL INPUT SYSTEM
    # DO NOT CHANGE ANYTHING

    pass


# ================================================================
# PDF UPLOAD MODE
# ================================================================

if input_mode == "Upload PDF":

    uploaded_pdf = st.file_uploader(
        "Upload Work Order PDF",
        type=["pdf"]
    )

    if uploaded_pdf:

        # ========================================================
        # EXTRACT DATA
        # ========================================================

        df_pdf = extract_qty_from_pdf(
            uploaded_pdf
        )

        # ========================================================
        # VALIDATION
        # ========================================================

        if df_pdf.empty:

            st.error(
                "❌ No valid Tag/Qty found"
            )

            st.stop()

        # ========================================================
        # PREVIEW
        # ========================================================

        st.success(
            "✅ PDF Data Extracted"
        )

        st.dataframe(
            df_pdf,
            use_container_width=True
        )

        # ========================================================
        # TOTAL QTY
        # ========================================================

        st.metric(
            "Total Order Qty",
            int(df_pdf["Qty"].sum())
        )

        # ========================================================
        # CONVERT TO EXISTING SYSTEM
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
        # BUILD EXISTING VARIABLES
        # ========================================================

        original_qty = {
            t: int(q)
            for t, q in zip(tags, qtys)
            if q > 0
        }

        demand = {
            t: ceil(
                int(q)
                *
                (1 + addon / 100)
            )
            for t, q in zip(tags, qtys)
            if q > 0
        }

        # ========================================================
        # OPTIONAL DEBUG
        # ========================================================

        st.write("Original Qty")

        st.json(original_qty)

        st.write("Demand Qty")

        st.json(demand)


# ================================================================
# SAMPLE TEMPLATE DOWNLOAD
# ================================================================

sample_df = pd.DataFrame({
    "Tag": ["XS", "S", "M", "L"],
    "Qty": [12000, 15000, 18000, 10000]
})

sample_buffer = BytesIO()

with pd.ExcelWriter(
    sample_buffer,
    engine="openpyxl"
) as writer:

    sample_df.to_excel(
        writer,
        index=False
    )

sample_buffer.seek(0)

st.download_button(
    "⬇️ Download Sample Template",
    data=sample_buffer,
    file_name="sample_qty_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
