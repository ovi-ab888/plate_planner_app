"""
Item Entry Module - Manual Entry & Excel Upload (Dynamic Column Support)
"""
import math
import pandas as pd
import streamlit as st

# Keywords used to auto-detect the Quantity column
QTY_KEYWORDS = ['quantity', 'qty', 'qty.', 'quantities', 'total', 'pcs', 'pieces', 'count']


def render_manual_entry(n_items=5):
    """Renders an Excel-style editable grid for manual item entry"""

    if 'manual_entry_df' not in st.session_state:
        st.session_state['manual_entry_df'] = pd.DataFrame(
            [{"Style": "", "Color": "", "Size": "", "Quantity": 0} for _ in range(n_items)]
        )
        st.session_state['manual_entry_last_n'] = n_items

    # ================================================================
    # "Number of Items" change hole grid resize kora
    # ================================================================
    if st.session_state.get('manual_entry_last_n') != n_items:
        current_df = st.session_state['manual_entry_df']
        current_len = len(current_df)

        if n_items > current_len:
            extra_rows = pd.DataFrame(
                [{"Style": "", "Color": "", "Size": "", "Quantity": 0} for _ in range(n_items - current_len)]
            )
            st.session_state['manual_entry_df'] = pd.concat([current_df, extra_rows], ignore_index=True)
        elif n_items < current_len:
            st.session_state['manual_entry_df'] = current_df.iloc[:n_items].reset_index(drop=True)

        st.session_state['manual_entry_last_n'] = n_items
        if 'manual_entry_editor' in st.session_state:
            del st.session_state['manual_entry_editor']



    # ================================================================
    # Excel-style Data Editor Grid
    # ================================================================
    edited_df = st.data_editor(
        st.session_state['manual_entry_df'],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="manual_entry_editor",
        column_config={
            "Style": st.column_config.TextColumn("Style", width="medium", required=False),
            "Color": st.column_config.TextColumn("Color", width="medium", required=False),
            "Size": st.column_config.TextColumn("Size", width="medium", required=False),
            "Quantity": st.column_config.NumberColumn("Quantity", min_value=0, step=1, width="small", required=False),
        },
    )

    # ================================================================
    # Grid theke data + item_meta build kora
    # ================================================================
    data = []
    item_meta = {}
    meta_columns = ["Style", "Color", "Size"]

    for idx, row in edited_df.iterrows():
        raw_qty = row.get("Quantity", 0)
        try:
            qty = int(raw_qty) if not pd.isna(raw_qty) else 0
        except (ValueError, TypeError):
            qty = 0

        if qty > 0:
            style_val = str(row.get("Style", "") or "").strip() or f"Item {idx + 1}"
            color_val = str(row.get("Color", "") or "").strip() or "N/A"
            size_val = str(row.get("Size", "") or "").strip() or "N/A"

            data.append({
                "SL": idx + 1,
                "Style": style_val,
                "Color": color_val,
                "Size": size_val,
                "Quantity": qty
            })
            item_meta[style_val] = {
                "Style": style_val,
                "Color": color_val,
                "Size": size_val
            }

    st.session_state['item_meta'] = item_meta
    st.session_state['item_meta_columns'] = meta_columns

    return data


def render_excel_upload(addon_percent):
    """
    Renders Excel upload UI, auto-detects the Quantity column,
    treats ALL other columns as dynamic metadata (original name + order preserved).
    Returns (original_qty, demand) dicts, or calls st.stop() if not ready.
    """

    uploaded_file = st.file_uploader(
        "Upload Excel file with Item Details",
        type=["xlsx", "xls"],
        help="Quantity column auto-detect hobe. Baki column (jekono naam) shoja report-e chole jabe."
    )

    if uploaded_file is None:
        st.info("📤 Please upload an Excel file to continue.")
        st.stop()

    try:
        df = pd.read_excel(uploaded_file)
        df = df.dropna(how='all')

        if df.empty or len(df.columns) == 0:
            st.error("❌ File-e kono valid data pawa jayni.")
            st.stop()

        # ================================================================
        # STEP 1: Auto-detect Quantity column (keyword match)
        # ================================================================
        qty_col = None
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if col_lower in QTY_KEYWORDS:
                qty_col = col
                break

        # Fallback: first numeric column
        if qty_col is None:
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    qty_col = col
                    break

        # Fallback: last column
        if qty_col is None:
            qty_col = df.columns[-1]

        if qty_col is None:
            st.error("❌ Could not find 'Quantity' column. Please ensure your file has a quantity column.")
            st.info("📌 Column names can be: 'Quantity', 'Qty', 'QTY', 'Total', 'Pcs'")
            st.stop()

        # ================================================================
        # STEP 2: Baki shob column = dynamic meta column (original order-e)
        # ================================================================
        meta_cols = [c for c in df.columns if c != qty_col]
        # Sompurno khali column (jegula shob row e blank/NaN) bad dao - eigula stray "Unnamed" column
        meta_cols = [c for c in meta_cols if not str(c).lower().strip().startswith('unnamed')]

        # ================================================================
        # STEP 3: Row by row process
        # ================================================================
        item_meta = {}
        tags = []
        qty_list = []
        preview_rows = []
        skipped_rows = 0

        for idx, row in df.iterrows():
            raw_qty = row[qty_col]

            if pd.isna(raw_qty):
                skipped_rows += 1
                continue

            try:
                qty_int = int(float(raw_qty))
            except (ValueError, TypeError):
                skipped_rows += 1
                continue

            if qty_int <= 0:
                skipped_rows += 1
                continue

            tag = f"Item {len(tags) + 1}"

            meta = {}
            preview_row = {}
            for col in meta_cols:
                val = row[col]
                val_str = "" if pd.isna(val) else str(val).strip()
                meta[str(col)] = val_str
                preview_row[str(col)] = val_str

            preview_row["Quantity"] = qty_int

            tags.append(tag)
            qty_list.append(qty_int)
            item_meta[tag] = meta
            preview_rows.append(preview_row)

        if not tags:
            st.error("❌ No valid data found in the file. Please check the format.")
            st.stop()

        preview_df = pd.DataFrame(preview_rows)

        st.success(f"✅ File loaded successfully! {len(tags)} valid items found.")
        if skipped_rows > 0:
            st.warning(f"⚠️ {skipped_rows} rows were skipped (empty or invalid quantity).")

        st.dataframe(preview_df, use_container_width=True)
        st.info(f"📋 Quantity column detected: '{qty_col}' | Other columns: {', '.join(str(c) for c in meta_cols)}")

        # ================================================================
        # STEP 4: Session state-e dynamic meta store
        # ================================================================
        st.session_state['item_meta'] = item_meta
        st.session_state['item_meta_columns'] = [str(c) for c in meta_cols]

        original_qty = {}
        demand = {}
        for t, q in zip(tags, qty_list):
            original_qty[t] = int(q)
            demand[t] = math.ceil(int(q) * (1 + addon_percent / 100))


        return original_qty, demand

    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.stop()
