# utils/excel_generator.py
"""
Excel Report Generator - Dynamic Column Support
"""
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

import pandas as pd
from io import BytesIO
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_full_summary(plates, demand, original_qty, item_meta=None, meta_columns=None):
    """Build complete summary DataFrame, with dynamic meta columns inserted after SL"""
    item_meta = item_meta or {}
    meta_columns = meta_columns or []

    rows = []
    sl = 1
    for tag in demand.keys():
        row = {"SL": sl}

        # Dynamic columns (original file er column name + order)
        meta = item_meta.get(tag, {})
        for col in meta_columns:
            row[col] = meta.get(col, "")

        row["Original QTY"] = original_qty.get(tag, 0)
        row["Produced (+Add-on)"] = demand[tag]

        for idx, p in enumerate(plates):
            if p and "layout" in p and "name" in p:
                ups = p["layout"].get(tag, 0)
                row[f"Plate {p['name']}"] = ups
            else:
                row[f"Plate {idx+1}"] = 0

        total_produced = 0
        for p in plates:
            if p and "layout" in p:
                ups = p["layout"].get(tag, 0)
                sheets = p.get("sheets", 0)
                total_produced += ups * sheets

        excess = total_produced - demand[tag]
        excess_percent = round((excess / demand[tag]) * 100, 2) if demand[tag] else 0

        row["Total Produced QTY"] = total_produced
        row["Excess"] = max(0, excess)
        row["Excess %"] = f"{max(0, excess_percent)}%"

        rows.append(row)
        sl += 1

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    total_row = {"SL": "📊"}
    for i, col in enumerate(meta_columns):
        total_row[col] = "TOTAL" if i == 0 else ""

    total_row["Original QTY"] = df["Original QTY"].sum()
    total_row["Produced (+Add-on)"] = df["Produced (+Add-on)"].sum()

    for idx, p in enumerate(plates):
        col_name = f"Plate {p['name']}" if "name" in p else f"Plate {idx+1}"
        total_row[col_name] = df[col_name].sum() if col_name in df.columns else 0

    total_row["Total Produced QTY"] = df["Total Produced QTY"].sum()
    total_excess = df["Excess"].sum()
    total_row["Excess"] = total_excess

    total_produced_qty = total_row["Total Produced QTY"]
    total_excess_percent = round((total_excess / total_produced_qty) * 100, 2) if total_produced_qty > 0 else 0
    total_row["Excess %"] = f"{total_excess_percent}%"

    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    return df


def generate_excel_report(plates, demand, original_qty, algo_name, waste_percent,
                           job_number="", item_meta=None, meta_columns=None):
    """Generate Excel report with multiple sheets"""
    try:
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            # Sheet 1: Summary
            summary_df = build_full_summary(plates, demand, original_qty, item_meta, meta_columns)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # Sheet 2: Plate Details
            plate_rows = []
            total_sheets_sum = 0
            total_ups_sum = 0
            for idx, p in enumerate(plates, 1):
                total_ups = sum(p["layout"].values())
                plate_rows.append({
                    "SL": idx,
                    "Plate ID": p.get("name", f"Plate {idx}"),
                    "Sheets Required": p.get("sheets", 0),
                    "Total UPS": total_ups,
                })
                total_sheets_sum += p.get("sheets", 0)
                total_ups_sum += total_ups

            plate_rows.append({
                "SL": "TOTAL",
                "Plate ID": "",
                "Sheets Required": total_sheets_sum,
                "Total UPS": total_ups_sum,
            })
            plate_df = pd.DataFrame(plate_rows)
            plate_df.to_excel(writer, sheet_name="Plate Details", index=False)

            # Sheet 3: Info
            info_df = pd.DataFrame({
                "Property": [
                    "Algorithm",
                    "Waste %",
                    "Total Plates",
                    "Total Sheets",
                    "Job Number",
                    "Generated On",
                    "Total Items"
                ],
                "Value": [
                    algo_name,
                    f"{waste_percent}%",
                    len(plates),
                    total_sheets_sum,
                    job_number if job_number else "N/A",
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(demand)
                ]
            })
            info_df.to_excel(writer, sheet_name="Info", index=False)

            # Sheet 4: Layout Details
            layout_rows = []
            for idx, p in enumerate(plates, 1):
                layout_str = ", ".join([f"{tag}:{ups}" for tag, ups in p["layout"].items()])
                layout_rows.append({
                    "Plate ID": p.get("name", f"Plate {idx}"),
                    "Sheets": p.get("sheets", 0),
                    "Layout": layout_str,
                    "Total UPS": sum(p["layout"].values())
                })
            layout_df = pd.DataFrame(layout_rows)
            layout_df.to_excel(writer, sheet_name="Layout Details", index=False)

        bio.seek(0)
        return bio

    except Exception as e:
        print(f"Excel Generation Error: {str(e)}")
        return None
