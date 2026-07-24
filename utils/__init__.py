"""
Utilities Package - Helper functions for the Plate Ratio System
"""

from utils.helpers import (
    plate_name,
    create_valid_layout,
    ensure_demand_met,
    calculate_waste_percent,
    build_full_summary
)

from utils.pdf_generator import generate_pdf_report
from utils.excel_generator import generate_excel_report
