"""
Algorithm Engine - Runs all algorithms, picks the best, validates plate capacity
"""
import pandas as pd
import streamlit as st

from algorithms import ALGORITHM_REGISTRY, get_algorithm
from algorithms.v1_helpers import calculate_waste_percent


def validate_plate_capacity(plates, demand, capacity):
    """
    Validate that each plate's total UPS equals the plate capacity.
    If not, fix the layout to match capacity.
    """
    if not plates:
        return plates

    for plate in plates:
        layout = plate["layout"]
        total_ups = sum(layout.values())

        if total_ups != capacity:
            while sum(layout.values()) > capacity:
                max_tag = max(layout, key=layout.get)
                if layout[max_tag] > 1:
                    layout[max_tag] -= 1
                else:
                    min_tag = min(layout.keys(), key=lambda t: layout.get(t, 0))
                    if layout.get(min_tag, 0) > 0:
                        layout[min_tag] = 0
                    else:
                        break

            while sum(layout.values()) < capacity:
                best_tag = max(layout.keys(), key=lambda t: demand.get(t, 0) / (layout.get(t, 1) + 1))
                layout[best_tag] = layout.get(best_tag, 0) + 1

        plate["layout"] = layout

    return plates


def run_algorithms(demand, capacity, max_plates):
    """
    Runs every registered algorithm, shows progress, returns:
    (results dict, comparison_df, best_algo, best_waste, best_plates)
    or (None, None, None, None, None) if none succeeded.
    """
    with st.spinner(f"🔍 Running {len(ALGORITHM_REGISTRY)} algorithms..."):
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        algo_list = list(ALGORITHM_REGISTRY.items())

        for idx, (algo_name, _) in enumerate(algo_list):
            status_text.text(f"Running {algo_name}... ({idx+1}/{len(algo_list)})")
            try:
                optimizer = get_algorithm(algo_name, demand, capacity, max_plates)
                plates = optimizer.optimize()
                if plates:
                    results[algo_name] = plates
            except Exception:
                pass
            progress_bar.progress((idx + 1) / len(algo_list))

        progress_bar.empty()
        status_text.empty()

    if not results:
        return None, None, None, None, None

    comparison_data = []
    for algo_name, plates in results.items():
        if plates:
            waste = calculate_waste_percent(plates, demand)
            comparison_data.append({
                "Algorithm": algo_name,
                "Waste %": waste,
                "Plates": len(plates),
                "Sheets": sum(p.get("sheets", 0) for p in plates)
            })

    comparison_df = pd.DataFrame(comparison_data).sort_values("Waste %")
    best_algo = comparison_df.iloc[0]["Algorithm"]
    best_waste = comparison_df.iloc[0]["Waste %"]
    best_plates = results[best_algo]

    # Validate/fix plate capacity for the winning algorithm
    best_plates = validate_plate_capacity(best_plates, demand, capacity)

    return results, comparison_df, best_algo, best_waste, best_plates
