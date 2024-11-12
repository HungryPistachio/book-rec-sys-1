# utils.py

import pandas as pd

def pad_missing_columns(input_data, feature_names):
    # Add missing columns to input_data to match feature_names, setting them to zero if missing
    for feature in feature_names:
        if feature not in input_data.columns:
            input_data[feature] = 0
    return input_data[feature_names]
