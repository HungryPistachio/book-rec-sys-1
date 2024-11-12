def pad_missing_columns(input_data, feature_names):
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0
    return input_data[feature_names]  # Ensure the column order matches feature_names
