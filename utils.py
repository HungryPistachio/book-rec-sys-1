def pad_missing_columns(input_data, fixed_vocabulary):
    # Pad input data with missing columns from fixed vocabulary
    for feature in fixed_vocabulary:
        if feature not in input_data.columns:
            input_data[feature] = 0  # Fill missing columns with zero

    # Drop any extra columns not in the fixed vocabulary
    return input_data[fixed_vocabulary]
