import logging

def pad_missing_columns(input_data, fixed_vocabulary):
    # Pad input data with missing columns from fixed_vocabulary
    for feature in fixed_vocabulary:
        if feature not in input_data.columns:
            input_data[feature] = 0  # Fill missing columns with zero

    # Log the current state of columns in input_data after padding
    logging.info(f"Input data columns after padding: {len(input_data.columns)} columns")
    logging.info(f"Expected columns (fixed vocabulary): {len(fixed_vocabulary)} columns")

    # Ensure columns are in the same order as fixed_vocabulary
    input_data = input_data.reindex(columns=fixed_vocabulary, fill_value=0)

    # Log the column names to ensure correct alignment
    logging.debug(f"Input data column names: {input_data.columns.tolist()}")
    logging.debug(f"Fixed vocabulary column names: {fixed_vocabulary}")

    return input_data
