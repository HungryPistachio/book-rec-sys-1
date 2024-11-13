import joblib
import pandas as pd
from dice_ml import Data, Model, Dice
import json
import logging

def load_fixed_vocabulary(file_path, expected_size=79):
    try:
        fixed_vocabulary = pd.read_csv(file_path)["Vocabulary"].tolist()
        if len(fixed_vocabulary) != expected_size:
            logging.warning(f"Fixed vocabulary size mismatch. Expected {expected_size}, got {len(fixed_vocabulary)}.")
        return fixed_vocabulary[:expected_size]
    except Exception as e:
        logging.error(f"Failed to load fixed vocabulary from {file_path}: {e}")
        return []

def pad_missing_columns(input_data, fixed_vocabulary):
    return input_data.reindex(columns=fixed_vocabulary, fill_value=0)

fixed_vocabulary = load_fixed_vocabulary('static/fixed_vocabulary.csv')
model = joblib.load("model/trained_model.joblib")

def initialize_dice(model, fixed_vocabulary):
    dummy_data = pd.DataFrame([[0] * len(fixed_vocabulary), [1] * len(fixed_vocabulary)], columns=fixed_vocabulary)
    dummy_data["label"] = [0, 1]
    data = Data(dataframe=dummy_data, continuous_features=fixed_vocabulary, outcome_name="label")
    dice_model = Model(model=model, backend="sklearn")
    return Dice(data, dice_model, method="random")

def get_dice_explanation(dice, input_data):
    try:
        input_data = pad_missing_columns(input_data, fixed_vocabulary)
        if input_data.shape[1] != len(fixed_vocabulary):
            logging.error("Mismatch between input data columns and fixed vocabulary size.")
            return json.dumps({"error": "Input data does not match the required vocabulary size."})
        cf = dice.generate_counterfactuals(input_data, total_CFs=1, desired_class="opposite")
        cf_example = cf.cf_examples_list[0]
        if hasattr(cf_example, "final_cfs_df") and isinstance(cf_example.final_cfs_df, pd.DataFrame):
            return json.dumps(cf_example.final_cfs_df.to_dict(orient="records"))
        else:
            logging.error("Counterfactual generation failed; final_cfs_df is not valid.")
            return json.dumps({"error": "Counterfactual generation failed."})
    except Exception as e:
        logging.error(f"Exception in get_dice_explanation: {e}")
        return json.dumps({"error": str(e)})
