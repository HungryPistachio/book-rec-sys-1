import joblib
import pandas as pd
from dice_ml.utils import helpers
from dice_ml import Data, Model, Dice
import json
from utils import pad_missing_columns
import logging
import pandas as pd

print("Classes in dice_ml module:", dir(Dice))  # Print Dice class details
print("Classes in dice_ml.Data:", dir(Data))    # Print Data class details
print("Classes in dice_ml.Model:", dir(Model))  # Print Model class details


def load_fixed_vocabulary(file_path):
    try:
        fixed_vocabulary = pd.read_csv(file_path)["Vocabulary"].tolist()
        logging.info("Fixed vocabulary loaded successfully.")
        return fixed_vocabulary
    except Exception as e:
        logging.error(f"Failed to load fixed vocabulary from {file_path}: {e}")
        return []
# Load the fixed vocabulary once when the module is imported
fixed_vocabulary = load_fixed_vocabulary('static/fixed_vocabulary.csv')
model = joblib.load("model/trained_model.joblib")

def initialize_dice(model, fixed_vocabulary):
    dummy_data = pd.DataFrame([[0] * len(fixed_vocabulary), [1] * len(fixed_vocabulary)], columns=fixed_vocabulary)
    dummy_data["label"] = [0, 1]

    data = Data(dataframe=dummy_data, continuous_features=fixed_vocabulary, outcome_name="label")
    dice_model = Model(model=model, backend="sklearn")
    dice = Dice(data, dice_model, method="random")

    return dice


def get_dice_explanation(dice, input_data):
    try:
        # Ensure input_data is a DataFrame
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])  # Convert to single-row DataFrame

        # Pad missing columns using fixed vocabulary
        input_data = pad_missing_columns(input_data, fixed_vocabulary)

        # Log final column count after padding and alignment
        logging.info(f"Final input_data column count after padding and alignment: {len(input_data.columns)}")
        logging.debug(f"Final input_data column names: {input_data.columns.tolist()}")

        # Generate counterfactuals
        cf = dice.generate_counterfactuals(input_data, total_CFs=1, desired_class="opposite")

        # Process the generated counterfactuals
        cf_example = cf.cf_examples_list[0]
        if hasattr(cf_example, "final_cfs_df") and isinstance(cf_example.final_cfs_df, pd.DataFrame):
            explanation_data = cf_example.final_cfs_df.to_dict(orient="records")
            json_explanation = json.dumps(explanation_data)
            return json_explanation
        else:
            error_msg = "Counterfactual generation failed; final_cfs_df is not a DataFrame or is missing."
            logging.error(error_msg)
            return json.dumps({"error": error_msg})

    except Exception as e:
        error_message = f"Exception in get_dice_explanation: {str(e)} of type {type(e).__name__}"
        logging.error(error_message)
        return json.dumps({"error": error_message})









