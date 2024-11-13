import joblib
import pandas as pd
import json
import logging
from dice_ml import Data, Model, Dice

logging.basicConfig(level=logging.INFO)

def load_fixed_vocabulary(file_path):
    try:
        fixed_vocabulary = pd.read_csv(file_path)["Vocabulary"].tolist()
        logging.info("Fixed vocabulary loaded with {} terms.".format(len(fixed_vocabulary)))
        return fixed_vocabulary
    except Exception as e:
        logging.error(f"Failed to load fixed vocabulary from {file_path}: {e}")
        return []

# Load the fixed vocabulary and trained model
fixed_vocabulary = load_fixed_vocabulary('static/fixed_vocabulary.csv')
model = joblib.load("model/trained_model.joblib")

def initialize_dice(model, fixed_vocabulary):
    # Initialize with dummy data to match fixed vocabulary structure
    dummy_data = pd.DataFrame([[0] * len(fixed_vocabulary)], columns=fixed_vocabulary)
    dummy_data["label"] = [0]  # Dummy label for initialization

    data = Data(dataframe=dummy_data, continuous_features=fixed_vocabulary, outcome_name="label")
    dice_model = Model(model=model, backend="sklearn")
    return Dice(data, dice_model, method="random")

# Helper function to create an input DataFrame using fixed vocabulary and fill it with values from feature names
def prepare_input_data(feature_names, vectorized_description, fixed_vocabulary):
    # Start with a DataFrame of zeros, with fixed vocabulary as columns
    input_df = pd.DataFrame(0, index=[0], columns=fixed_vocabulary)

    # Fill in values based on feature names and vectorized descriptions
    for word, weight in zip(feature_names, vectorized_description):
        if word in input_df.columns:
            input_df.at[0, word] = weight

    # Remove columns where all values are zero to reduce dimensionality
    input_df = input_df.loc[:, (input_df != 0).any(axis=0)]

    logging.info(f"Prepared input data shape after filtering zeros: {input_df.shape}")
    return input_df

def get_dice_explanation(dice, feature_names, vectorized_description):
    try:
        # Prepare input data
        input_data = prepare_input_data(feature_names, vectorized_description, fixed_vocabulary)

        if input_data.shape[1] == 0:
            logging.error("Input data contains no features after filtering zeros.")
            return json.dumps({"error": "No features left in input data after filtering zero columns."})

        # Generate counterfactual explanation using DiCE
        cf = dice.generate_counterfactuals(input_data, total_CFs=1, desired_class="opposite")
        cf_example = cf.cf_examples_list[0]

        if hasattr(cf_example, "final_cfs_df") and isinstance(cf_example.final_cfs_df, pd.DataFrame):
            explanation_data = cf_example.final_cfs_df.to_dict(orient="records")
            logging.info("Dice explanation generated successfully.")
            return json.dumps(explanation_data)
        else:
            error_msg = "Counterfactual generation failed; final_cfs_df is not valid."
            logging.error(error_msg)
            return json.dumps({"error": error_msg})

    except Exception as e:
        error_message = f"Exception in get_dice_explanation: {str(e)}"
        logging.error(error_message)
        return json.dumps({"error": error_message})

# Initialize DiCE with the model and fixed vocabulary
dice = initialize_dice(model, fixed_vocabulary)
