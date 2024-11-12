import joblib
import pandas as pd
from dice_ml.utils import helpers
from dice_ml import Data, Model, Dice
import json
import logging

print("Classes in dice_ml module:", dir(Dice))  # Print Dice class details
print("Classes in dice_ml.Data:", dir(Data))    # Print Data class details
print("Classes in dice_ml.Model:", dir(Model))  # Print Model class details

# Load the trained model and set up DiCE
# def initialize_dice():
#     # Load the model pipeline (TF-IDF + RandomForest)
#     model = joblib.load("trained_model.joblib")
#
#     # Define the feature names (from TF-IDF vectorizer)
#     feature_names = model.named_steps['tfidfvectorizer'].get_feature_names_out()
#
#     # Create a DataFrame with dummy data for DiCE's input requirements
#     data_df = pd.DataFrame(columns=feature_names)
#     data_df["label"] = [0, 1]  # Two dummy classes
#
#     # Initialize DiCE with this data and model
#     data = Data(dataframe=data_df, continuous_features=feature_names, outcome_name="label")
#     dice_model = Model(model=model.named_steps["randomforestclassifier"], backend="sklearn")
#     dice = Dice(data, dice_model, method="random")
#
#     return dice
model = joblib.load("model/trained_model.joblib")

def initialize_dice(model, tfidf_feature_names):
    # Create a DataFrame with dummy data using the TF-IDF feature names
    dummy_data = pd.DataFrame([[0] * len(tfidf_feature_names), [1] * len(tfidf_feature_names)], columns=tfidf_feature_names)
    dummy_data["label"] = [0, 1]  # Add a label column

    # Initialize DiCE with the dummy data
    data = Data(dataframe=dummy_data, continuous_features=tfidf_feature_names, outcome_name="label")

    # Pass in the model directly if it's already trained
    dice_model = Model(model=model, backend="sklearn")
    dice = Dice(data, dice_model, method="random")

    return dice


def pad_missing_columns(input_data, model):
    # Get the expected columns from the model (adjust based on model's actual method for feature names)
    model_feature_names = model.get_feature_names_out()  # Ensure this matches your model's feature retrieval method

    # Pad input data with missing columns
    for feature in model_feature_names:
        if feature not in input_data.columns:
            input_data[feature] = 0  # Fill missing columns with zero

    # Drop any extra columns not expected by the model
    return input_data[model_feature_names]  # Align columns exactly with model

def get_dice_explanation(dice, input_data, feature_names):
    feature_names = tfidf_data.get("feature_names", [])

    try:
        # Ensure input_data is a DataFrame
        if isinstance(input_data, dict):
            print("Converting input_data dict to DataFrame")
            input_data = pd.DataFrame([input_data])  # Convert to single-row DataFrame

        # Pad missing columns based on the dynamically generated TF-IDF feature names
        input_data = pad_missing_columns(input_data, feature_names)

        # Logging the TF-IDF feature names used for counterfactuals
        print("Using dynamically generated TF-IDF feature names:", feature_names)

        # Generate counterfactuals
        cf = dice.generate_counterfactuals(input_data, total_CFs=1, desired_class="opposite")

        # Process the generated counterfactuals
        cf_example = cf.cf_examples_list[0]
        if hasattr(cf_example, "final_cfs_df") and isinstance(cf_example.final_cfs_df, pd.DataFrame):
            explanation_data = cf_example.final_cfs_df.to_dict(orient="records")
            json_explanation = json.dumps(explanation_data)
            print("Explanation JSON structure:", json_explanation)
            return json_explanation
        else:
            error_msg = "Counterfactual generation failed; final_cfs_df is not a DataFrame or is missing."
            print(error_msg)
            return json.dumps({"error": error_msg})

    except Exception as e:
        error_message = f"Exception in get_dice_explanation: {str(e)} of type {type(e).__name__}"
        print(error_message)
        return json.dumps({"error": error_message})
