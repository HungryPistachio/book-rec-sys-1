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
def initialize_dice():
    # Load the model pipeline (TF-IDF + RandomForest)
    model = joblib.load("model/trained_model.joblib")

    # Extract the feature names from the TF-IDF vectorizer
    feature_names = model.named_steps['tfidfvectorizer'].get_feature_names_out()

    # Create a DataFrame with dummy data
    # Ensure all features are numeric and match the feature names
    dummy_data = pd.DataFrame([[0]*len(feature_names), [1]*len(feature_names)], columns=feature_names)
    dummy_data["label"] = [0, 1]  # Add a label column

    # Initialize DiCE with the dummy data
    data = Data(dataframe=dummy_data, continuous_features=feature_names.tolist(), outcome_name="label")
    dice_model = Model(model=model.named_steps["randomforestclassifier"], backend="sklearn")
    dice = Dice(data, dice_model, method="random")

    return dice

# Function to get a counterfactual explanation for a recommendation

# dice_explanation.py
def get_dice_explanation(dice, input_data):
    try:
        # Ensure input_data is a DataFrame
        if isinstance(input_data, dict):
            print("Converting input_data dict to DataFrame")
            input_data = pd.DataFrame([input_data])  # Convert to single-row DataFrame

        # Remove any columns named "12th"
        if "12th" in input_data.columns:
            print("Removing column '12th' from input_data.")
            input_data = input_data.drop(columns=["12th"])

        # Remove any occurrences of "12th" within data values
        input_data = input_data.applymap(lambda x: None if x == "12th" else x)

        # Log the modified data structure for verification
        print("Modified input_data structure after removal process:")
        print("Column names:", input_data.columns.tolist())
        print("Data sample:\n", input_data.head())

        # Generate counterfactuals
        cf = dice.generate_counterfactuals(input_data, total_CFs=1, desired_class="opposite")

        # Process the generated counterfactuals as usual
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






