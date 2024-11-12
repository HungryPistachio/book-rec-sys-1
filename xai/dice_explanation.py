import joblib
import pandas as pd
from dice_ml.utils import helpers
from dice_ml import Data, Model, Dice
import json

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

    # Define the feature names (from TF-IDF vectorizer)
    feature_names = model.named_steps['tfidfvectorizer'].get_feature_names_out()

    # Create a DataFrame with numeric dummy data for DiCE's input requirements
    data_df = pd.DataFrame({
        "feature1": [1.0, 2.0],  # Ensure features are float
        "feature2": [1.5, 2.5]
    })
    data_df["label"] = [0, 1]  # Label should also be an integer

    # Initialize DiCE with this data and model
    data = Data(dataframe=data_df, continuous_features=["feature1", "feature2"], outcome_name="label")
    dice_model = Model(model=model.named_steps["randomforestclassifier"], backend="sklearn")
    dice = Dice(data, dice_model, method="random")

    return dice

# Function to get a counterfactual explanation for a recommendation
# Function to get a counterfactual explanation for a recommendation
def get_dice_explanation(dice, input_data):
    try:
        # Generate counterfactuals
        cf = dice.generate_counterfactuals(input_data, total_CFs=1, desired_class="opposite")
        # Convert the result to JSON string
        return json.dumps(cf.cf_examples_list[0].final_cfs_df.to_dict())
    except Exception as e:
        return json.dumps({"error": str(e)})  # Return error as JSON string
