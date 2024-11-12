import joblib
import pandas as pd
from dice_ml.utils import helpers
from dice_ml import Dice, Data, Model

# Load the trained model and set up DiCE
def initialize_dice():
    # Load the model pipeline (TF-IDF + RandomForest)
    model = joblib.load("trained_model.joblib")

    # Define the feature names (from TF-IDF vectorizer)
    feature_names = model.named_steps['tfidfvectorizer'].get_feature_names_out()

    # Create a DataFrame with dummy data for DiCE's input requirements
    data_df = pd.DataFrame(columns=feature_names)
    data_df["label"] = [0, 1]  # Two dummy classes

    # Initialize DiCE with this data and model
    data = Data(dataframe=data_df, continuous_features=feature_names, outcome_name="label")
    dice_model = Model(model=model.named_steps["randomforestclassifier"], backend="sklearn")
    dice = Dice(data, dice_model, method="random")

    return dice


# Function to get a counterfactual explanation for a recommendation
def get_dice_explanation(dice, input_data):
    try:
        # Generate counterfactuals
        cf = dice.generate_counterfactuals(input_data, total_CFs=1, desired_class="opposite")
        return cf.cf_examples_list[0].final_cfs_df.to_json(orient='records')
    except Exception as e:
        return {"error": str(e)}
