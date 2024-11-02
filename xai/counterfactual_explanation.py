import dice_ml
from dice_ml import Dice
from sklearn.linear_model import LogisticRegression
import pandas as pd

def get_counterfactual_explanation(book_title, description_vector, tfidf_matrix, feature_names):
    # Create a simple logistic regression model
    model = LogisticRegression()
    labels = [1 if "mystery" in desc else 0 for desc in tfidf_matrix]  # Example binary labels
    model.fit(tfidf_matrix, labels)

    # Prepare DiCE data and model objects
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=feature_names)
    tfidf_df['label'] = labels

    d = dice_ml.Data(dataframe=tfidf_df, continuous_features=[], outcome_name='label')
    m = dice_ml.Model(model=model, backend="sklearn")
    exp = Dice(d, m)

    # Generate counterfactuals using pre-vectorized query instance
    query_instance = pd.DataFrame([description_vector], columns=feature_names)
    counterfactual = exp.generate_counterfactuals(query_instance, total_CFs=1, desired_class="opposite")

    general_explanation = (
        f"Counterfactuals show how changing certain words would alter the recommendation for '{book_title}'. "
    )

    return {
        "title": book_title,
        "general_explanation": general_explanation,
        "counterfactual_output": counterfactual.cf_examples_list[0].final_cfs_df.columns.tolist()
    }
