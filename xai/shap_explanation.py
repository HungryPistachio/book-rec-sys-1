import shap
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd

def get_shap_explanation(book_title, description_vector, filtered_tfidf_matrix, feature_names, all_books):
    # Train a simple logistic regression model for explanations
    model = LogisticRegression()
    labels = [1 if "mystery" in desc else 0 for desc in all_books]  # Example binary labels
    model.fit(filtered_tfidf_matrix, labels)

    # Initialize the SHAP explainer for linear models
    explainer = shap.LinearExplainer(model, filtered_tfidf_matrix, feature_perturbation="independent")

    # Get SHAP values for the description vector
    shap_values = explainer.shap_values(np.array([description_vector]))

    # Pair features with their corresponding SHAP values
    explanation_list = list(zip(feature_names, shap_values[0]))

    # General explanation
    general_explanation = (
        f"SHAP values indicate the importance of each word in predicting recommendations. "
        f"A higher SHAP value means a stronger influence on the recommendation for '{book_title}'."
    )

    return {
        "title": book_title,
        "general_explanation": general_explanation,
        "shap_output": explanation_list  # List of (feature, SHAP value) pairs
    }
