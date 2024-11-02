import shap
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd

def get_shap_explanation(book_title, description_vector, tfidf_matrix, feature_names):
    # Train a simple logistic regression model for explanations
    model = LogisticRegression()
    labels = np.random.randint(0, 2, size=(tfidf_matrix.shape[0],))  # Random binary labels as example
    model.fit(tfidf_matrix, labels)

    # Initialize the SHAP explainer for linear models
    explainer = shap.LinearExplainer(model, tfidf_matrix, feature_perturbation="independent")

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
