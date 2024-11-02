import lime
import lime.lime_text
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

def get_lime_explanation(book_title, book_description, all_books, tfidf_matrix, feature_names):
    # Train a simple logistic regression model for explanations
    model = LogisticRegression()
    labels = np.random.randint(0, 2, size=(tfidf_matrix.shape[0],))  # Random binary labels as example
    model.fit(tfidf_matrix, labels)

    # Define the LIME explainer using feature names
    explainer = lime.lime_text.LimeTextExplainer(class_names=["Recommendation"])

    # Function that predicts similarity between query instance and TF-IDF matrix
    def predict_fn(text_vector):
        text_tfidf = np.array(text_vector)
        similarities = np.dot(tfidf_matrix, text_tfidf.T)
        return similarities.mean(axis=0).reshape(-1, 1)

    # Generate explanation using LIME
    exp = explainer.explain_instance(
        pd.Series(description_vector, index=feature_names),
        predict_fn,
        num_features=6
    )

    # Extract explanation list
    explanation_list = exp.as_list()

    # General explanation
    general_explanation = (
        f"LIME identifies words that contribute to the similarity score, explaining why the book '{book_title}' "
        f"might be recommended based on the input."
    )

    return {
        "title": book_title,
        "general_explanation": general_explanation,
        "lime_output": explanation_list  # List of words and importance weights
    }
