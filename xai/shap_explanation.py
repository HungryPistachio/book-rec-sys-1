import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

def get_shap_explanation(book_title, book_description, all_descriptions):
    # Vectorize the descriptions
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_descriptions)

    # Create a simple logistic regression model to explain (could use any model)
    model = LogisticRegression()
    labels = np.random.randint(0, 2, size=(tfidf_matrix.shape[0],))
    model.fit(tfidf_matrix, labels)

    # Generate SHAP values
    explainer = shap.LinearExplainer(model, tfidf_matrix, feature_dependence="independent")
    shap_values = explainer(vectorizer.transform([book_description]))

    explanation_list = zip(vectorizer.get_feature_names_out(), shap_values.values[0])

    # General explanation
    general_explanation = (
        f"SHAP values explain the impact of each word on the model's recommendation. "
        f"A higher SHAP value means the word increases the likelihood of recommending the book '{book_title}'."
    )
    
    return {
        "title": book_title,
        "general_explanation": general_explanation, # General explanation once
        "shap_output": list(explanation_list)       # Raw SHAP output
    }
