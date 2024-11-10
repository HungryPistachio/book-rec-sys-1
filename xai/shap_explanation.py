import shap
import joblib
import json
import matplotlib.pyplot as plt
import uuid
import os
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model directly
loaded_model = joblib.load("model/trained_model.joblib")

def get_shap_explanation(recommendations):
    explanations = []
    
    # Extract book descriptions to dynamically fit the TF-IDF vectorizer
    descriptions = [rec["description"] for rec in recommendations]
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_vectorizer.fit(descriptions)  # Fit to current descriptions
    description_vectors = tfidf_vectorizer.transform(descriptions)
    
    # Initialize SHAP explainer with the loaded model
    explainer = shap.Explainer(loaded_model)
    
    for i, recommendation in enumerate(recommendations):
        title = recommendation["title"]
        description_vector = description_vectors[i].toarray()

        # Generate SHAP values using the explainer
        shap_values = explainer(description_vector)

        # Log generated SHAP values
        logging.info(f"Generated SHAP values for '{title}'")

        # Ensure 'feature_names' is set to vectorizer's feature names
        if tfidf_vectorizer.get_feature_names_out() is not None:
            feature_names = tfidf_vectorizer.get_feature_names_out()
        else:
            logging.error(f"'feature_names' is missing or None for '{title}'. Skipping this entry.")
            continue

        # Get the top 10 features
        base_value = shap_values[0].base_values[0] if isinstance(shap_values[0].base_values, np.ndarray) else shap_values[0].base_values
        values = shap_values[0].values[0] if isinstance(shap_values[0].values, np.ndarray) else shap_values[0].values
        top_indices = np.argsort(np.abs(values))[::-1][:10]
        top_base_value = base_value
        top_values = values[top_indices]
        top_feature_names = [feature_names[i] for i in top_indices]

        # Generate a unique filename for each explanation image
        image_filename = f"shap_plot_{uuid.uuid4()}.png"
        image_path = os.path.join("images", image_filename)

        try:
            # Create and save the SHAP waterfall plot for the top 10 features
            shap.waterfall_plot(shap.Explanation(base_values=top_base_value, values=top_values, feature_names=top_feature_names), show=False)
            plt.savefig(image_path, bbox_inches='tight')
            plt.close()

            # Append image path in response
            explanations.append({
                "title": title,
                "image_url": f"/images/{image_filename}"
            })
        except Exception as e:
            logging.error(f"Failed to generate SHAP plot for '{title}': {e}")

    # Wrap explanations in a dictionary to match expected format
    return json.dumps({"explanations": explanations})
