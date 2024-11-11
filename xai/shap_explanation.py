import shap
import joblib
import json
import matplotlib.pyplot as plt
import uuid
import os
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load the saved model directly
loaded_model = joblib.load("model/trained_model.joblib")

def get_shap_explanation(recommendations):
    explanations = []

    # Debugging: Log recommendations to ensure descriptions exist
    logging.info(f"Received recommendations: {recommendations}")

    # Extract book descriptions to dynamically fit the TF-IDF vectorizer
    descriptions = [rec.get("description", "").strip() for rec in recommendations if rec.get("description")]
    
    # Check for empty descriptions after extraction
    if not descriptions:
        logging.error("No valid descriptions found in recommendations. Cannot proceed with SHAP explanation.")
        return json.dumps({"explanations": explanations})

    # Fit TF-IDF to descriptions
    try:
        tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        description_vectors = tfidf_vectorizer.fit_transform(descriptions)
        feature_names = tfidf_vectorizer.get_feature_names_out()
    except Exception as e:
        logging.error(f"Failed to fit TF-IDF vectorizer: {e}")
        return json.dumps({"explanations": explanations})

    # Initialize SHAP explainer
    explainer = shap.Explainer(loaded_model)

    for i, recommendation in enumerate(recommendations):
        title = recommendation.get("title", f"Recommendation {i + 1}")
        try:
            description_vector = description_vectors[i].toarray()

            # Generate SHAP values
            shap_values = explainer(description_vector)

            # Log generated SHAP values
            logging.info(f"Generated SHAP values for '{title}'")

            # Retrieve and process SHAP values
            if feature_names is not None:
                base_value = shap_values[0].base_values[0] if isinstance(shap_values[0].base_values, np.ndarray) else shap_values[0].base_values
                values = shap_values[0].values[0] if isinstance(shap_values[0].values, np.ndarray) else shap_values[0].values
                top_indices = np.argsort(np.abs(values))[::-1][:10]
                top_values = values[top_indices]
                top_feature_names = [feature_names[j] for j in top_indices]
            else:
                logging.error(f"'feature_names' is missing for '{title}'. Skipping this entry.")
                continue

            # Generate unique filename and plot
            image_filename = f"shap_plot_{uuid.uuid4()}.png"
            image_path = os.path.join("images", image_filename)

            # Create SHAP waterfall plot
            shap.waterfall_plot(shap.Explanation(base_values=base_value, values=top_values, feature_names=top_feature_names), show=False)
            plt.savefig(image_path, bbox_inches='tight')
            plt.close()

            # Add explanation info to results
            explanations.append({
                "title": t
