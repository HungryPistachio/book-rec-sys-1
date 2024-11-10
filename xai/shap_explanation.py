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

# Ensure images directory exists for storing SHAP plots
os.makedirs("images", exist_ok=True)

# Create a new instance of the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

def get_shap_explanation(recommendations):
    explanations = []
    explainer = shap.Explainer(loaded_model)

    for recommendation in recommendations:
        title = recommendation["title"]
        description_vector = np.array(recommendation["description_vector"]).reshape(1, -1)

        # Get the feature names (i.e., the words in the description) from the TF-IDF vectorizer
        feature_names = vectorizer.get_feature_names_out()

        # Generate SHAP values using the explainer
        try:
            shap_values = explainer(description_vector, feature_names=feature_names)
            logging.info(f"Generated SHAP values for '{title}'")

            # Extract the base value, SHAP values, and feature names for plotting
            base_value = shap_values[0].base_values[0] if isinstance(shap_values[0].base_values, np.ndarray) else shap_values[0].base_values
            values = shap_values[0].values[0] if isinstance(shap_values[0].values, np.ndarray) else shap_values[0].values
            feature_names = shap_values[0].feature_names

            # Select top 10 influential features
            top_indices = np.argsort(np.abs(values))[::-1][:10]
            top_values = values[top_indices]
            top_feature_names = [feature_names[i] for i in top_indices]

            # Generate a unique filename for each SHAP plot
            image_filename = f"shap_plot_{uuid.uuid4()}.png"
            image_path = os.path.join("images", image_filename)

            # Create and save the SHAP waterfall plot
            shap.waterfall_plot(shap.Explanation(base_values=base_value, values=top_values, feature_names=top_feature_names), show=False)
            plt.savefig(image_path, bbox_inches='tight')
            plt.close()

            # Append image path in the response
            explanations.append({
                "title": title,
                "image_url": f"/images/{image_filename}"
            })
        except Exception as e:
            logging.error(f"Failed to generate SHAP explanation for '{title}': {e}")
            continue

    # Wrap explanations in a dictionary and return as JSON
    return json.dumps({"explanations": explanations})
