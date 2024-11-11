import shap
import joblib
import json
import matplotlib.pyplot as plt
import uuid
import os
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)

# Load the saved model directly
loaded_model = joblib.load("model/trained_model.joblib")

def get_shap_explanation(recommendations):
    explanations = []

    # Combine title, authors, and description
    combined_texts = [
        f"{rec.get('title', '')} {', '.join(rec.get('authors', ['']))} {rec.get('description', '')}"
        for rec in recommendations
    ]

    # Initialize a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_vectors = tfidf_vectorizer.fit_transform(combined_texts)

    # Initialize SHAP explainer with the loaded model
    explainer = shap.Explainer(loaded_model)

    for i, recommendation in enumerate(recommendations):
        title = recommendation.get("title", f"Recommendation {i + 1}")
        description_vector = tfidf_vectors[i].toarray()

        # Generate SHAP values
        shap_values = explainer(description_vector)

        try:
            # Ensure base_value is within a reasonable range
            base_value = shap_values[0].base_values
            if np.isscalar(base_value):
                base_value = np.array([base_value])
            if base_value.ndim == 0:
                base_value = np.expand_dims(base_value, 0)
            base_value = base_value[0]

            # Fallback to default if base_value is unrealistically large or small
            if abs(base_value) > 1000:
                logging.warning(f"Base value {base_value} is out of range; defaulting to 0.0")
                base_value = 0.0

            # Check and clip values within a reasonable range
            values = shap_values[0].values
            if np.isscalar(values):
                values = np.array([values])
            if values.ndim == 1:
                values = np.expand_dims(values, axis=0)
            values = values[0]

            # Clip values to prevent excessively large magnitudes
            values = np.clip(values, -1, 1)

            # Retrieve feature names from vectorizer
            feature_names = tfidf_vectorizer.get_feature_names_out()

            # Get the top 10 influential features
            top_indices = np.argsort(np.abs(values))[::-1][:10]
            top_values = values[top_indices]
            top_feature_names = [feature_names[idx] for idx in top_indices]

            # Generate a unique filename for each explanation image
            image_filename = f"shap_plot_{uuid.uuid4()}.png"
            image_path = os.path.join("images", image_filename)

            # Create and save the SHAP waterfall plot for the top 10 features
            shap.waterfall_plot(
                shap.Explanation(
                    base_values=base_value,
                    values=top_values,
                    feature_names=top_feature_names
                ),
                show=False
            )
            plt.savefig(image_path, bbox_inches='tight')
            plt.close()

            explanations.append({
                "title": title,
                "image_url": f"/images/{image_filename}"
            })
        except Exception as e:
            logging.error(f"Failed to generate SHAP plot for '{title}': {e}")

    return json.dumps({"explanations": explanations})
