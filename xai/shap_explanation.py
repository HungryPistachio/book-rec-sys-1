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

        logging.info(f"SHAP values for '{title}' generated.")

        try:
            # Use a static base_value as a safeguard
            base_value = 0.0
            values = np.array(shap_values[0].values).flatten()

            # Cap the SHAP values if they are excessively large
            values = np.clip(values, -3, 3)  # Lower cap for values

            # Retrieve feature names from vectorizer
            feature_names = tfidf_vectorizer.get_feature_names_out()

            # Limit to the top 1 or 2 features for even smaller plots
            top_indices = np.argsort(np.abs(values))[::-1][:2]
            top_values = values[top_indices]
            top_feature_names = [feature_names[idx] for idx in top_indices]

            # Generate a unique filename for each explanation image
            image_filename = f"shap_plot_{uuid.uuid4()}.png"
            image_path = os.path.join("images", image_filename)

            # Set a smaller figure size and lower DPI for manageable plot dimensions
            fig, ax = plt.subplots(figsize=(5, 3))  
            # Create the SHAP waterfall plot for the top 2 features
            shap.waterfall_plot(
                shap.Explanation(
                    base_values=base_value,
                    values=top_values,
                    feature_names=top_feature_names
                ),
                show=False
            )

            plt.tight_layout()
            plt.savefig(image_path, bbox_inches='tight', dpi=9, format='png')
            plt.close()
            logging.info(f"Image saved at path: {image_path}")
            logging.info(f"Checking existence of image file: {os.path.exists(image_path)} at path: {image_path}")

            explanations.append({
                "title": title,
                "image_url": f"/images/{image_filename}"
            })
        except Exception as e:
            logging.error(f"Failed to generate SHAP plot for '{title}': {e}")

    return json.dumps({"explanations": explanations})
