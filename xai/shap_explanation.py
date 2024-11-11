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

# Load the saved model directly (assuming it's compatible with these inputs)
loaded_model = joblib.load("model/trained_model.joblib")

def get_shap_explanation(recommendations):
    explanations = []

    # Combine titles, authors, and descriptions for vectorization
    combined_texts = [
        f"{rec['title']} {', '.join(rec['authors'])} {rec['description']}"
        for rec in recommendations
    ]

    # Initialize a single TF-IDF vectorizer for all text
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_vectors = tfidf_vectorizer.fit_transform(combined_texts)  # Fit and transform in one step

    # Initialize SHAP explainer
    explainer = shap.Explainer(loaded_model)

    for i, recommendation in enumerate(recommendations):
        title = recommendation["title"]

        # Get the vectorized form of the combined text
        description_vector = tfidf_vectors[i].toarray()

        # Generate SHAP values using the explainer
        shap_values = explainer(description_vector)

        logging.info(f"Generated SHAP values for '{title}'")

        # Get feature names from vectorizer
        feature_names = tfidf_vectorizer.get_feature_names_out()
        base_value = shap_values[0].base_values[0] if isinstance(shap_values[0].base_values, np.ndarray) else shap_values[0].base_values
        values = shap_values[0].values[0] if isinstance(shap_values[0].values, np.ndarray) else shap_values[0].values

        # Get top 10 influential features
        top_indices = np.argsort(np.abs(values))[::-1][:10]
        top_values = values[top_indices]
        top_feature_names = [feature_names[i] for i in top_indices]

        # Create and save SHAP plot for the top 10 features
        image_filename = f"shap_plot_{uuid.uuid4()}.png"
        image_path = os.path.join("images", image_filename)
        
        try:
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
