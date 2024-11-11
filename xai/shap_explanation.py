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

    # Initialize a TF-IDF vectorizer with a fixed max_features to match model input size
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=1471)
    tfidf_vectors = tfidf_vectorizer.fit_transform(combined_texts).toarray()
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Use Kernel SHAP with a fixed-size feature vector
    explainer = shap.KernelExplainer(loaded_model.predict, tfidf_vectors)

    for i, recommendation in enumerate(recommendations):
        title = recommendation.get("title", f"Recommendation {i + 1}")
        description_vector = tfidf_vectors[i]

        # Generate SHAP values for each recommendation using Kernel SHAP
        shap_values = explainer.shap_values(description_vector, nsamples=100)

        logging.info(f"SHAP values for '{title}' generated.")

        try:
            # Filter non-zero SHAP values and their feature names for clarity
            filtered_indices = [idx for idx, val in enumerate(shap_values[0]) if abs(val) > 1e-3]
            top_feature_names = [feature_names[idx] for idx in filtered_indices]
            top_values = [shap_values[0][idx] for idx in filtered_indices]

            # Plot only significant SHAP values as a bar chart
            image_filename = f"shap_plot_{uuid.uuid4()}.png"
            image_path = os.path.join("images", image_filename)

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(top_feature_names, top_values)
            ax.set_xlabel("SHAP Value Impact")
            ax.set_title(f"Top SHAP Features for '{title}'")
            plt.tight_layout()
            plt.savefig(image_path, bbox_inches='tight', dpi=10, format='png')
            plt.close()

            explanations.append({
                "title": title,
                "image_url": f"/images/{image_filename}"
            })
            logging.info(f"Image saved at path: {image_path}")

        except Exception as e:
            logging.error(f"Failed to generate SHAP plot for '{title}': {e}")

    return json.dumps({"explanations": explanations})
