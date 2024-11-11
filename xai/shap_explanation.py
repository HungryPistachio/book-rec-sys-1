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

            # Sanity check: ensure values are within a reasonable range
            if np.max(np.abs(values)) > 10:  # Adjust threshold as needed
                logging.warning(f"SHAP values for '{title}' are abnormally large; scaling down.")
                values = np.clip(values, -10, 10)

            # Retrieve feature names from vectorizer
            feature_names = tfidf_vectorizer.get_feature_names_out()

            # Get the top 10 influential features
            top_indices = np.argsort(np.abs(values))[::-1][:10]
            top_values = values[top_indices]
            top_feature_names = [feature_names[idx] for idx in top_indices]

            # Generate a unique filename for each explanation image
            image_filename = f"shap_plot_{uuid.uuid4()}.png"
            image_path = os.path.join("images", image_filename)

            # Create the figure with explicit size
            fig, ax = plt.subplots(figsize=(8, 6))

            # Add annotation text with transform
            ax.text(0.5, 1.05, f"SHAP Explanation for '{title}'", transform=ax.transAxes, 
                    ha='center', fontsize=12, fontweight='bold')

            # Create the SHAP waterfall plot for the top 10 features
            shap.waterfall_plot(
                shap.Explanation(
                    base_values=base_value,
                    values=top_values,
                    feature_names=top_feature_names
                ),
                show=False
            )
            
            # Save the plot with specified dpi and close the plot to free memory
            plt.savefig(image_path, bbox_inches='tight', dpi=300, format='png')
            plt.close()

            explanations.append({
                "title": title,
                "image_url": f"/images/{image_filename}"
            })
        except Exception as e:
            logging.error(f"Failed to generate SHAP plot for '{title}': {e}")

    return json.dumps({"explanations": explanations})
