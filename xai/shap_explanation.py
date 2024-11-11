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

    # Combine title, authors, and description for vectorization context
    combined_texts = [
        f"{rec.get('title', '')} {', '.join(rec.get('authors', ['']))} {rec.get('description', '')}"
        for rec in recommendations
    ]

    # Initialize a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_vectors = tfidf_vectorizer.fit_transform(combined_texts)

    # Initialize a Kernel SHAP explainer
    explainer = shap.KernelExplainer(lambda x: loaded_model.predict(x), tfidf_vectors.toarray())

    for i, recommendation in enumerate(recommendations):
        title = recommendation.get("title", f"Recommendation {i + 1}")

        # Re-vectorize each book individually for SHAP
        book_text = f"{recommendation.get('title', '')} {', '.join(recommendation.get('authors', ['']))} {recommendation.get('description', '')}"
        description_vector = tfidf_vectorizer.transform([book_text]).toarray()

        # Log the description vector to confirm differentiation
        logging.info(f"Description vector for '{title}': {description_vector}")

        # Generate SHAP values for the specific vector
        shap_values = explainer.shap_values(description_vector)
        logging.info(f"SHAP values for '{title}' generated.")

        try:
            # Use a static base_value as a safeguard
            base_value = 0.0
            values = np.round(np.clip(np.array(shap_values[0]).flatten(), -3, 3), 2)

            # Retrieve feature names and filter out features with 0 SHAP values
            feature_names = tfidf_vectorizer.get_feature_names_out()
            non_zero_indices = [idx for idx, val in enumerate(values) if val != 0]
            top_indices = sorted(non_zero_indices, key=lambda idx: abs(values[idx]), reverse=True)[:5]
            top_values = values[top_indices]
            top_feature_names = [
                (name[:10] + "...") if len(name) > 10 else name for name in [feature_names[idx] for idx in top_indices]
            ]

            # Generate a unique filename for each explanation image
            image_filename = f"shap_plot_{uuid.uuid4()}.png"
            image_path = os.path.join("images", image_filename)

            # Create a bar plot for the top features with non-zero SHAP values
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(top_feature_names, top_values, color=['green' if val > 0 else 'red' for val in top_values])
            ax.set_xlabel("SHAP Value")
            ax.set_title(f"Top SHAP Features for {title}")
            plt.tight_layout()

            # Save the plot with minimal DPI
            plt.savefig(image_path, bbox_inches='tight', dpi=10, format='png')
            plt.close()
            logging.info(f"Image saved at path: {image_path}")

            explanations.append({
                "title": title,
                "image_url": f"/images/{image_filename}"
            })
        except Exception as e:
            logging.error(f"Failed to generate SHAP plot for '{title}': {e}")

    return json.dumps({"explanations": explanations})
