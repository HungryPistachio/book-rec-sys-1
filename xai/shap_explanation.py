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

# Load the trained model
loaded_model = joblib.load("model/trained_model.joblib")

def get_shap_explanation(recommendations):
    explanations = []

    # Combine titles, authors, and descriptions into a single string for each book
    combined_texts = [
        f"{rec.get('title', '')} {', '.join(rec.get('authors', ['']))} {rec.get('description', '')}"
        for rec in recommendations
    ]
    
    # Step 1: Fit the TF-IDF vectorizer on all combined texts to create a fixed vocabulary of 1471 features
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=1471)
    tfidf_vectorizer.fit(combined_texts)  # Define the vocabulary based on all recommendations
    
    # Step 2: Transform each individual description using this fixed vocabulary
    tfidf_vectors = tfidf_vectorizer.transform(combined_texts).toarray()  # Ensures 1471 features for each entry

    # Initialize SHAP explainer with the model’s predict function
    explainer = shap.Explainer(loaded_model.predict, tfidf_vectors)

    for i, recommendation in enumerate(recommendations):
        title = recommendation.get("title", f"Recommendation {i + 1}")
        vector = tfidf_vectors[i].reshape(1, -1)  # Reshape for SHAP input

        # Generate SHAP values
        shap_values = explainer(vector)
        logging.info(f"SHAP values for '{title}' generated.")

        try:
            # Extract SHAP values and feature names, filtering out zero weights
            values = np.array(shap_values[0].values).flatten()
            non_zero_indices = values != 0
            values = values[non_zero_indices]
            feature_names = np.array(tfidf_vectorizer.get_feature_names_out())[non_zero_indices]

            # Generate a unique filename for each explanation image
            image_filename = f"shap_plot_{uuid.uuid4()}.png"
            image_path = os.path.join("images", image_filename)

            # Plot the SHAP values using a bar chart
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(feature_names, values)
            ax.set_title(f"SHAP Explanation for '{title}'")
            plt.tight_layout()
            plt.savefig(image_path, bbox_inches='tight', dpi=100, format='png')
            plt.close()
            
            explanations.append({
                "title": title,
                "image_url": f"/images/{image_filename}"
            })
            logging.info(f"Image saved at path: {image_path}")
        except Exception as e:
            logging.error(f"Failed to generate SHAP plot for '{title}': {e}")

    return json.dumps({"explanations": explanations})
