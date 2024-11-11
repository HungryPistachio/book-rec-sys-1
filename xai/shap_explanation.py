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
    
    # Configure TF-IDF vectorizer without a pre-set vocabulary
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=20)  # Limit to 20 keywords per book
    
    # Combine titles, authors, and descriptions dynamically
    combined_texts = [
        f"{rec.get('title', '')} {', '.join(rec.get('authors', ['']))} {rec.get('description', '')}"
        for rec in recommendations
    ]
    
    # Generate TF-IDF vectors for each description, focusing on top 20 features per book
    tfidf_vectors = tfidf_vectorizer.fit_transform(combined_texts).toarray()
    
    # Adjust vectors to meet the model's required input size, say 1471 features, by padding or truncating
    required_features = 1471
    vectors = [np.pad(vec, (0, max(0, required_features - len(vec))), 'constant')[:required_features]
               for vec in tfidf_vectors]
    
    # Initialize SHAP explainer with the model’s predict function
    explainer = shap.Explainer(loaded_model.predict, np.array(vectors))

    for i, recommendation in enumerate(recommendations):
        title = recommendation.get("title", f"Recommendation {i + 1}")
        vector = vectors[i].reshape(1, -1)  # Reshape for SHAP input

        # Generate SHAP values
        shap_values = explainer(vector)
        logging.info(f"SHAP values for '{title}' generated.")
        
        try:
            # Use a static base value as a safeguard
            base_value = 0.0
            values = np.array(shap_values[0].values).flatten()
            
            # Filter out zero or near-zero features for a cleaner plot
            non_zero_indices = np.nonzero(values)
            values = values[non_zero_indices]
            feature_names = np.array(tfidf_vectorizer.get_feature_names_out())[non_zero_indices]

            # Generate a unique filename for each explanation image
            image_filename = f"shap_plot_{uuid.uuid4()}.png"
            image_path = os.path.join("images", image_filename)

            # Plot using a bar chart for simplicity
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(feature_names, values)
            ax.set_title(f"SHAP Explanation for '{title}'")
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
