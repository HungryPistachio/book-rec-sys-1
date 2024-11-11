import shap
import joblib
import json
import matplotlib.pyplot as plt
import uuid
import os
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)

# Load the saved model directly
loaded_model = joblib.load("model/trained_model.joblib")

# Initialize Sentence Transformer for dynamic embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings

def get_shap_explanation(recommendations):
    explanations = []

    # Combine title, authors, and description for each recommendation
    combined_texts = [
        f"{rec.get('title', '')} {', '.join(rec.get('authors', ['']))} {rec.get('description', '')}"
        for rec in recommendations
    ]

    # Generate sentence embeddings for the descriptions
    embeddings = embedding_model.encode(combined_texts)  # Produces a fixed 384-length vector for each text

    # Initialize Kernel SHAP explainer with the model's predict function
    explainer = shap.KernelExplainer(loaded_model.predict, embeddings)

    for i, recommendation in enumerate(recommendations):
        title = recommendation.get("title", f"Recommendation {i + 1}")
        embedding_vector = embeddings[i]

        # Generate SHAP values for the embedding vector
        shap_values = explainer.shap_values(embedding_vector, nsamples=100)

        logging.info(f"SHAP values for '{title}' generated.")

        try:
            # Filter and display only the most impactful features in the embedding
            significant_indices = np.argsort(np.abs(shap_values[0]))[::-1][:10]  # Top 10 features
            top_values = np.array(shap_values[0])[significant_indices]
            top_features = [f"Dim {idx}" for idx in significant_indices]  # Embedding dimensions

            # Generate a unique filename for each explanation image
            image_filename = f"shap_plot_{uuid.uuid4()}.png"
            image_path = os.path.join("images", image_filename)

            # Create a bar chart for top SHAP values
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(top_features, top_values)
            ax.set_xlabel("SHAP Value Impact")
            ax.set_title(f"Top SHAP Dimensions for '{title}'")
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
