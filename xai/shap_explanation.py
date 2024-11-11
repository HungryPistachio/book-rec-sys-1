import shap
import joblib
import json
import matplotlib.pyplot as plt
import uuid
import os
import numpy as np
import logging

# Load the saved model directly
loaded_model = joblib.load("model/trained_model.joblib")

def get_shap_explanation(recommendations):
    explanations = []
    logging.info("Starting SHAP explanation generation for recommendations.")

    # Check if recommendations have valid 'description_vector' and 'feature_names'
    for rec in recommendations:
        if "description_vector" not in rec or "feature_names" not in rec:
            logging.error("Each recommendation must include 'description_vector' and 'feature_names'.")
            return json.dumps({"explanations": []})

    # Initialize SHAP explainer with the loaded model
    explainer = shap.Explainer(loaded_model)

    for i, recommendation in enumerate(recommendations):
        title = recommendation.get("title", f"Recommendation {i + 1}")
        description_vector = np.array(recommendation["description_vector"]).reshape(1, -1)
        feature_names = recommendation["feature_names"]

        try:
            # Generate SHAP values using the explainer
            shap_values = explainer(description_vector)

            # Ensure SHAP values were computed correctly
            logging.info(f"Generated SHAP values for '{title}'")

            # Extract top 10 influential features
            base_value = shap_values[0].base_values[0] if isinstance(shap_values[0].base_values, np.ndarray) else shap_values[0].base_values
            values = shap_values[0].values[0] if isinstance(shap_values[0].values, np.ndarray) else shap_values[0].values
            top_indices = np.argsort(np.abs(values))[::-1][:10]
            top_values = values[top_indices]
            top_feature_names = [feature_names[j] for j in top_indices]

            # Generate a unique filename for the SHAP plot
            image_filename = f"shap_plot_{uuid.uuid4()}.png"
            image_path = os.path.join("images", image_filename)

            # Create and save the SHAP waterfall plot
            shap.waterfall_plot(
                shap.Explanation(base_values=base_value, values=top_values, feature_names=top_feature_names),
                show=False
            )
            plt.savefig(image_path, bbox_inches='tight')
            plt.close()

            # Append explanation details
            explanations.append({
                "title": title,
                "image_url": f"/images/{image_filename}"
            })

        except Exception as e:
            logging.error(f"Failed to generate SHAP plot for '{title}': {e}")

    # Return explanations as JSON
    return json.dumps({"explanations": explanations})
