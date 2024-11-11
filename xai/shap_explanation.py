import shap
import json
import matplotlib.pyplot as plt
import uuid
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def get_shap_explanation(recommendations):
    explanations = []

    # Initialize SHAP explainer with a mock model; replace this if an actual model is used
    explainer = shap.Explainer(lambda x: np.random.rand(x.shape[0], 1))  # Replace this with actual model if available

    for i, recommendation in enumerate(recommendations):
        title = recommendation.get("title", f"Recommendation {i + 1}")
        description_vector = np.array(recommendation.get("description_vector", []))

        # Check if description_vector and feature_names exist and are valid
        if description_vector.size == 0 or not recommendation.get("feature_names"):
            logging.error(f"No valid description vector or feature names for '{title}'. Skipping.")
            continue

        feature_names = recommendation["feature_names"]

        # Generate SHAP values
        shap_values = explainer(description_vector.reshape(1, -1))

        logging.info(f"SHAP values for '{title}' generated.")

        try:
            # Use a static base_value as a safeguard
            base_value = 0.0
            values = shap_values[0].values.flatten()

            # Limit to the top 2 features for simplicity
            top_indices = np.argsort(np.abs(values))[::-1][:2]
            top_values = values[top_indices]
            top_feature_names = [feature_names[idx] for idx in top_indices]

            # Generate a unique filename for each explanation image
            image_filename = f"shap_plot_{uuid.uuid4()}.png"
            image_path = os.path.join("images", image_filename)

            # Set a smaller figure size and lower DPI for manageable plot dimensions
            fig, ax = plt.subplots(figsize=(3, 3))  # Adjust size as needed

            # Create the SHAP waterfall plot for the top features
            shap.waterfall_plot(
                shap.Explanation(
                    base_values=base_value,
                    values=top_values,
                    feature_names=top_feature_names
                ),
                show=False
            )

            # Save the plot
            plt.savefig(image_path, bbox_inches='tight', dpi=20, format='png')
            plt.close()

            explanations.append({
                "title": title,
                "image_url": f"images/{image_filename}"
            })
            logging.info(f"Image saved at path: {image_path}")
            
        except Exception as e:
            logging.error(f"Failed to generate SHAP plot for '{title}': {e}")

    return json.dumps({"explanations": explanations})
