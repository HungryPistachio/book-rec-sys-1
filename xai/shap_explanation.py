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
    explainer = shap.Explainer(loaded_model)

    for recommendation in recommendations:
        title = recommendation["title"]
        description_vector = np.array(recommendation["description_vector"]).reshape(1, -1)

        # Generate SHAP values using the explainer
        shap_values = explainer(description_vector)

        # Debugging information for SHAP values
        logging.info(f"Generated SHAP values for '{title}': {shap_values}")

        # Check and log each component of the SHAP values
        if not shap_values:
            logging.error(f"No SHAP values returned for '{title}'.")
            continue

        if not hasattr(shap_values[0], 'base_values') or shap_values[0].base_values is None:
            logging.error(f"'base_values' is missing or None for '{title}'.")
            continue
        if not hasattr(shap_values[0], 'values') or shap_values[0].values is None:
            logging.error(f"'values' is missing or None for '{title}'.")
            continue
        if not hasattr(shap_values[0], 'feature_names') or shap_values[0].feature_names is None:
            logging.error(f"'feature_names' is missing or None for '{title}'.")
            continue

        # Extract valid data from the SHAP explanation
        base_value = shap_values[0].base_values
        values = shap_values[0].values
        feature_names = shap_values[0].feature_names

        # Select the top 10 features by absolute SHAP value
        top_indices = np.argsort(np.abs(values))[::-1][:10]
        top_base_value = base_value if isinstance(base_value, (float, int)) else base_value[0]
        top_values = values[top_indices]
        top_feature_names = [feature_names[i] for i in top_indices]

        # Generate a unique filename for each explanation image
        image_filename = f"shap_plot_{uuid.uuid4()}.png"
        image_path = os.path.join("images", image_filename)

        try:
            # Create and save the SHAP waterfall plot for the top 10 features
            shap.waterfall_plot(shap.Explanation(base_values=top_base_value, values=top_values, feature_names=top_feature_names), show=False)
            plt.savefig(image_path, bbox_inches='tight')
            plt.close()

            # Append image path in response
            explanations.append({
                "title": title,
                "image_url": f"/images/{image_filename}"
            })
        except Exception as e:
            logging.error(f"Failed to generate SHAP plot for '{title}': {e}")

    return json.dumps(explanations)
