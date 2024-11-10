import shap
import joblib
import json
import matplotlib.pyplot as plt
import uuid
import os
import numpy as np

# Load the saved model directly
loaded_model = joblib.load("model/trained_model.joblib")

def get_shap_explanation(recommendations):
    explanations = []
    explainer = shap.Explainer(loaded_model)

    for recommendation in recommendations:
        title = recommendation["title"]
        # Convert description vector to a 2D array format for SHAP
        description_vector = np.array(recommendation["description_vector"]).reshape(1, -1)

        # Generate SHAP values using the explainer, getting a full explanation object
        shap_values = explainer(description_vector)

        # Generate a unique filename for each explanation image
        image_filename = f"shap_plot_{uuid.uuid4()}.png"
        image_path = os.path.join("images", image_filename)

        # Select the first output for base values and SHAP values, assuming single output
        base_value = shap_values.base_values[0] if hasattr(shap_values, "base_values") else shap_values[0].base_values[0]
        values = shap_values.values[0] if hasattr(shap_values, "values") else shap_values[0].values[0]

        # Create and save the SHAP waterfall plot
        shap.waterfall_plot(shap.Explanation(base_values=base_value, values=values, feature_names=shap_values.feature_names), show=False)
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()

        # Append image path in response
        explanations.append({
            "title": title,
            "image_url": f"/images/{image_filename}"
        })

    return json.dumps(explanations)
