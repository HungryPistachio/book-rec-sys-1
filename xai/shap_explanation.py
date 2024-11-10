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

        # Create and save the SHAP waterfall plot using shap_values[0]
        shap.plots.waterfall(shap_values[0], show=False)
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()

        # Append image path in response
        explanations.append({
            "title": title,
            "image_url": f"/images/{image_filename}"
        })

    return json.dumps(explanations)
