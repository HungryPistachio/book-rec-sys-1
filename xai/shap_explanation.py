import shap
import joblib
import json
import matplotlib.pyplot as plt
import uuid
import os
import numpy as np

# Load the saved model directly (assumes scikit-learn version 1.0.2 is compatible)
loaded_model = joblib.load("model/trained_model.joblib")

def get_shap_explanation(recommendations, model):
    explanations = []
    explainer = shap.Explainer(model)  # Use `model` here

    for recommendation in recommendations:
        title = recommendation["title"]
        description_vector = np.array(recommendation["description_vector"])  # Ensure it's a NumPy array

        # Generate SHAP values
        shap_values = explainer.shap_values(description_vector)

        # Generate a unique filename for each explanation image
        image_filename = f"shap_plot_{uuid.uuid4()}.png"
        image_path = os.path.join("images", image_filename)

        # Create and save the SHAP plot
        shap.plots.waterfall(shap_values, show=False)
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()

        # Append image path in response
        explanations.append({
            "title": title,
            "image_url": f"/images/{image_filename}"
        })

    return json.dumps(explanations)
