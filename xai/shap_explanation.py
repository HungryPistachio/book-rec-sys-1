import shap
import joblib
import json
import matplotlib.pyplot as plt
import uuid
import os

# Load the saved model directly (assumes scikit-learn version 1.0.2 is compatible)
loaded_model = joblib.load("model/trained_model.joblib")

# Define SHAP explanation function
def get_shap_explanation(recommendations):
    explanations = []
    explainer = shap.Explainer(loaded_model)  # Use the directly loaded model

    for recommendation in recommendations:
        title = recommendation["title"]
        description_vector = recommendation["description_vector"]

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

# Example call to the function
# recommendations = [{"title": "Book Title", "description_vector": some_vector}]
# print(get_shap_explanation(recommendations))
