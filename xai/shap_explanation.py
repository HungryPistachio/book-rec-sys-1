import shap
import joblib
import json
import matplotlib.pyplot as plt
import uuid
import os
import sklearn


# Load the model saved with scikit-learn 1.5.2
model = joblib.load("model/trained_model.joblib")

# Wrap in a dictionary
model_data = {"model": model, "scikit_version": sklearn.__version__}

# Save as a new file
joblib.dump(model_data, "generic_model_data.joblib")

model_new = f"(model/generic_model_data.joblib)"

def get_shap_explanation(recommendations):  # Expect recommendations list directly
    explanations = []
    explainer = shap.Explainer(model_new)  # Assuming `model` is already loaded

    for recommendation in recommendations:
        title = recommendation["title"]
        description_vector = recommendation["description_vector"]

        shap_values = explainer.shap_values(description_vector)

        # Generate a unique filename
        image_filename = f"shap_plot_{uuid.uuid4()}.png"
        image_path = f"images/{image_filename}"

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
