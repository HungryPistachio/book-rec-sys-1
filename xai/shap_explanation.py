import shap
import joblib
import json
import matplotlib.pyplot as plt
import uuid
import os
import sklearn

# Step 1: Load the original model (saved in 1.5.2) and save it as a dictionary
original_model = joblib.load("model/trained_model.joblib")  # Load with a unique name
model_data = {"model": original_model, "scikit_version": sklearn.__version__}

# Save the wrapped model as a new file
joblib.dump(model_data, "model/generic_model_data.joblib")

# Step 2: Load the saved dictionary (saved in 1.0.5 environment) and extract the model for SHAP processing
loaded_model_data = joblib.load("model/generic_model_data.joblib")
loaded_model = loaded_model_data["model"]  # Access the model with a unique name
print("Loaded model version:", loaded_model_data["scikit_version"])

# Define SHAP explanation function
def get_shap_explanation(recommendations):
    explanations = []
    explainer = shap.Explainer(loaded_model)  # Use `loaded_model` here

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
