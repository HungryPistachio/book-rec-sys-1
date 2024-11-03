import json
import logging
import shap
import numpy as np

logging.basicConfig(level=logging.INFO)

def get_shap_explanation(description_vector, feature_names):
    logging.info("Starting SHAP explanation.")

    # For SHAP, we're directly interpreting the description vector
    explainer = shap.Explainer(lambda x: np.array([description_vector]))
    shap_values = explainer(np.array([description_vector]))

    # Extract feature contributions for SHAP
    explanation_output = list(zip(feature_names, shap_values[0].values))
    logging.info("SHAP explanation generated.")
    print("SHAP explanations provide feature impacts on the recommendation.")

    response = {
        "general_explanation": "SHAP explanation for the book recommendation.",
        "explanation_output": explanation_output
    }
    return json.dumps(response)
