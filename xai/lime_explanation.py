import json
import logging
import lime
from lime.lime_text import LimeTextExplainer
import numpy as np

logging.basicConfig(level=logging.INFO)

def get_lime_explanation(description_vector, feature_names):
    logging.info("Starting LIME explanation.")
    explainer = LimeTextExplainer(class_names=['Book'])

    # Convert description_vector and feature_names to interpretable format for LIME
    logging.debug(f"Description Vector: {description_vector}")
    logging.debug(f"Feature Names: {feature_names}")

    explanation = explainer.explain_instance(
        ''.join(feature_names),  # Using feature names as text input
        lambda x: np.array([description_vector]),  # Using vector as direct output
        num_features=len(feature_names)
    )

    explanation_output = explanation.as_list()
    logging.info("LIME explanation generated.")
    print("LIME explanations identify the feature contributions to the recommendation.")

    response = {
        "general_explanation": "LIME explanation for the book recommendation.",
        "explanation_output": explanation_output
    }
    return json.dumps(response)
