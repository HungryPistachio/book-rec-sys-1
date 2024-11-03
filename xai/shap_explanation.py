import json
import logging
import numpy as np
import shap

logging.basicConfig(level=logging.INFO)

def get_shap_explanation(description_vector, feature_names):
    logging.info("Starting SHAP explanation.")

    # Convert description_vector to a single sample array
    sample = np.array(description_vector).reshape(1, -1)
    
    try:
        # Initialize a SHAP explainer for text data using a masker
        masker = shap.maskers.Text()
        
        # Set up the explainer with a sample prediction function
        explainer = shap.Explainer(lambda x: np.array([sample[0]]), masker=masker)

        # Generate the explanation for the single instance
        explanation = explainer(sample)

        # Simplify explanation output
        shap_values = explanation.values[0].tolist()  # Convert explanation values to a list for JSON serialization
        feature_impact = list(zip(feature_names, shap_values))

        # Log the explanation details
        logging.info("SHAP explanation generated successfully.")

        # Create a response in JSON format
        response = {
            "general_explanation": "SHAP explanation for the book recommendation.",
            "explanation_output": feature_impact
        }
        return json.dumps(response)
    except Exception as e:
        logging.error(f"Error in SHAP explanation: {e}")
        return json.dumps({"error": "Failed to generate SHAP explanation"})

