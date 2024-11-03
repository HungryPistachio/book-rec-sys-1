import json
import logging
import numpy as np
from lime.lime_text import LimeTextExplainer

logging.basicConfig(level=logging.INFO)

def get_lime_explanation(description_vector, feature_names):
    logging.info("Starting LIME explanation generation.")
    
    # Ensure description_vector and feature_names are properly structured
    if not isinstance(description_vector, list) or not isinstance(feature_names, list):
        logging.error("Invalid input format for LIME explanation.")
        return json.dumps({"error": "Invalid input format"})
    
    try:
        explainer = LimeTextExplainer(class_names=['Book'])
        
        # Create input text and define explanation function
        input_text = ' '.join(feature_names)  # Mock input text for LIME
        explanation = explainer.explain_instance(
            input_text,
            lambda x: np.array([description_vector]),  # Model approximation
            num_features=len(feature_names)
        )
        
        explanation_output = explanation.as_list()
        logging.info("LIME explanation generated successfully.")
        print("LIME explanations identify the feature contributions to the recommendation.")
        
        response = {
            "general_explanation": "LIME explanation for the book recommendation.",
            "explanation_output": explanation_output
        }
        return json.dumps(response)

    except Exception as e:
        logging.error(f"Error generating LIME explanation: {e}")
        return json.dumps({"error": "LIME explanation generation failed"})
