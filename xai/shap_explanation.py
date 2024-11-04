import json
import logging
import numpy as np
import shap

logging.basicConfig(level=logging.INFO)

def get_shap_explanation(recommendations):
    logging.info("Starting SHAP explanation generation.")
    explanation_outputs = []

    # Ensure SHAP can run without a full ML model by creating a mock explainer
    for rec in recommendations:
        try:
            description_vector = rec['description_vector']
            feature_names = rec['feature_names']

            # Verify the input structure
            if not description_vector or not feature_names:
                raise ValueError("Recommendation is missing required fields.")

            explainer = shap.Explainer(lambda x: x)  # Use identity function for demonstration

            # Run explanation
            shap_values = explainer(np.array([description_vector]))  # Single instance
            explanation_data = shap_values.values[0].tolist()

            # Pair feature names with SHAP values
            explanation_output = list(zip(feature_names, explanation_data))
            explanation_outputs.append({
                "title": rec["title"],
                "general_explanation": "SHAP explanation for the book recommendation.",
                "explanation_output": explanation_output
            })
            logging.info(f"SHAP explanation generated for '{rec['title']}'.")

        except Exception as e:
            logging.error(f"Error generating SHAP explanation for '{rec['title']}': {e}")
            explanation_outputs.append({
                "title": rec.get("title", "Unknown Title"),
                "general_explanation": "Error generating SHAP explanation.",
                "explanation_output": []
            })

    return json.dumps(explanation_outputs)
