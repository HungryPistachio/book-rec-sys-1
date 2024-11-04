import json
import logging
import numpy as np
from lime.lime_text import LimeTextExplainer

logging.basicConfig(level=logging.INFO)

def get_lime_explanation(recommendations):
    logging.info("Starting LIME explanation generation for multiple recommendations.")

    # Initialize LIME explainer
    explainer = LimeTextExplainer(class_names=['Book'])
    explanations = []

    # Loop through each recommendation
    for idx, rec in enumerate(recommendations):
        try:
            description_vector = rec.get("description_vector", [])
            feature_names = rec.get("feature_names", [])
            input_text = ' '.join(feature_names[:500])  # Limit input text for LIME

            # Generate explanation using LIME
            explanation = explainer.explain_instance(
                input_text,
                lambda x: np.array([description_vector] * len(x)),
                num_features=min(len(feature_names), 100)
            )

            # Filter for significant features only
            explanation_output = [(word, weight) for word, weight in explanation.as_list() if weight > 0.0]
            explanations.append({
                "title": rec.get("title", f"Recommendation {idx + 1}"),
                "general_explanation": "LIME explanation for the book recommendation.",
                "explanation_output": explanation_output
            })

        except Exception as e:
            logging.error(f"Error generating LIME explanation for recommendation {idx + 1}: {e}")
            explanations.append({
                "title": rec.get("title", f"Recommendation {idx + 1}"),
                "general_explanation": "Failed to generate explanation.",
                "explanation_output": []
            })

    logging.info("LIME explanations generated successfully for all recommendations.")
    return json.dumps(explanations)
