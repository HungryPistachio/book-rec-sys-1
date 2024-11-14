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
            description_vector = rec.get("vectorized_description", [])
            feature_names = rec.get("feature_names", [])
            input_text = ' '.join(feature_names[:85000])  # Limit input text for LIME
            logging.info(f"Received input_text: {input_text}")
            logging.info(f"Received description_vector: {description_vector}")
            # Generate explanation using LIME
            explanation = explainer.explain_instance(
                input_text,
                lambda x: np.array([description_vector] * len(x)),
                num_features=min(len(feature_names), 85000)
            )

            # Get the top 10 most influential features without stop word filtering
            explanation_output = sorted(explanation.as_list(), key=lambda x: abs(x[1]), reverse=True)[:10]

            # Add to explanations list
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
