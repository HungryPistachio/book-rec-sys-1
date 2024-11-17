import json
import logging
import numpy as np
from lime.lime_text import LimeTextExplainer

logging.basicConfig(level=logging.INFO)

def get_lime_explanation(recommendations):
    logging.info("Starting LIME explanation generation for multiple recommendations.")

    explainer = LimeTextExplainer(class_names=['Book'])
    explanations = []

    for idx, rec in enumerate(recommendations):
        try:
            # Extract vectorized descriptions and feature names
            description_vector = rec.get("vectorized_descriptions", [])
            feature_names = rec.get("feature_names", [])

            # Normalize feature weights
            description_vector = np.array(description_vector)
            if description_vector.max() > 0:
                description_vector = description_vector / description_vector.max()

            # Select top 300 features based on weight
            top_features = sorted(
                zip(description_vector, feature_names),
                key=lambda x: x[0],
                reverse=True
            )[:300]
            input_text = ' '.join([feature for _, feature in top_features])

            logging.info(f"Input text for LIME explanation: {input_text[:200]}...")  # Log snippet of input text

            # Generate LIME explanation
            explanation = explainer.explain_instance(
                input_text,
                lambda x: np.array([description_vector] * len(x)),  # Dummy classifier
                num_features=100,  # Analyze top 100 features
                num_samples=3000  # Moderate perturbation count
            )

            # Extract explanation details
            explanation_output = explanation.as_list()[:10]  # Take top 10 features regardless of weight

            logging.info(f"LIME explanation generated: {explanation_output}")

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
