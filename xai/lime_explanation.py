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
            description_vector = rec.get("vectorized_descriptions", [])
            feature_names = rec.get("feature_names", [])

            # Normalize description vector to prevent skew
            description_vector = np.array(description_vector)
            if description_vector.max() > 0:
                description_vector = description_vector / description_vector.max()

            # Refine input text using the most relevant features
            input_text = ' '.join(
                [feature for _, feature in sorted(
                    zip(description_vector, feature_names), 
                    reverse=True
                )[:min(len(feature_names), 500)]]  # Limit to top 500 features
            )

            logging.info(f"Input text for LIME explanation: {input_text[:200]}...")  # Log snippet of input text

            # Generate explanation using LIME
            explanation = explainer.explain_instance(
                input_text,
                lambda x: np.array([description_vector] * len(x)),
                num_features=min(len(feature_names), 100),  # Limit to top 100 features
                num_samples=5000  # Focused sampling for better perturbations
            )

            # Extract explanation details
            inclusion_threshold = 0.005  # Include features with at least 0.5% weight
            explanation_output = [
                (word, weight) for word, weight in explanation.as_list()
                if abs(weight) >= inclusion_threshold
            ][:10]  # Take top 10 features

            if not explanation_output:  # Fallback: Include features with highest weights
                explanation_output = sorted(
                    explanation.as_list(), key=lambda x: abs(x[1]), reverse=True
                )[:10]

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
