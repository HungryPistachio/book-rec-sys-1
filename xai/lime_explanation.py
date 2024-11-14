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
            description_vector = rec.get("vectorized_descriptions", [])
            feature_names = rec.get("feature_names", [])
            input_text = ' '.join(feature_names[:85000])  # Limit input text for LIME
            logging.info(f"Received input_text: {input_text}")
            logging.info(f"Received description_vector: {description_vector}")

            # Generate explanation using LIME with increased num_features and num_samples
            explanation = explainer.explain_instance(
                input_text,
                lambda x: np.array([description_vector] * len(x)),
                num_features=len(feature_names),  # Use more features for wider coverage
                num_samples=1000  # Adjust for potentially finer granularity
            )

            # Get the top 10 most influential features, with an inclusion threshold for low-weight features
            inclusion_threshold = 0.05  # Set threshold for minimum weight to include in output
            explanation_output = [
                (word, weight) for word, weight in explanation.as_list()
                if abs(weight) >= inclusion_threshold
            ][:5]  # Take up to 10 if available

            # If no features meet the threshold, include the 10 highest regardless of weight
            if not explanation_output:
                explanation_output = sorted(
                    explanation.as_list(), key=lambda x: abs(x[1]), reverse=True
                )[:5]

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
