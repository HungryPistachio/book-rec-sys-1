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

            # Normalize feature weights
            description_vector = np.array(description_vector)
            if max(description_vector) > 0:
                description_vector = description_vector / max(description_vector)

            # Focus on top 200 features
            top_features = sorted(
                zip(description_vector, feature_names),
                key=lambda x: x[0],
                reverse=True
            )[:200]
            input_text = ' '.join([feature for _, feature in top_features])

            logging.info(f"Input text for LIME explanation: {input_text[:200]}...")  # Log a snippet

            # Dynamic classifier for perturbation variability
            def dynamic_classifier(perturbations):
                predictions = []
                for text in perturbations:
                    overlap = sum([description_vector[feature_names.index(word)] if word in feature_names else 0 
                                   for word in text.split()])
                    predictions.append([overlap / len(text.split()) if len(text.split()) > 0 else 0])
                return np.array(predictions)

            # Generate explanation using LIME
            explanation = explainer.explain_instance(
                input_text,
                dynamic_classifier,
                num_features=100,  # Limit to top 100 features
                num_samples=3000  # Focused sample size
            )

            # Extract explanation details
            inclusion_threshold = 0.001  # Include minor contributions
            explanation_output = [
                (word, weight) for word, weight in explanation.as_list() 
                if abs(weight) >= inclusion_threshold
            ][:10]

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
