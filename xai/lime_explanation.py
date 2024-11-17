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
            # Extract features from the frontend recommendation
            feature_names = rec.get("feature_names", [])
            vectorized_descriptions = rec.get("vectorized_descriptions", [])

            if not feature_names or not vectorized_descriptions:
                raise ValueError("Feature names or vectorized descriptions missing.")

            # Use the significant features as input text
            input_text = ' '.join(feature_names)
            logging.info(f"Input text for LIME explanation: {input_text}")

            # Define a mock prediction function
            def mock_predict(texts):
                """
                Simulates predictions for LIME by using feature importance weights.
                Generates probabilities for the single class 'Book'.
                """
                probabilities = []
                for text in texts:
                    # Calculate the relevance score by summing weights of matched features
                    scores = [vectorized_descriptions[feature_names.index(word)] 
                              if word in feature_names else 0 
                              for word in text.split()]
                    relevance_score = sum(scores)

                    # Convert the relevance score to a probability (normalized between 0 and 1)
                    probability = np.clip(relevance_score / max(vectorized_descriptions, default=1), 0, 1)
                    probabilities.append([probability])

                # Return a 2D array where each row corresponds to a sample
                probabilities = np.array(probabilities)
                logging.info(f"Mock predictions shape: {probabilities.shape}, Example: {probabilities[:1]}")
                return probabilities

            # Generate LIME explanation
            explanation = explainer.explain_instance(
                input_text,
                mock_predict,
                num_features=min(len(feature_names), 20),  # Limit to top 20 features for better interpretability
                num_samples=5000  # Sufficient sample size for meaningful results
            )

            # Extract explanation details
            explanation_output = explanation.as_list()

            # Log and append explanation
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
