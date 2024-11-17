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
                Simulates predictions by mapping perturbed texts to feature significance.
                """
                # Simulate predictions for each perturbed text
                predictions = []
                for text in texts:
                    # Calculate relevance scores by matching features in perturbed text
                    scores = [vectorized_descriptions[feature_names.index(word)] 
                              if word in feature_names else 0 
                              for word in text.split()]
                    
                    # Aggregate the scores (e.g., mean or sum)
                    predictions.append([sum(scores)])
                
                # Convert predictions to a 2D array
                predictions = np.array(predictions)
                logging.info(f"Mock predictions shape: {predictions.shape}, Example: {predictions[:1]}")
                return predictions

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
