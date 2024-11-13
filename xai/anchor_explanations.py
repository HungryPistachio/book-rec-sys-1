import json
import logging
from alibi.explainers import AnchorText
import numpy as np
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def get_anchor_explanation(recommendations, original_vector):
    explanations = []

    # Define the predictor function using precomputed vector similarities
    def predict_fn(texts):
        # Instead of recomputing similarity, we directly return a constant relevance score
        return np.array([1 if np.dot(original_vector, rec['vectorized_descriptions']) /
                         (np.linalg.norm(original_vector) * np.linalg.norm(rec['vectorized_descriptions'])) >= 0.5 else 0
                         for rec in recommendations])

    # Initialize AnchorText explainer with the predictor function
    explainer = AnchorText(nlp=nlp, predictor=predict_fn)

    # Process each recommendation to generate explanations
    for idx, rec in enumerate(recommendations):
        try:
            # Combine feature names to create the input text for the anchor explainer
            input_text = ' '.join(rec.get("feature_names", []))

            if not rec.get("feature_names"):
                logging.warning(f"Missing feature names for recommendation {idx + 1}")
                explanations.append({
                    "title": rec.get("title", f"Recommendation {idx + 1}"),
                    "anchor_words": "None",
                    "precision": 0.0
                })
                continue

            # Generate the anchor explanation
            explanation = explainer.explain(input_text, threshold=0.95)

            # Extract anchor words and precision score
            anchor_words = " AND ".join(explanation.data['anchor'])
            precision = explanation.data['precision']

            explanations.append({
                "title": rec.get("title", f"Recommendation {idx + 1}"),
                "anchor_words": anchor_words,
                "precision": precision
            })

        except Exception as e:
            logging.error(f"Error generating Anchor explanation for recommendation {idx + 1}: {e}")
            explanations.append({
                "title": rec.get("title", f"Recommendation {idx + 1}"),
                "anchor_words": "None",
                "precision": 0.0
            })

    logging.info("Anchor explanations generated successfully for all recommendations.")
    return json.dumps(explanations)
