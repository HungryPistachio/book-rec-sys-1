import json
import logging
import spacy
import numpy as np
from alibi.explainers import AnchorText

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize the alibi AnchorText explainer
explainer = AnchorText(nlp=nlp, use_unk=True)

def predict_fn(texts):
    # Replace this with your own prediction logic, e.g., using a trained model or rule-based classification
    return np.array([1 if "good" in text.lower() else 0 for text in texts])

def get_anchor_explanation(recommendations):
    explanations = []

    for idx, rec in enumerate(recommendations):
        try:
            description = rec.get("description", "")
            if not description:
                logging.warning(f"No description found for recommendation {idx + 1}")
                continue

            # Generate anchor explanation
            explanation = explainer.explain(description, predict_fn=predict_fn, threshold=0.95)

            # Extract anchor words and precision
            anchor_words = " AND ".join(explanation.anchor)
            precision = explanation.precision

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

    logging.info("Anchor explanations generated for all recommendations.")
    return json.dumps(explanations)
