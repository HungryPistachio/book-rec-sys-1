import json
import spacy
from anchor import anchor_text
import logging

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize AnchorText explainer
explainer = anchor_text.AnchorText(nlp, ['Relevant', 'Not Relevant'], use_unk_distribution=True)

def get_anchor_explanation(recommendations):
    explanations = []

    for idx, rec in enumerate(recommendations):
        try:
            description = rec.get("description", "")
            if not description:
                logging.warning(f"No description found for recommendation {idx + 1}")
                continue

            # Generate the Anchor explanation
            explanation = explainer.explain_instance(
                description,
                predict_fn,
                threshold=0.95
            )

            # Get anchor words and precision score
            anchor_words = " AND ".join(explanation.names())
            precision = explanation.precision()

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

def predict_fn(texts):
    # Dummy prediction function for demonstration; should return 1 for 'relevant' and 0 otherwise.
    return [1 if "good" in text.lower() else 0 for text in texts]
