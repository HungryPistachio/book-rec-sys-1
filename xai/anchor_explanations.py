import json
import logging
from alibi.explainers import AnchorText
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def get_anchor_explanation(recommendations, original_description):
    explanations = []

    # Create an input text for TF-IDF by combining feature names from original_description
    vectorizer = TfidfVectorizer()
    all_descriptions = [original_description] + [' '.join(rec['feature_names']) for rec in recommendations]
    tfidf_matrix = vectorizer.fit_transform(all_descriptions).toarray()
    original_vector = tfidf_matrix[0]

    # Define the predictor function based on similarity to the original vector
    def predict_fn(texts):
        text_vectors = vectorizer.transform(texts).toarray()
        similarities = np.dot(text_vectors, original_vector) / (np.linalg.norm(text_vectors, axis=1) * np.linalg.norm(original_vector))
        return np.array([int(sim >= 0.5) for sim in similarities])

    # Initialize AnchorText explainer with the predictor function
    explainer = AnchorText(nlp=nlp, predictor=predict_fn, use_unk=True)

    # Process each recommendation to generate explanations
    for idx, rec in enumerate(recommendations):
        try:
            # Combine feature names to create the input text
            input_text = ' '.join(rec.get("feature_names", []))

            # If feature names are missing, log a warning and skip this recommendation
            if not rec.get("feature_names"):
                logging.warning(f"Missing feature names for recommendation {idx + 1}")
                explanations.append({
                    "title": rec.get("title", f"Recommendation {idx + 1}"),
                    "anchor_words": "None",
                    "precision": 0.0
                })
                continue

            # Generate the anchor explanation using the input text
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
