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

    # Compute the original vector using TF-IDF for the original description
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([original_description] + [rec["description"] for rec in recommendations]).toarray()
    original_vector = tfidf_matrix[0]  # Vector for the original description

    # Define predictor function based on similarity with the original vector
    def predict_fn(text_vectors):
        similarities = np.dot(text_vectors, original_vector) / (
            np.linalg.norm(text_vectors, axis=1) * np.linalg.norm(original_vector))
        # Return 1 if similarity exceeds 0.5, else 0
        return np.array([int(sim >= 0.5) for sim in similarities])

    # Initialize AnchorText explainer with the predictor function
    explainer = AnchorText(nlp=nlp, predictor=predict_fn)

    # Process each recommendation for explanations
    for idx, rec in enumerate(recommendations):
        try:
            # Directly use the precomputed vectorized description
            description_vector = np.array(rec.get("vectorized_descriptions", []))

            # Ensure the vector is available
            if not description_vector.size:
                logging.warning(f"Missing vectorized description for recommendation {idx + 1}")
                explanations.append({
                    "title": rec.get("title", f"Recommendation {idx + 1}"),
                    "anchor_words": "None",
                    "precision": 0.0
                })
                continue

            # Combine feature names for anchor input
            input_text = ' '.join(rec.get("feature_names", []))

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
