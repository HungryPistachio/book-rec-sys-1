import json
import spacy
from alibi.explainers import AnchorText
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize TF-IDF Vectorizer (you can reuse this across explanations)
vectorizer = TfidfVectorizer()

def get_anchor_explanation(recommendations, original_description):
    explanations = []

    # Transform descriptions into vectors for similarity comparison
    all_descriptions = [original_description] + [rec['description'] for rec in recommendations]
    tfidf_matrix = vectorizer.fit_transform(all_descriptions).toarray()
    original_vector = tfidf_matrix[0]

    def predictor(texts):
        # Vectorize each text and compute similarity with the original description vector
        text_vectors = vectorizer.transform(texts).toarray()
        similarities = np.dot(text_vectors, original_vector) / (np.linalg.norm(text_vectors, axis=1) * np.linalg.norm(original_vector))
        # Return 1 if similarity exceeds threshold, otherwise 0
        return [int(sim >= 0.5) for sim in similarities]

    # Initialize AnchorText explainer with the predictor function
    explainer = AnchorText(nlp, predictor=predictor, use_unk=True)

    for idx, rec in enumerate(recommendations):
        try:
            description = rec.get("description", "")
            if not description:
                logging.warning(f"No description found for recommendation {idx + 1}")
                continue

            # Generate the Anchor explanation
            explanation = explainer.explain(description, threshold=0.95)

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
