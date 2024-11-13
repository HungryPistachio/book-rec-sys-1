import json
import logging
from alibi.explainers import AnchorText
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def get_anchor_explanation(recommendations, original_feature_names):
    explanations = []

    # Create the original "description" by joining feature names
    original_description = ' '.join(original_feature_names)

    # Create an input text for TF-IDF by joining feature names for each recommendation
    vectorizer = TfidfVectorizer()
    all_descriptions = [original_description] + [' '.join(rec['feature_names']) for rec in recommendations]
    tfidf_matrix = vectorizer.fit_transform(all_descriptions).toarray()
    original_vector = tfidf_matrix[0]

    # Define the predictor function based on similarity with the original vector
    def predict_fn(texts):
        text_vectors = vectorizer.transform(texts).toarray()
        original_norm = np.linalg.norm(original_vector)
        text_norms = np.linalg.norm(text_vectors, axis=1)

        # Avoid division by zero
        valid_norms = (original_norm != 0) & (text_norms != 0)
        similarities = np.zeros(text_vectors.shape[0])  # Default similarities to zero
        similarities[valid_norms] = np.dot(text_vectors[valid_norms], original_vector) / (
            text_norms[valid_norms] * original_norm
        )
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
