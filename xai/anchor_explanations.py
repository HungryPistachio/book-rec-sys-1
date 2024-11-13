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

    # Initialize TF-IDF Vectorizer and vectorize descriptions
    vectorizer = TfidfVectorizer()
    all_descriptions = [original_description] + [' '.join(rec.get("feature_names", [])) for rec in recommendations]
    tfidf_matrix = vectorizer.fit_transform(all_descriptions).toarray()
    original_vector = tfidf_matrix[0]

    # Define predictor function, ensuring output is numpy array
    def predict_fn(texts):
        # Vectorize each text
        text_vectors = vectorizer.transform(texts).toarray()
        similarities = []

        # Calculate similarity only if both norms are non-zero
        original_norm = np.linalg.norm(original_vector)
        for text_vector in text_vectors:
            text_norm = np.linalg.norm(text_vector)
            if original_norm == 0 or text_norm == 0:
                similarities.append(0)  # No similarity if one of the vectors has zero norm
            else:
                similarity = np.dot(text_vector, original_vector) / (text_norm * original_norm)
                similarities.append(int(similarity >= 0.5))

    # Convert to numpy array as expected by the Anchor explainer
        return np.array(similarities)


    # Initialize AnchorText explainer with the predictor function
    explainer = AnchorText(nlp=nlp, predictor=predict_fn)

    # Process each recommendation to generate explanations
    for idx, rec in enumerate(recommendations):
        try:
            # Combine feature names to create input text, logging missing data
            feature_names = rec.get("feature_names", [])
            if not feature_names:
                logging.warning(f"Missing feature names for recommendation {idx + 1}")
                explanations.append({
                    "title": rec.get("title", f"Recommendation {idx + 1}"),
                    "anchor_words": "None",
                    "precision": 0.0
                })
                continue

            input_text = ' '.join(feature_names)

            # Generate the anchor explanation
            explanation = explainer.explain(input_text, threshold=0.95)

            # Extract anchor words and precision
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
