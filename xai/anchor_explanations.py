import json
import spacy
from alibi.explainers import AnchorText
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def get_anchor_explanation(recommendations, original_description):
    explanations = []

    # Initialize TF-IDF Vectorizer and fit it with the original and recommended descriptions
    all_descriptions = [original_description] + [rec['description'] for rec in recommendations]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_descriptions).toarray()
    original_vector = tfidf_matrix[0]

    # Define a prediction function based on vector similarity with the original description
    def predict_fn(texts):
        text_vectors = vectorizer.transform(texts).toarray()
        similarities = np.dot(text_vectors, original_vector) / (np.linalg.norm(text_vectors, axis=1) * np.linalg.norm(original_vector))
        # Return 1 if similarity exceeds threshold, otherwise 0
        return [int(sim >= 0.5) for sim in similarities]

    # Initialize AnchorText explainer
    explainer = AnchorText(nlp=nlp, predictor=predict_fn, use_unk=True)

    for idx, rec in enumerate(recommendations):
        try:
            # Retrieve description and combine feature names to create input text
            description = rec.get("description", "")
            feature_names = rec.get("feature_names", [])
            input_text = ' '.join(feature_names)

            # If description or feature names are missing, log a warning and skip to the next item
            if not description or not feature_names:
                logging.warning(f"Missing description or feature names for recommendation {idx + 1}")
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
