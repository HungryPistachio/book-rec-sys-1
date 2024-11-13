import json
import logging
from alibi.explainers import AnchorText
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def filter_stopwords(feature_names):
    return [word for word in feature_names if word not in ENGLISH_STOP_WORDS]
# Load spaCy model once
nlp = spacy.load('en_core_web_sm')

# Initialize TfidfVectorizer and fit it once
vectorizer = TfidfVectorizer()
def get_anchor_explanation_for_recommendation(recommendation, original_feature_names):
    # Filter out stop words from feature names
    filtered_original_feature_names = filter_stopwords(original_feature_names)
    original_description = ' '.join(filtered_original_feature_names)

    # Apply the same filtering to recommendation feature names
    filtered_recommendation_features = filter_stopwords(recommendation['feature_names'])
    if not filtered_recommendation_features:
        logging.warning("Recommendation has no usable features after stop word filtering.")
        return {
            "title": recommendation.get("title", "Recommendation"),
            "anchor_words": "None",
            "precision": 0.0
        }
    original_description = ' '.join(original_feature_names)
    vectorizer.fit([original_description])
    original_vector = vectorizer.transform([original_description]).toarray()[0]
    return original_vector

def get_anchor_explanation_for_recommendation(recommendation, original_vector):
    # Create input text for TF-IDF from recommendation's feature names
    input_text = ' '.join(recommendation['feature_names'])

    # Define the predictor function based on similarity with the original vector
    def predict_fn(texts):
        text_vectors = vectorizer.transform(texts).toarray()
        similarities = np.dot(text_vectors, original_vector) / (
                np.linalg.norm(text_vectors, axis=1) * np.linalg.norm(original_vector)
        )
        # Return predictions as a numpy array
        return np.array([int(sim >= 0.5) for sim in similarities])

    # Initialize AnchorText explainer once per recommendation
    explainer = AnchorText(nlp=nlp, predictor=predict_fn)

    try:
        # Generate the anchor explanation using the input text
        explanation = explainer.explain(input_text, threshold=0.95)

        # Extract anchor words and precision score
        anchor_words = " AND ".join(explanation.data['anchor'])
        precision = explanation.data['precision']

        return {
            "title": recommendation.get("title", "Recommendation"),
            "anchor_words": anchor_words,
            "precision": precision
        }

    except Exception as e:
        logging.error(f"Error generating Anchor explanation: {e}")
        return {
            "title": recommendation.get("title", "Recommendation"),
            "anchor_words": "None",
            "precision": 0.0
        }

def get_anchor_explanation(recommendations, original_feature_names):
    explanations = []
    # Prepare the vectorizer and original vector once
    original_vector = prepare_vectorizer_and_original_vector(original_feature_names)
    for idx, rec in enumerate(recommendations):
        explanation = get_anchor_explanation_for_recommendation(rec, original_vector)
        explanations.append(explanation)
        logging.info(f"Anchor explanation generated for recommendation {idx + 1}")
    logging.info("Anchor explanations generated successfully for all recommendations.")
    return json.dumps(explanations)
