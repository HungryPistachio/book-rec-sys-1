import json
import logging
from alibi.explainers import AnchorText
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load spaCy model once
nlp = spacy.load('en_core_web_sm')

# Initialize TfidfVectorizer once
vectorizer = TfidfVectorizer()

# Function to filter out stop words
def filter_stopwords(feature_names):
    return [word for word in feature_names if word not in ENGLISH_STOP_WORDS]

# Function to prepare the vectorizer and create the original vector
def prepare_vectorizer_and_original_vector(original_feature_names):
    # Filter out stop words and create a combined description
    filtered_original_feature_names = filter_stopwords(original_feature_names)
    original_description = ' '.join(filtered_original_feature_names)

    # Fit vectorizer on the original description and create the original vector
    vectorizer.fit([original_description])
    original_vector = vectorizer.transform([original_description]).toarray()[0]
    return original_vector

def get_anchor_explanation_for_recommendation(recommendation, original_vector):
    logging.info(f"get_anchor_explanation_for_recommendation received recommendations data: {json.dumps(recommendation, indent=2)}")
    logging.info(f"get_anchor_explanation_for_recommendation received original_vector data: {json.dumps(original_vector, indent=2)}")
    # Create input text from recommendation's feature names
    input_text = ' '.join(filter_stopwords(recommendation.get('feature_names', [])))
    if not input_text:
        logging.warning("Recommendation has no usable features.")
        return {
            "title": recommendation.get("title", "Recommendation"),
            "anchor_words": "None",
            "precision": 0.0
        }

    # Define the predictor function based on similarity with the original vector
    def predict_fn(texts):
        logging.info(f"prediction function received texts data: {json.dumps(texts, indent=2)}")
        text_vectors = vectorizer.transform(texts).toarray()
        logging.info(f"prediction function changed texts to array: {json.dumps(text_vectors, indent=2)}")
        similarities = np.dot(text_vectors, original_vector) / (
                np.linalg.norm(text_vectors, axis=1) * np.linalg.norm(original_vector)
        )
        return np.array([int(sim >= 0.5) for sim in similarities])

    # Initialize AnchorText explainer
    explainer = AnchorText(nlp=nlp, predictor=predict_fn)

    try:
        # Generate the anchor explanation using the input text
        explanation = explainer.explain(input_text, threshold=0.95)
        logging.info(f"get_anchor_explanation_for_recommendation created explanation data: {json.dumps(explanation, indent=2)}")

        # Extract anchor words and precision score
        anchor_words = " AND ".join(explanation.data['anchor'])
        precision = explanation.data['precision']

        # Log the generated explanation
        logging.info(json.dumps({
            "title": recommendation.get("title", "Recommendation"),
            "anchor_words": anchor_words,
            "precision": precision
        }))

        # Return the generated explanation
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
    logging.info(f"get_anchor_explanation received recommendations data: {json.dumps(recommendations, indent=2)}")
    logging.info(f"get_anchor_explanation received original_feature_names data: {json.dumps(original_feature_names, indent=2)}")
    explanations = []
    # Prepare the vectorizer and original vector once
    original_vector = prepare_vectorizer_and_original_vector(original_feature_names)
    logging.info(f"get_anchor_explanation received original_vector data: {json.dumps(original_vector, indent=2)}")
    for idx, rec in enumerate(recommendations):
        explanation = get_anchor_explanation_for_recommendation(rec, original_vector)
        explanations.append(explanation)
        logging.info(f"Anchor explanation generated for recommendation {idx + 1}")
    logging.info("Anchor explanations generated successfully for all recommendations.")
    return json.dumps(explanations)
