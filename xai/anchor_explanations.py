import json
import logging
from alibi.explainers import AnchorText
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy model once
nlp = spacy.load('en_core_web_sm')

# Initialize TfidfVectorizer once
vectorizer = TfidfVectorizer()


# Function to prepare the vectorizer and create the original vector
def prepare_vectorizer_and_original_vector(original_feature_names):
    original_description = ' '.join(original_feature_names)
    vectorizer.fit([original_description])
    original_vector = vectorizer.transform([original_description]).toarray()[0]

    return original_vector

def get_top_features(feature_names, description_vector, top_n=20):
    """Get the top N features by vector weight."""
    # Pair each feature with its corresponding vector value
    feature_importance = list(zip(feature_names, description_vector))
    # Sort by the absolute vector value in descending order
    sorted_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
    # Select the top N features
    top_features = [feature for feature, _ in sorted_features[:top_n]]
    logging.info(f"Top {top_n} features from get_top_features: {top_features}")
    return ' '.join(top_features)  # Join as a single input text

def get_anchor_explanation_for_recommendation(recommendation, original_vector):
    feature_names = recommendation.get('feature_names', [])
    description_vector = recommendation.get('vectorized_descriptions', [])

    # Select the top 20 features based on their vector weights
    input_text = get_top_features(feature_names, description_vector, top_n=20)
    logging.info(f"get_anchor_explanation_for_recommendation generated input_text: {input_text}")
    if not input_text:
        logging.warning("Recommendation has no usable features.")
        return {
            "title": recommendation.get("title", "Recommendation"),
            "anchor_words": "None",
            "precision": 0.0
        }

    # Define the predictor function based on similarity with the original vector
    def predict_fn(texts):
        # logging.info(f"Received texts in predict_fn: {texts}")

        # Filter out any "Hello world" or other unexpected inputs
        filtered_texts = [text for text in texts if text != "Hello world"]
        if not filtered_texts:
            logging.warning("No valid texts received in predict_fn.")
            return np.array([0] * len(texts))  # Default to "not similar" prediction for unexpected inputs

        # Vectorize the valid texts
        text_vectors = vectorizer.transform(filtered_texts).toarray()

        # Calculate norms, handle cases where text_vectors or original_vector norm might be zero
        text_norms = np.linalg.norm(text_vectors, axis=1)
        original_norm = np.linalg.norm(original_vector)

        # Only calculate similarities for non-zero norms to prevent NaN values
        valid_norms = (text_norms != 0) & (original_norm != 0)
        similarities = np.zeros(text_vectors.shape[0])  # Default similarities to zero
        if valid_norms.any():
            similarities[valid_norms] = np.dot(text_vectors[valid_norms], original_vector) / (
                    text_norms[valid_norms] * original_norm
            )

        # Generate binary predictions based on similarity threshold
        predictions = np.array([int(sim >= 0.75) for sim in similarities])

        # Log similarities for debugging
        # logging.info(f"Computed similarities: {similarities.tolist()}")
        # logging.info(f"Generated predictions: {predictions.tolist()}")

        # Pad predictions to match the original input length if needed
        return np.pad(predictions, (0, len(texts) - len(predictions)), 'constant')



    # Initialize AnchorText explainer
    explainer = AnchorText(nlp=nlp, predictor=predict_fn)

    try:
        # Generate the anchor explanation
        explanation = explainer.explain(input_text, threshold=0.30)
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
    # Ensure original_feature_names is a list of words
    if isinstance(original_feature_names, str):
        original_feature_names = original_feature_names.split()

    # logging.info(f"get_anchor_explanation received recommendations data: {json.dumps(recommendations, indent=2)}")
    # logging.info(f"get_anchor_explanation processed original_feature_names data: {original_feature_names}")

    explanations = []
    # Prepare the vectorizer and original vector once
    original_vector = prepare_vectorizer_and_original_vector(original_feature_names)
    # logging.info(f"get_anchor_explanation received original_vector data: {json.dumps(original_vector.tolist(), indent=2)}")

    for idx, rec in enumerate(recommendations):
        explanation = get_anchor_explanation_for_recommendation(rec, original_vector)
        explanations.append(explanation)
        logging.info(f"Anchor explanation generated for recommendation {idx + 1}")
        logging.info("Anchor explanations generated successfully for all recommendations.")
    return json.dumps(explanations)
