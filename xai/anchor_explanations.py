import json
import logging
from alibi.explainers import AnchorText
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Global NLP model
nlp = None

def get_spacy_model():
    """
    Load spaCy model only when needed to conserve memory.
    """
    global nlp
    if nlp is None:
        nlp = spacy.load('en_core_web_sm')
    return nlp

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

def prepare_vectorizer_and_original_vector(original_feature_names):
    """
    Prepare the vectorizer and create the original vector.
    """
    original_description = ' '.join(original_feature_names)
    vectorizer.fit([original_description])
    original_vector = vectorizer.transform([original_description]).toarray()[0]
    return original_vector

def get_top_features(feature_names, description_vector, top_n=3):
    """
    Get the top N features by vector weight.
    """
    feature_importance = list(zip(feature_names, description_vector))
    sorted_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
    top_features = [feature for feature, _ in sorted_features[:top_n]]
    return top_features

def dummy_predictor(texts):
    """
    Dummy predictor returning constant values as a NumPy array.
    """
    return np.ones(len(texts), dtype=np.int32)

def get_anchor_explanation_for_recommendation(recommendation, original_feature_names):
    feature_names = recommendation.get("feature_names", [])
    description_vector = recommendation.get("vectorized_descriptions", [])

    # Use top 10 features for input text
    top_features = get_top_features(feature_names, description_vector, top_n=10)
    input_text = ' '.join(top_features)
    logging.info(f"Input text for Anchor explanation: {input_text}")

    if not input_text:
        logging.warning(f"No significant features for recommendation: {recommendation.get('title', 'Unknown')}")
        return {
            "title": recommendation.get("title", "Recommendation"),
            "anchor_words": "None",
            "precision": 0.0
        }

    # Initialize AnchorText explainer
    explainer = AnchorText(nlp=get_spacy_model(), predictor=dummy_predictor)

    try:
        explanation = explainer.explain(
            input_text,
            threshold=0.10,  # Lower threshold
            beam_size=10,  # Increase beam search
            sample_proba=0.5  # Adjust sampling
        )
        anchor_words = " AND ".join(explanation.data.get('anchor', []))
        precision = explanation.data.get('precision', 0.0)

        logging.info(f"Generated explanation with precision: {precision}, anchors: {anchor_words}")
        return {
            "title": recommendation.get("title", "Recommendation"),
            "anchor_words": anchor_words,
            "precision": float(precision)
        }
    except Exception as e:
        logging.error(f"Error generating Anchor explanation: {e}")
        return {
            "title": recommendation.get("title", "Recommendation"),
            "anchor_words": "None",
            "precision": 0.0
        }


def get_anchor_explanation(recommendations, original_feature_names):
    """
    Generate anchor explanations for all recommendations.
    """
    explanations = []

    # Prepare the vectorizer once
    if isinstance(original_feature_names, str):
        original_feature_names = original_feature_names.split()
    prepare_vectorizer_and_original_vector(original_feature_names)

    for idx, rec in enumerate(recommendations):
        explanation = get_anchor_explanation_for_recommendation(rec, original_feature_names)
        # Convert NumPy types to native Python types for JSON serialization
        explanation = json.loads(json.dumps(explanation, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o))
        explanations.append(explanation)

    return json.dumps(explanations, indent=4)
