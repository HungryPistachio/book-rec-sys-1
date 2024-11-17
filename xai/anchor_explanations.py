import json
import logging
from alibi.explainers import AnchorText
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load the spaCy language model
nlp = spacy.load('en_core_web_sm')

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

def get_top_features(feature_names, description_vector, top_n=10):
    """
    Get the top N features by vector weight.
    """
    feature_importance = list(zip(feature_names, description_vector))
    sorted_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
    top_features = [feature for feature, _ in sorted_features[:top_n]]
    return top_features

def get_anchor_explanation_for_recommendation(recommendation, original_feature_names):
    """
    Generate an anchor explanation with heuristic precision using Alibi's AnchorText.
    """
    feature_names = recommendation.get('feature_names', [])
    description_vector = recommendation.get('vectorized_descriptions', [])

    # Select the top 10 features for input text
    top_features = get_top_features(feature_names, description_vector, top_n=10)
    input_text = ' '.join(top_features)

    if not input_text:
        logging.warning(f"No significant features found for recommendation: {recommendation.get('title', 'Unknown')}")
        return {
            "title": recommendation.get("title", "Recommendation"),
            "anchor_words": "None",
            "precision": 0.0
        }

    # Initialize AnchorText explainer
    explainer = AnchorText(nlp=nlp, predictor=None)  # No predictor used, heuristic-based explanation

    try:
        # Generate anchors based on the input text
        explanation = explainer.explain(input_text, threshold=0.95)  # Threshold may not apply without predict_fn
        anchor_words = " AND ".join(explanation.data['anchor']) if 'anchor' in explanation.data else "None"
        precision = explanation.data.get('precision', "N/A")

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
        explanations.append(explanation)

    return json.dumps(explanations, indent=4)
